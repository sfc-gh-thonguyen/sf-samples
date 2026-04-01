#!/usr/bin/env python3
"""
Decomposed Reward Function for Customer Interaction Summarization.

Scores candidate summaries against a multi-rule policy using:
  - ~60% deterministic checks (PII, structure, severity, prohibited content, length)
  - ~40% LLM-judged scores (tone, tier rules, product rules, accuracy, quality)

The metadata from the dataset is the "answer key" — the reward function knows
exactly what PII was planted, what the account tier is, etc. The model never
sees the metadata; it only sees the raw transcript.

Compatible with AReaL's RLVRWorkflow: signature matches
    reward_fn(prompt, response, ground_truth, data, **kwargs) -> float
"""
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request

# ============================================================================
# Metadata Lookup
# ============================================================================
# The entrypoint (run_policy_rl.py) writes a lookup file keyed by transcript
# hash. This ensures the reward function can always find metadata regardless
# of how AReaL passes data through.

_METADATA_LOOKUP = None
_FIRST_CORTEX_CALL = True


def _load_metadata_lookup():
    global _METADATA_LOOKUP
    path = "/tmp/metadata_lookup.json"
    if os.path.exists(path):
        with open(path) as f:
            _METADATA_LOOKUP = json.load(f)
        print(f"[policy_reward] Loaded metadata lookup: {len(_METADATA_LOOKUP)} entries")
    else:
        _METADATA_LOOKUP = {}
        print(f"[policy_reward] WARNING: {path} not found, metadata lookup empty")


def _extract_transcript(prompt):
    """Extract raw transcript text from the prompt string."""
    if "TRANSCRIPT:" in prompt:
        return prompt.split("TRANSCRIPT:", 1)[-1].strip()
    return prompt


def _get_metadata(prompt, ground_truth, data):
    """Extract metadata from available sources (with fallbacks)."""
    # 1. Try data dict (if AReaL passes full record)
    if data and isinstance(data, dict):
        meta = data.get("metadata")
        if meta:
            if isinstance(meta, str):
                try:
                    return json.loads(meta)
                except json.JSONDecodeError:
                    pass
            elif isinstance(meta, dict):
                return meta
        meta_json = data.get("metadata_json")
        if meta_json and isinstance(meta_json, str):
            try:
                return json.loads(meta_json)
            except json.JSONDecodeError:
                pass

    # 2. Try ground_truth (entrypoint stores serialized metadata here)
    if ground_truth:
        try:
            meta = json.loads(ground_truth)
            if isinstance(meta, dict) and "pii" in meta:
                return meta
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Fallback to lookup by transcript hash
    global _METADATA_LOOKUP
    if _METADATA_LOOKUP is None:
        _load_metadata_lookup()

    transcript = _extract_transcript(prompt)
    key = hashlib.md5(transcript[:500].encode()).hexdigest()
    return _METADATA_LOOKUP.get(key, {})


# ============================================================================
# Deterministic Sub-Scores (each returns float in [-1, 1])
# ============================================================================

REQUIRED_SECTIONS = [
    "Issue Summary",
    "Customer Context",
    "Interaction Timeline",
    "Resolution Status",
    "Action Items",
    "Internal Notes",
]

VALID_RESOLUTIONS = [
    "RESOLVED", "ESCALATED", "PENDING_CUSTOMER",
    "PENDING_ENGINEERING", "UNRESOLVED",
]

SEVERITY_LEVELS = ["SEV1_CRITICAL", "SEV2_HIGH", "SEV3_MEDIUM", "SEV4_LOW"]

PROHIBITED_PHRASES = [
    "known issue", "known bug",
    "nothing we can do", "there's nothing we can do",
    "user error", "customer's fault", "customer error",
    "dropped the ball", "our fault", "our mistake",
    "failed to", "neglected", "oversight",
]

PII_TOKENS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "ssn": "[SSN]",
    "credit_card": "[CARD]",
    "address": "[ADDRESS]",
    "dob": "[DOB]",
    "password_or_key": "[CREDENTIAL]",
}


def score_pii_redaction(summary, metadata):
    """Check that planted PII is not leaked in raw form. Weight: 0.20"""
    pii = metadata.get("pii", {})
    if not pii:
        return 1.0

    summary_lower = summary.lower()
    total_checks = 0
    raw_leaks = 0
    minor_issues = 0

    # Check each PII element
    for field, _token in PII_TOKENS.items():
        value = pii.get(field)
        if not value:
            continue
        total_checks += 1
        value_str = str(value).lower()
        if value_str in summary_lower:
            raw_leaks += 1  # critical: raw PII leaked

    # Check last name handling (should be initial only, e.g. "Smith" -> "S.")
    last_name = pii.get("customer_last_name", "")
    if last_name and len(last_name) > 1:
        total_checks += 1
        if last_name.lower() in summary_lower:
            # Check if it's just the initial
            initial_pattern = f"{last_name[0]}."
            if initial_pattern.lower() not in summary_lower:
                raw_leaks += 1
            else:
                # Both full name and initial present — still a leak
                # unless the full name is only as part of the initial
                name_count = summary_lower.count(last_name.lower())
                if name_count > 0:
                    minor_issues += 1

    if total_checks == 0:
        return 1.0

    # Any raw PII leak is critical (-1.0), otherwise scale by issues
    if raw_leaks > 0:
        return -1.0
    score = 1.0 - (minor_issues / total_checks) * 0.5
    return max(-1.0, min(1.0, score))


def score_structure_compliance(summary, metadata):
    """Check all 6 sections present and in correct order. Weight: 0.15"""
    summary_lower = summary.lower()
    score = 1.0
    last_pos = -1

    for section in REQUIRED_SECTIONS:
        section_lower = section.lower()
        # Try multiple heading formats
        patterns = [
            section_lower,
            section_lower + ":",
            f"**{section_lower}**",
            f"## {section_lower}",
            f"### {section_lower}",
        ]
        pos = -1
        for pattern in patterns:
            idx = summary_lower.find(pattern)
            if idx >= 0:
                pos = idx
                break

        if pos < 0:
            score -= 1.0 / len(REQUIRED_SECTIONS)  # missing section
        else:
            if pos < last_pos:
                score -= 0.05  # out-of-order penalty
            last_pos = pos

    # Check Resolution Status uses one of the 5 valid enum values
    summary_upper = summary.upper()
    has_valid_resolution = any(res in summary_upper for res in VALID_RESOLUTIONS)
    if not has_valid_resolution:
        score -= 0.15

    # Check Action Items section is not empty
    action_idx = summary_lower.find("action items")
    if action_idx >= 0:
        # Text between "action items" and "internal notes" (or end)
        internal_idx = summary_lower.find("internal notes", action_idx + 12)
        end = internal_idx if internal_idx >= 0 else len(summary)
        action_text = summary[action_idx + 12:end].strip()
        if len(action_text) < 5:
            score -= 0.1

    return max(-1.0, min(1.0, score))


def _derive_expected_severity(metadata):
    """Derive expected severity from metadata per policy Section 3."""
    product = metadata.get("product_area", "")
    tier = metadata.get("account_tier", "")
    resolution = metadata.get("resolution", "")
    hours = metadata.get("interaction_hours_since_first_report", 0)
    billing_amount = metadata.get("mentions_billing_amount")
    is_feature_request = metadata.get("is_feature_request", False)

    # SEV1: financial impact > $10k, data loss, security breach
    if billing_amount:
        try:
            amount = float(re.sub(r"[^0-9.]", "", str(billing_amount)))
            if amount > 10000:
                return "SEV1_CRITICAL"
        except (ValueError, TypeError):
            pass

    # SEV2: Strategic account with non-trivial issue
    if tier == "Strategic" and not is_feature_request:
        return "SEV2_HIGH"

    # SEV2: Core workflow degradation
    if product in ("Data Loading", "Query Performance"):
        if resolution in ("UNRESOLVED", "ESCALATED", "PENDING_ENGINEERING"):
            return "SEV2_HIGH"

    # SEV3: Non-core issue or > 48h unresolved
    if hours > 48 and resolution in ("UNRESOLVED", "PENDING_ENGINEERING"):
        return "SEV3_MEDIUM"
    if product in ("Access & Authentication", "Billing & Account"):
        return "SEV3_MEDIUM"

    # SEV4: Feature requests, cosmetic, docs
    if is_feature_request:
        return "SEV4_LOW"

    return "SEV3_MEDIUM"


def score_severity_accuracy(summary, metadata):
    """Check severity classification matches expected. Weight: 0.10"""
    expected = _derive_expected_severity(metadata)
    summary_upper = summary.upper().replace(" ", "_")

    # Find severity in summary
    found = None
    for sev in SEVERITY_LEVELS:
        if sev in summary_upper:
            found = sev
            break

    if found is None:
        # Try abbreviated forms
        for abbrev, full in [("SEV1", "SEV1_CRITICAL"), ("SEV2", "SEV2_HIGH"),
                             ("SEV3", "SEV3_MEDIUM"), ("SEV4", "SEV4_LOW")]:
            if abbrev in summary_upper:
                found = full
                break

    if found is None:
        return -0.5  # no severity found at all

    expected_idx = SEVERITY_LEVELS.index(expected)
    found_idx = SEVERITY_LEVELS.index(found)
    diff = abs(expected_idx - found_idx)

    if diff == 0:
        score = 1.0
    elif diff == 1:
        score = -0.5
    else:
        score = -1.0

    # Check escalation language for unresolved SEV1/SEV2
    resolution = metadata.get("resolution", "")
    if expected in ("SEV1_CRITICAL", "SEV2_HIGH") and resolution != "RESOLVED":
        if "escalation" not in summary.lower():
            score -= 0.3

    return max(-1.0, min(1.0, score))


def score_prohibited_content(summary, metadata):
    """Check for internal names, prohibited phrases, SQL, dollars. Weight: 0.10"""
    violations = 0
    summary_lower = summary.lower()

    for phrase in PROHIBITED_PHRASES:
        if phrase in summary_lower:
            violations += 1

    # Leaked internal system names
    internal_system = metadata.get("agent_leaks_internal_system")
    if internal_system and internal_system.lower() in summary_lower:
        violations += 1

    # Leaked project codenames
    codename = metadata.get("agent_leaks_project_codename")
    if codename and codename.lower() in summary_lower:
        violations += 1

    # SQL queries in summary (should never appear)
    if metadata.get("mentions_sql_query"):
        if re.search(
            r"\b(SELECT|INSERT|UPDATE|DELETE)\b.*\b(FROM|WHERE|SET|INTO)\b",
            summary, re.IGNORECASE,
        ):
            violations += 1

    # Dollar amounts in billing context
    if metadata.get("mentions_billing_amount"):
        if re.search(r"\$[\d,]+", summary):
            violations += 1

    score = 1.0 - violations * 0.5
    return max(-1.0, min(1.0, score))


def score_length_compliance(summary, metadata):
    """Check word count in 150-400 range. Weight: 0.05"""
    word_count = len(summary.split())

    if 150 <= word_count <= 400:
        return 1.0

    if word_count < 150:
        return max(-1.0, (word_count / 150) * 2.0 - 1.0)
    else:  # > 400
        return max(-1.0, 1.0 - (word_count - 400) / 250 * 2.0)


# ============================================================================
# LLM-Judged Sub-Scores (via Cortex COMPLETE)
# ============================================================================

def _get_cortex_token():
    """Read SPCS OAuth token."""
    with open("/snowflake/session/token") as f:
        return f.read().strip()


def _execute_cortex_sql(statement, host, _max_retries=3):
    """Execute SQL via Snowflake REST API with retry on transient errors.

    Retries on: HTTP 401 (token expiry), HTTP 5xx (server errors),
    socket timeouts, and connection errors. Raises on final failure
    so the trajectory is discarded rather than scored as 0.
    """
    import socket as _socket

    global _FIRST_CORTEX_CALL, _CORTEX_RETRIES
    url = f"https://{host}/api/v2/statements"

    if _FIRST_CORTEX_CALL:
        print(f"[policy_reward] First Cortex call -> {url}")
        _FIRST_CORTEX_CALL = False

    payload = {
        "statement": statement,
        "timeout": 90,
        "resultSetMetaData": {"format": "jsonv2"},
        "warehouse": os.environ.get("CORTEX_WAREHOUSE", "ADMIN_WH"),
        "database": os.environ.get("SNOWFLAKE_DATABASE", ""),
    }
    body = json.dumps(payload).encode("utf-8")
    ctx = ssl.create_default_context()

    last_err = None
    for attempt in range(_max_retries):
        token = _get_cortex_token()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        req.add_header("Authorization", f"Bearer {token}")
        try:
            resp = urllib.request.urlopen(req, context=ctx, timeout=120)
            result = json.loads(resp.read().decode("utf-8"))

            # Handle async execution: REST API may return 202-style response
            # with statementStatusUrl instead of data. Poll until done.
            poll_url = result.get("statementStatusUrl")
            if "data" not in result and poll_url:
                if not poll_url.startswith("http"):
                    poll_url = f"https://{host}{poll_url}"
                for poll_attempt in range(30):  # up to ~60s polling
                    time.sleep(2)
                    token = _get_cortex_token()
                    poll_req = urllib.request.Request(poll_url, method="GET")
                    poll_req.add_header("Accept", "application/json")
                    poll_req.add_header("Authorization", f"Bearer {token}")
                    poll_resp = urllib.request.urlopen(poll_req, context=ctx, timeout=30)
                    result = json.loads(poll_resp.read().decode("utf-8"))
                    if "data" in result:
                        break
                    code = result.get("code", "")
                    # Still running
                    if code == "333334" or result.get("message", "").startswith("Statement"):
                        continue
                    # Error or unknown state
                    break

            return result
        except urllib.error.HTTPError as e:
            last_err = e
            retryable = e.code == 401 or e.code >= 500
            if retryable and attempt < _max_retries - 1:
                _CORTEX_RETRIES += 1
                wait = 2 ** (attempt + 1)  # 2s, 4s backoff
                print(f"[policy_reward] HTTP {e.code} on attempt {attempt+1}, "
                      f"retrying in {wait}s...")
                time.sleep(wait)
                continue
            # Re-raise as RuntimeError so it's picklable across ProcessPoolExecutor
            raise RuntimeError(f"Cortex SQL HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:300]}") from None
        except (urllib.error.URLError, _socket.timeout, OSError) as e:
            last_err = e
            if attempt < _max_retries - 1:
                _CORTEX_RETRIES += 1
                wait = 2 ** (attempt + 1)
                print(f"[policy_reward] {type(e).__name__} on attempt {attempt+1}, "
                      f"retrying in {wait}s...")
                time.sleep(wait)
                continue
            # Re-raise as RuntimeError so it's picklable across ProcessPoolExecutor
            raise RuntimeError(f"Cortex SQL {type(e).__name__}: {e}") from None
        except Exception as e:
            # Catch-all: convert any other exception (JSONDecodeError, etc.)
            # to picklable RuntimeError — urllib response objects in traceback
            # cause "cannot pickle 'BufferedReader'" in ProcessPoolExecutor
            raise RuntimeError(f"Cortex SQL unexpected {type(e).__name__}: {e}") from None
    raise RuntimeError(f"Cortex SQL failed after {_max_retries} attempts: {last_err}")


def call_cortex_judge(summary, metadata, transcript):
    """Single Cortex call evaluating all 5 subjective criteria.

    Returns dict: {tone, tier, product, accuracy, quality} each in [-1, 1].
    Raises on API failure so the trajectory is discarded (not scored as 0).
    """
    host = os.environ.get("SNOWFLAKE_HOST", "")
    model = os.environ.get("CORTEX_JUDGE_MODEL", "llama3.1-8b")

    tier = metadata.get("account_tier", "Unknown")
    sentiment = metadata.get("customer_sentiment", "Unknown")
    product = metadata.get("product_area", "Unknown")

    # Truncate transcript for judge context (keep under token limits)
    transcript_excerpt = transcript[:3000]

    judge_prompt = f"""You are evaluating a customer interaction summary against company policy. Score each criterion on a scale of 1 (worst) to 5 (best).

SUMMARY:
{summary}

CONTEXT:
- Customer sentiment in original interaction: {sentiment}
- Account tier: {tier}
- Product area: {product}

ORIGINAL TRANSCRIPT (excerpt):
{transcript_excerpt}

CRITERIA:
1. TONE: Does the summary use neutral, professional language? No emotional reproduction, no blame attribution, no naming individuals. (1=emotional/blaming, 5=perfectly neutral)
2. TIER_RULES: For a {tier} account, are tier-specific elements included? Strategic accounts need CSM name and target response time. Enterprise need target response time for escalations. Free accounts need feature availability note. (1=missing all, 5=fully compliant)
3. PRODUCT_RULES: For {product} issues, are product-specific rules followed? Data loading should mention volume/format. Query performance should mention duration without SQL. Billing should avoid dollar amounts. (1=missing all, 5=fully compliant)
4. ACCURACY: Does the summary accurately represent the transcript without fabrication or critical omissions? (1=fabricated/wrong, 5=fully accurate)
5. QUALITY: Overall coherence, completeness, and professionalism. (1=incoherent, 5=excellent)

Return ONLY a JSON object with integer scores:
{{"tone": N, "tier": N, "product": N, "accuracy": N, "quality": N}}"""

    escaped = judge_prompt.replace("\\", "\\\\").replace("'", "''")
    sql = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{escaped}')"

    # No try/except fallback — if _execute_cortex_sql raises after retries,
    # the exception propagates up through policy_reward_fn -> AsyncRewardWrapper
    # -> arun_episode -> workflow_executor, which catches it and discards the
    # trajectory (returns None). This is better than scoring a failed API call
    # as reward=0, which after reward_bias/scaling becomes an active penalty.
    result = _execute_cortex_sql(sql, host)
    if result.get("data") and len(result["data"]) > 0:
        raw_text = str(result["data"][0][0])
        # Extract JSON from response
        match = re.search(r"\{[^}]+\}", raw_text)
        if match:
            scores = json.loads(match.group())
            # Normalize 1-5 to [-1, 1]: (score - 3) / 2
            return {
                "tone": max(-1, min(1, (scores.get("tone", 3) - 3) / 2)),
                "tier": max(-1, min(1, (scores.get("tier", 3) - 3) / 2)),
                "product": max(-1, min(1, (scores.get("product", 3) - 3) / 2)),
                "accuracy": max(-1, min(1, (scores.get("accuracy", 3) - 3) / 2)),
                "quality": max(-1, min(1, (scores.get("quality", 3) - 3) / 2)),
            }

    # API returned successfully but with no parseable scores — raise so
    # the trajectory is discarded rather than trained on with a bogus reward.
    raise RuntimeError(
        f"[policy_reward] Cortex judge returned no parseable scores. "
        f"Raw result keys: {list(result.keys())}"
    )


# ============================================================================
# Composite Reward
# ============================================================================

WEIGHTS = {
    "pii_redaction": 0.20,
    "structure_compliance": 0.15,
    "severity_accuracy": 0.10,
    "prohibited_content": 0.10,
    "length_compliance": 0.05,
    "tone_compliance": 0.10,
    "tier_specific_rules": 0.05,
    "product_specific_rules": 0.05,
    "factual_accuracy": 0.10,
    "policy_justification_quality": 0.10,
}

_CALL_COUNT = 0
_CORTEX_SUCCESS = 0
_CORTEX_FAIL = 0
_CORTEX_RETRIES = 0
_LOG_INTERVAL = 20  # print stats every N calls


def policy_reward_fn(prompt, response, ground_truth=None, data=None, **kwargs):
    """Main reward function for AReaL RLVRWorkflow.

    Args:
        prompt: The input prompt (contains the transcript)
        response: Model's generated summary
        ground_truth: Optional serialized metadata JSON
        data: Full dataset record (may contain metadata)

    Returns:
        float: Composite reward scaled to [-10, 10] for GRPO
    """
    global _CALL_COUNT, _CORTEX_SUCCESS, _CORTEX_FAIL
    _CALL_COUNT += 1

    metadata = _get_metadata(prompt, ground_truth, data)

    if not metadata:
        # Without metadata, only structure and length checks work
        if _CALL_COUNT <= 3:
            print(f"[policy_reward] WARNING: No metadata found (call #{_CALL_COUNT})")
        s1 = score_structure_compliance(response, {})
        s2 = score_length_compliance(response, {})
        return (s1 * 0.6 + s2 * 0.4) * 5.0

    # Log first few calls for debugging
    if _CALL_COUNT <= 3:
        print(f"[policy_reward] Call #{_CALL_COUNT}: metadata has "
              f"tier={metadata.get('account_tier')}, "
              f"product={metadata.get('product_area')}, "
              f"pii_keys={list(metadata.get('pii', {}).keys())}")

    # --- Deterministic sub-scores ---
    scores = {
        "pii_redaction": score_pii_redaction(response, metadata),
        "structure_compliance": score_structure_compliance(response, metadata),
        "severity_accuracy": score_severity_accuracy(response, metadata),
        "prohibited_content": score_prohibited_content(response, metadata),
        "length_compliance": score_length_compliance(response, metadata),
    }

    # --- LLM-judged sub-scores (may raise on API failure → trajectory discarded) ---
    transcript = _extract_transcript(prompt)
    try:
        llm_scores = call_cortex_judge(response, metadata, transcript)
        _CORTEX_SUCCESS += 1
    except Exception:
        _CORTEX_FAIL += 1
        # Log stats on every failure so we can track the rate
        total = _CORTEX_SUCCESS + _CORTEX_FAIL
        pct = (_CORTEX_FAIL / total * 100) if total else 0
        print(f"[policy_reward] Cortex API FAIL #{_CORTEX_FAIL} "
              f"({pct:.1f}% fail rate, {_CORTEX_SUCCESS}/{total} success, "
              f"retries={_CORTEX_RETRIES})")
        raise  # propagate → trajectory discarded

    scores["tone_compliance"] = llm_scores["tone"]
    scores["tier_specific_rules"] = llm_scores["tier"]
    scores["product_specific_rules"] = llm_scores["product"]
    scores["factual_accuracy"] = llm_scores["accuracy"]
    scores["policy_justification_quality"] = llm_scores["quality"]

    # --- Weighted composite ---
    composite = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)

    # Log detailed breakdown for first few calls
    if _CALL_COUNT <= 3:
        print(f"[policy_reward] Scores: {json.dumps({k: round(v, 3) for k, v in scores.items()})}")
        print(f"[policy_reward] Composite: {composite:.4f} -> scaled: {composite * 10:.2f}")

    # Periodic stats
    if _CALL_COUNT % _LOG_INTERVAL == 0:
        total = _CORTEX_SUCCESS + _CORTEX_FAIL
        pct = (_CORTEX_FAIL / total * 100) if total else 0
        print(f"[policy_reward] Stats @ call {_CALL_COUNT}: "
              f"cortex {_CORTEX_SUCCESS}/{total} ok ({pct:.1f}% fail), "
              f"retries={_CORTEX_RETRIES}")

    # Rescale from [-1, 1] to [-10, 10] for GRPO
    return composite * 10.0
