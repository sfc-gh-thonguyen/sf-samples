#!/usr/bin/env python3
"""
Decomposed Reward Function for Customer Interaction Summarization.
(Local vLLM Judge variant — no Cortex API dependency)

Scores candidate summaries against a multi-rule policy using:
  - ~60% deterministic checks (PII, structure, severity, prohibited content, length)
  - ~40% LLM-judged scores (tone, tier rules, product rules, accuracy, quality)

The LLM judge uses 5 local vLLM servers (Qwen3-8B) on GPUs 3-7, accessed via
OpenAI-compatible HTTP API with round-robin load balancing across ports.

Compatible with AReaL's RLVRWorkflow: signature matches
    reward_fn(prompt, response, ground_truth, data, **kwargs) -> float
"""
import asyncio
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

# When ENABLE_LLM_JUDGE=0, skip all judge calls and use only deterministic rewards.
# Phase 1 (no judge) trains faster on structural signal; phase 2 adds the judge back.
ENABLE_LLM_JUDGE = os.environ.get("ENABLE_LLM_JUDGE", "1") == "1"

# ============================================================================
# Metadata Lookup
# ============================================================================
_METADATA_LOOKUP = None


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


# ============================================================================
# <think>/<answer> Tag Extraction
# ============================================================================
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_answer_content(response):
    """Extract content inside <answer>...</answer> tags from model response.

    Returns the inner content if <answer> tags are present, otherwise returns
    the full response stripped of any <think> blocks (backward compatibility
    for early training before the model learns the format).
    """
    m = _ANSWER_RE.search(response)
    if m:
        return m.group(1).strip()
    # Fallback: strip <think> tags and return whatever remains
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def _get_metadata(prompt, ground_truth, data):
    """Extract metadata from available sources (with fallbacks)."""
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

    if ground_truth:
        try:
            meta = json.loads(ground_truth)
            if isinstance(meta, dict) and "pii" in meta:
                return meta
        except (json.JSONDecodeError, TypeError):
            pass

    global _METADATA_LOOKUP
    if _METADATA_LOOKUP is None:
        _load_metadata_lookup()

    transcript = _extract_transcript(prompt)
    key = hashlib.md5(transcript[:500].encode()).hexdigest()
    return _METADATA_LOOKUP.get(key, {})


# ============================================================================
# Policy Document (embedded — source of truth is /datagen/policy.md)
# ============================================================================

_POLICY_DOCUMENT = """\
# Customer Interaction Summarization Policy v2.1

**Classification: INTERNAL**
**Effective Date: January 1, 2026**
**Policy Owner: VP of Customer Experience**

---

## Purpose

All customer support interactions (chat transcripts, call logs, email threads) must be summarized before entering downstream systems (CRM, analytics, escalation queues). Summaries must comply with this policy to protect customer privacy, ensure consistent communication standards, and support accurate operational reporting.

This policy defines the rules that govern how summaries are generated. Any automated summarization system must demonstrably comply with every applicable rule.

---

## 1. Required Structure

Every summary must contain the following sections in this exact order. Omitting or reordering sections is a violation.

| # | Section | Description |
|---|---------|-------------|
| 1 | **Issue Summary** | One to two sentences describing the core problem. Must lead with the customer's experience, not the technical root cause. |
| 2 | **Customer Context** | Account tier (Free/Pro/Enterprise/Strategic), tenure, and any relevant history mentioned in the interaction. |
| 3 | **Interaction Timeline** | Chronological sequence of key events during the interaction. Use bullet points. |
| 4 | **Resolution Status** | One of: `RESOLVED`, `ESCALATED`, `PENDING_CUSTOMER`, `PENDING_ENGINEERING`, `UNRESOLVED`. Must reflect the actual outcome, not the agent's intent. |
| 5 | **Action Items** | Concrete next steps with owners (customer, support, engineering). If none, state "No action items." |
| 6 | **Internal Notes** | Optional. Agent observations, known related issues, or context for the next handler. |

---

## 2. PII Redaction Rules

Summaries must never contain raw customer PII. Violations of this section are treated as **critical**.

| Rule | PII Type | Required Handling |
|------|----------|-------------------|
| 2.1 | Email addresses | Replace with `[EMAIL]` |
| 2.2 | Phone numbers | Replace with `[PHONE]` |
| 2.3 | Social Security / national ID numbers | Replace with `[SSN]` |
| 2.4 | Credit card / payment card numbers | Replace with `[CARD]` |
| 2.5 | Physical addresses (street-level or more specific) | Replace with `[ADDRESS]`. City/state/country may be retained. |
| 2.6 | Passwords, API keys, tokens, secrets | Replace with `[CREDENTIAL]` |
| 2.7 | Date of birth | Replace with `[DOB]` |
| 2.8 | Customer full name | Retain **only** the first name. Replace last name with the initial followed by a period (e.g., "Jane D."). If only a first name is given, retain as-is. |

If a PII element appears in a quoted customer statement, the quote must still be redacted. There are no exceptions.

---

## 3. Severity Classification

Every summary must assign a severity label based on the **customer impact**, not the technical complexity. Apply the **highest matching** severity.

| Severity | Criteria |
|----------|----------|
| `SEV1_CRITICAL` | Complete service outage, data loss, security breach, or financial impact exceeding $10,000. Applies regardless of account tier. |
| `SEV2_HIGH` | Major feature degradation affecting the customer's core workflow. OR any issue affecting a Strategic-tier account regardless of scope. |
| `SEV3_MEDIUM` | Non-core feature issue, intermittent errors, or performance degradation. OR any unresolved issue older than 48 hours. |
| `SEV4_LOW` | Cosmetic issues, documentation questions, feature requests, or general inquiries. |

**Escalation trigger**: `SEV1_CRITICAL` and `SEV2_HIGH` issues that are not `RESOLVED` must include the phrase "Escalation recommended" in the Resolution Status section.

---

## 4. Tone and Language Rules

### 4.1 Sentiment Neutralization

Summaries must use **neutral, professional language** regardless of the customer's tone during the interaction. Specific rules:

- **Never** reproduce the customer's emotional language (e.g., do NOT write "customer was furious" or "customer expressed extreme frustration"). Instead, use calibrated descriptors:
  - Mild dissatisfaction -> "customer expressed concern"
  - Moderate frustration -> "customer indicated dissatisfaction with the experience"
  - Anger / hostility -> "customer communicated urgency regarding the issue"
  - Profanity or abuse -> "customer used strong language" (never reproduce the actual words)

### 4.2 Blameless Language

- **Never** attribute fault to a specific team, individual, or system by name. Write "the issue was caused by a configuration error" not "the billing team misconfigured the account."
- **Never** name individual support agents. Write "the support agent" or "the handling agent."
- **Never** use language that implies negligence: avoid "failed to," "neglected," "oversight." Use "the issue was not addressed in the initial interaction" instead.

### 4.3 Prohibited Phrases

The following phrases or patterns must never appear in a summary:

| Prohibited | Replacement |
|-----------|-------------|
| "It's a known issue" / "known bug" | "This issue has been previously reported and is being tracked." |
| "There's nothing we can do" | "Available options within current capabilities were explored." |
| "This is user error" / "customer's fault" | "The issue originated from the account configuration." |
| "We dropped the ball" / admission of fault | Describe what happened factually without judgment. |
| Internal project codenames (e.g., "Project Falcon") | Use the customer-facing product name only. |
| Internal system/service names (e.g., "snowpipe-prod-3", "ingestion-worker-east") | Replace with `[INTERNAL_SYSTEM]` or describe the function generically (e.g., "the data ingestion service"). |

---

## 5. Account Tier-Specific Rules

The customer's account tier affects summarization requirements.

### 5.1 Strategic Accounts

- The Customer Context section **must** include the assigned Customer Success Manager's name (first name and last initial only).
- Any `PENDING_ENGINEERING` or `ESCALATED` status must include a target response time: "Target response: 4 business hours."
- The Issue Summary must explicitly state the account tier: "Strategic-tier customer reported..."

### 5.2 Enterprise Accounts

- Any `PENDING_ENGINEERING` or `ESCALATED` status must include: "Target response: 1 business day."
- If the interaction mentions contract terms, SLA, or renewal, append to Internal Notes: "Contract/SLA reference detected - flag for account management review."

### 5.3 Pro Accounts

- Standard summarization rules apply. No additional requirements.

### 5.4 Free Accounts

- The Internal Notes section must include: "Free-tier account - verify feature availability before escalation."
- Feature requests from Free accounts must be tagged: "Feature request (Free tier - not covered under current plan)."

---

## 6. Product-Specific Rules

### 6.1 Data Loading / Ingestion Issues

- If the interaction involves data loading failures, the Interaction Timeline must include the approximate data volume mentioned by the customer (e.g., "Customer reported failure loading ~50GB dataset").
- If the customer mentions specific file formats, include them in the Issue Summary.

### 6.2 Query Performance Issues

- If the interaction involves slow queries, include the reported query duration and the customer's expected duration in the Interaction Timeline (e.g., "Query taking ~45 minutes, customer expects <5 minutes").
- Do NOT include actual SQL queries in the summary. Write "Customer provided a sample query for analysis" and note this in Action Items if engineering review is needed.

### 6.3 Access / Authentication Issues

- If the interaction involves login failures or access issues, **never** include the specific error codes or authentication tokens in the summary.
- Note the authentication method if mentioned (SSO, username/password, key-pair) in the Customer Context section.

### 6.4 Billing / Account Issues

- **Never** include specific dollar amounts from invoices or quotes in the summary. Use relative terms: "unexpected increase in charges," "billing discrepancy for the current period."
- If a refund or credit is discussed, note "billing adjustment discussed" in Action Items without specifying amounts.

---

## 7. Multi-Issue Interactions

When a single interaction covers multiple distinct issues:

1. Create a **separate Issue Summary** for each issue, numbered (e.g., "Issue 1: ...", "Issue 2: ...").
2. Each issue gets its own Severity Classification.
3. The Resolution Status reflects the **overall** interaction outcome. If any issue is unresolved, the overall status cannot be `RESOLVED`.
4. The **highest** severity among all issues becomes the interaction's primary severity.
5. Maximum of 3 issues per summary. If the interaction covers more than 3 distinct issues, summarize the top 3 by severity and add to Internal Notes: "Additional issues discussed - review full transcript."

---

## 8. Length Constraints

| Section | Constraint |
|---------|-----------|
| Issue Summary | 1-3 sentences |
| Customer Context | 1-2 sentences |
| Interaction Timeline | 3-8 bullet points |
| Resolution Status | 1 sentence + escalation note if applicable |
| Action Items | 1-5 bullet points |
| Internal Notes | 0-3 sentences |
| **Total summary** | **150-400 words** |

Summaries exceeding 400 words must be revised for conciseness. Summaries under 150 words likely omit required content.\
"""

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

REQUIRED_JSON_KEYS = [
    "issue_summary", "customer_context", "interaction_timeline",
    "resolution_status", "action_items", "internal_notes",
]


def _try_parse_json(response):
    """Try to parse JSON from a model response, handling markdown code fences.

    Returns (parsed_dict, True) on success, (None, False) on failure.
    Tries multiple strategies: direct parse, fence stripping, {}-extraction.
    """
    text = response.strip()

    # Strategy 1: Strip markdown code fences if present
    clean = text
    if clean.startswith("```"):
        lines = clean.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        clean = "\n".join(lines).strip()

    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed, True
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract first { ... last } from full response
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, True
        except (json.JSONDecodeError, ValueError):
            pass

    return None, False


def score_json_validity(summary, metadata):
    """Check JSON validity with graduated scoring for partial attempts.

    Tier 1: Valid JSON parse     → [~-1.0, 1.0] based on field completeness
           - Preamble penalty: up to -0.6 for 200+ chars before first {
           - Duplicate penalty: -0.3 per extra JSON block
    Tier 2: Has {} with keys     → [-0.2, 0.4] based on key count
    Tier 3: Has ```json + keys   → [-0.5, 0.0] based on key count
    Tier 4: Some key names found → [-0.8, -0.4] based on key count
    Tier 5: No JSON structure    → -1.0
    """
    parsed, ok = _try_parse_json(summary)

    # --- Tier 1: Valid JSON parse (original logic) ---
    if ok:
        score = 1.0
        penalty_per_field = 0.2
        for key in REQUIRED_JSON_KEYS:
            if key not in parsed:
                score -= penalty_per_field

        res = parsed.get("resolution_status", "")
        if isinstance(res, str) and res.upper() not in [r for r in VALID_RESOLUTIONS]:
            score -= 0.15

        ai = parsed.get("action_items")
        if ai is not None and not isinstance(ai, list):
            score -= 0.1

        # Penalize preamble text before the JSON object.
        # Clean JSON should start immediately with { — prose wastes tokens.
        # Steeper ramp: -0.6 cap over 200 chars (was -0.4 over 500).
        text = summary.strip()
        brace_pos = text.find("{")
        if brace_pos > 0:
            preamble_len = len(text[:brace_pos].strip())
            if preamble_len > 0:
                score -= min(0.6, preamble_len / 200.0 * 0.6)

        # Penalize duplicate JSON blocks — model should emit exactly one.
        # Strategy: count ```json fences, and count how many separate valid
        # JSON objects can be extracted (try parsing after removing the first one).
        fence_count = text.lower().count("```json") + text.lower().count("``` json")
        if fence_count >= 2:
            score -= 0.3 * (fence_count - 1)
        else:
            # No fence duplication — check for repeated { } objects.
            # Find the first valid JSON span, then check if there's another after it.
            first_brace = text.find("{")
            if first_brace >= 0:
                # Walk from first { and find its matching } by brace counting
                depth = 0
                end_of_first = -1
                for i in range(first_brace, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end_of_first = i
                            break
                if end_of_first >= 0:
                    after_first = text[end_of_first + 1:].strip()
                    # If there's another { after the first complete object, it's a duplicate
                    if after_first.startswith("{") or after_first.startswith("```"):
                        score -= 0.3

        return max(-1.0, min(1.0, score))

    # --- Soft scoring for partial JSON attempts ---
    text = summary.strip()
    n_keys = len(REQUIRED_JSON_KEYS)  # 7
    key_hits = sum(1 for k in REQUIRED_JSON_KEYS if f'"{k}"' in text)

    # Tier 2: Has { } wrapping with required keys inside
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        inner = text[brace_start:brace_end + 1]
        inner_key_hits = sum(1 for k in REQUIRED_JSON_KEYS if f'"{k}"' in inner)
        if inner_key_hits >= 4:
            # Range: [-0.2, 0.4] — rewards structured attempts
            return -0.2 + (inner_key_hits / n_keys) * 0.6

    # Tier 3: Has ```json fence with some required keys
    has_fence = "```json" in text or "```\n" in text
    if has_fence and key_hits >= 3:
        # Range: [-0.5, 0.0] — rewards JSON fence + key awareness
        return -0.5 + (key_hits / n_keys) * 0.5

    # Tier 4: Contains some required key names as quoted strings
    if key_hits >= 1:
        # Range: [-0.8, -0.4] — minimal credit for key awareness
        return -0.8 + (key_hits / n_keys) * 0.4

    # Tier 5: No JSON structure at all
    return -1.0

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

    for field, _token in PII_TOKENS.items():
        value = pii.get(field)
        if not value:
            continue
        total_checks += 1
        value_str = str(value).lower()
        if value_str in summary_lower:
            raw_leaks += 1

    last_name = pii.get("customer_last_name", "")
    if last_name and len(last_name) > 1:
        total_checks += 1
        if last_name.lower() in summary_lower:
            initial_pattern = f"{last_name[0]}."
            if initial_pattern.lower() not in summary_lower:
                raw_leaks += 1
            else:
                name_count = summary_lower.count(last_name.lower())
                if name_count > 0:
                    minor_issues += 1

    if total_checks == 0:
        return 1.0

    if raw_leaks > 0:
        return -1.0
    score = 1.0 - (minor_issues / total_checks) * 0.5
    return max(-1.0, min(1.0, score))


def score_structure_compliance(summary, metadata):
    """Check all required fields present with non-empty content. Weight: 0.10

    JSON path: checks for required keys in parsed dict.
    Fallback: text-based section header search (for early training).
    """
    parsed, ok = _try_parse_json(summary)

    if ok:
        # JSON path — check required keys have non-empty values
        # Map JSON keys to REQUIRED_SECTIONS for consistency
        json_to_section = {
            "issue_summary": "Issue Summary",
            "customer_context": "Customer Context",
            "interaction_timeline": "Interaction Timeline",
            "resolution_status": "Resolution Status",
            "action_items": "Action Items",
            "internal_notes": "Internal Notes",
        }
        score = 1.0
        for key in json_to_section:
            val = parsed.get(key)
            if val is None or (isinstance(val, str) and len(val.strip()) < 5):
                score -= 1.0 / len(json_to_section)
            elif isinstance(val, list) and len(val) == 0:
                score -= 1.0 / len(json_to_section)

        # Check action_items has substantive entries
        ai = parsed.get("action_items")
        if isinstance(ai, list) and len(ai) > 0:
            if all(isinstance(x, str) and len(x.strip()) < 5 for x in ai):
                score -= 0.1

        return max(-1.0, min(1.0, score))

    # Fallback: text-based section header search
    summary_lower = summary.lower()
    score = 1.0
    last_pos = -1

    for section in REQUIRED_SECTIONS:
        section_lower = section.lower()
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
            score -= 1.0 / len(REQUIRED_SECTIONS)
        else:
            if pos < last_pos:
                score -= 0.05
            last_pos = pos

    summary_upper = summary.upper()
    has_valid_resolution = any(res in summary_upper for res in VALID_RESOLUTIONS)
    if not has_valid_resolution:
        score -= 0.15

    return max(-1.0, min(1.0, score))


def _derive_expected_severity(metadata):
    """Derive expected severity from metadata per policy Section 3."""
    product = metadata.get("product_area", "")
    tier = metadata.get("account_tier", "")
    resolution = metadata.get("resolution", "")
    hours = metadata.get("interaction_hours_since_first_report", 0)
    billing_amount = metadata.get("mentions_billing_amount")
    is_feature_request = metadata.get("is_feature_request", False)

    if billing_amount:
        try:
            amount = float(re.sub(r"[^0-9.]", "", str(billing_amount)))
            if amount > 10000:
                return "SEV1_CRITICAL"
        except (ValueError, TypeError):
            pass

    if tier == "Strategic" and not is_feature_request:
        return "SEV2_HIGH"

    if product in ("Data Loading", "Query Performance"):
        if resolution in ("UNRESOLVED", "ESCALATED", "PENDING_ENGINEERING"):
            return "SEV2_HIGH"

    if hours > 48 and resolution in ("UNRESOLVED", "PENDING_ENGINEERING"):
        return "SEV3_MEDIUM"
    if product in ("Access & Authentication", "Billing & Account"):
        return "SEV3_MEDIUM"

    if is_feature_request:
        return "SEV4_LOW"

    return "SEV3_MEDIUM"


def score_severity_accuracy(summary, metadata):
    """Check severity classification matches expected. Weight: 0.10

    Smooth scoring to provide GRPO gradient signal:
        exact match  → +1.0
        off-by-1     → +0.3   (was -0.5 — the 1.5-point cliff killed learning)
        off-by-2     → -0.3
        off-by-3     → -0.7
        not found    → -0.6   (worse than off-by-1, better than off-by-3)

    Escalation penalty halved to -0.15 (was -0.3) so it doesn't dominate.
    """
    expected = _derive_expected_severity(metadata)

    # JSON path: extract severity field directly
    parsed, ok = _try_parse_json(summary)
    found = None
    if ok:
        sev_val = parsed.get("severity", "")
        if isinstance(sev_val, str):
            sev_normalized = sev_val.upper().replace(" ", "_")
            if sev_normalized in SEVERITY_LEVELS:
                found = sev_normalized

    # Fallback: substring search in raw text
    if found is None:
        summary_upper = summary.upper().replace(" ", "_")
        for sev in SEVERITY_LEVELS:
            if sev in summary_upper:
                found = sev
                break

    if found is None:
        for abbrev, full in [("SEV1", "SEV1_CRITICAL"), ("SEV2", "SEV2_HIGH"),
                             ("SEV3", "SEV3_MEDIUM"), ("SEV4", "SEV4_LOW")]:
            if abbrev in summary.upper():
                found = full
                break

    if found is None:
        return -0.6

    expected_idx = SEVERITY_LEVELS.index(expected)
    found_idx = SEVERITY_LEVELS.index(found)
    diff = abs(expected_idx - found_idx)

    # Smooth gradient: exact → off-by-1 is only 0.7 gap (was 1.5)
    if diff == 0:
        score = 1.0
    elif diff == 1:
        score = 0.3
    elif diff == 2:
        score = -0.3
    else:
        score = -0.7

    # Escalation penalty: halved so severity signal isn't swamped
    resolution = metadata.get("resolution", "")
    if expected in ("SEV1_CRITICAL", "SEV2_HIGH") and resolution != "RESOLVED":
        if "escalation" not in summary.lower():
            score -= 0.15

    return max(-1.0, min(1.0, score))


def score_prohibited_content(summary, metadata):
    """Check for internal names, prohibited phrases, SQL, dollars. Weight: 0.10"""
    violations = 0
    summary_lower = summary.lower()

    for phrase in PROHIBITED_PHRASES:
        if phrase in summary_lower:
            violations += 1

    internal_system = metadata.get("agent_leaks_internal_system")
    if internal_system and internal_system.lower() in summary_lower:
        violations += 1

    codename = metadata.get("agent_leaks_project_codename")
    if codename and codename.lower() in summary_lower:
        violations += 1

    if metadata.get("mentions_sql_query"):
        if re.search(
            r"\b(SELECT|INSERT|UPDATE|DELETE)\b.*\b(FROM|WHERE|SET|INTO)\b",
            summary, re.IGNORECASE,
        ):
            violations += 1

    if metadata.get("mentions_billing_amount"):
        if re.search(r"\$[\d,]+", summary):
            violations += 1

    score = 1.0 - violations * 0.5
    return max(-1.0, min(1.0, score))


def score_length_compliance(summary, metadata):
    """Check word count in 150-400 range. Weight: 0.05

    JSON path: count words across all string values.
    Fallback: count words of raw string.
    """
    parsed, ok = _try_parse_json(summary)
    if ok:
        # Concatenate all string values for word counting
        text_parts = []
        for v in parsed.values():
            if isinstance(v, str):
                text_parts.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        text_parts.append(item)
        word_count = len(" ".join(text_parts).split())
    else:
        word_count = len(summary.split())

    if 150 <= word_count <= 400:
        return 1.0

    if word_count < 150:
        return max(-1.0, (word_count / 150) * 2.0 - 1.0)
    else:
        return max(-1.0, 1.0 - (word_count - 400) / 250 * 2.0)


def score_thinking_structure(response, metadata):
    """Check that response has <think>...</think> reasoning before the answer.

    Graduated scoring:
      <think> with substantial reasoning (>= 20 chars): +1.0
      <think> with short reasoning:                     +0.5
      No <think> tag:                                   -1.0
    """
    has_think = bool(re.search(r"<think>(.+?)</think>", response, re.DOTALL))

    if has_think:
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        think_len = len(think_match.group(1).strip()) if think_match else 0
        return 1.0 if think_len >= 20 else 0.5
    else:
        return -1.0


# ============================================================================
# LLM-Judged Sub-Scores (via local vLLM server)
# ============================================================================

_JUDGE_PORTS = os.environ.get("LOCAL_JUDGE_PORTS", "38899,38900,38901,38902,38903").split(",")
_JUDGE_URLS = [f"http://localhost:{p.strip()}/v1/chat/completions" for p in _JUDGE_PORTS]
_JUDGE_MODEL = os.environ.get("LOCAL_JUDGE_MODEL", "Qwen/Qwen3-8B")
_FIRST_JUDGE_CALL = True
_call_counter = None  # initialized per-worker on first call


def call_local_judge(summary, metadata, transcript, _max_retries=3):
    """Call local vLLM judge evaluating all 5 subjective criteria.

    Returns dict: {tone, tier, product, accuracy, quality} each in [-1, 1].
    Raises RuntimeError on failure so the trajectory is discarded.

    All exceptions are raised as RuntimeError(...) from None to ensure
    picklability across ProcessPoolExecutor boundaries.
    """
    global _FIRST_JUDGE_CALL, _call_counter

    # Seed counter from PID on first call so each ProcessPoolExecutor worker
    # starts on a different judge (avoids all 32 workers hammering judge 0)
    if _call_counter is None:
        _call_counter = os.getpid() % len(_JUDGE_URLS)

    # Round-robin across judge servers (PID-seeded for even spread across workers)
    judge_url = _JUDGE_URLS[_call_counter % len(_JUDGE_URLS)]
    _call_counter += 1

    payload = _build_judge_payload(summary, metadata, transcript)
    body = json.dumps(payload).encode("utf-8")

    if _FIRST_JUDGE_CALL:
        print(f"[policy_reward] First judge call -> {judge_url} "
              f"(model={_JUDGE_MODEL}, {len(_JUDGE_URLS)} judges)")
        _FIRST_JUDGE_CALL = False

    last_err = None
    for attempt in range(_max_retries):
        try:
            req = urllib.request.Request(
                judge_url, data=body, method="POST",
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=180)
            result = json.loads(resp.read().decode("utf-8"))

            content = result["choices"][0]["message"]["content"]

            # Strip <think>...</think> tags (Qwen3 thinking mode)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Extract JSON from response (model may wrap in markdown)
            match = re.search(r"\{[^}]+\}", content)
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

            # Got a response but no parseable JSON
            raise RuntimeError(
                f"Judge returned no parseable JSON scores. Content: {content[:200]}"
            ) from None

        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if attempt < _max_retries - 1:
                time.sleep(1)
                continue
            raise RuntimeError(
                f"Local judge HTTP {e.code} after {_max_retries} attempts"
            ) from None
        except urllib.error.URLError as e:
            last_err = str(e)
            if attempt < _max_retries - 1:
                time.sleep(1)
                continue
            raise RuntimeError(
                f"Local judge URLError after {_max_retries} attempts: {e}"
            ) from None
        except json.JSONDecodeError as e:
            last_err = str(e)
            if attempt < _max_retries - 1:
                time.sleep(1)
                continue
            raise RuntimeError(
                f"Local judge JSON parse error after {_max_retries} attempts: {e}"
            ) from None
        except RuntimeError:
            raise  # already a RuntimeError, re-raise as-is
        except Exception as e:
            # Catch-all: convert to picklable RuntimeError
            raise RuntimeError(
                f"Local judge unexpected {type(e).__name__}: {e}"
            ) from None

    raise RuntimeError(f"Local judge failed after {_max_retries} attempts: {last_err}")


# ============================================================================
# Async LLM Judge (native asyncio — no ProcessPoolExecutor needed)
# ============================================================================

# Module-level aiohttp session (lazy-initialized)
_aiohttp_session = None


def _build_judge_payload(summary, metadata, transcript):
    """Build the judge HTTP request payload (shared by sync and async paths)."""
    tier = metadata.get("account_tier", "Unknown")
    sentiment = metadata.get("customer_sentiment", "Unknown")
    product = metadata.get("product_area", "Unknown")
    resolution = metadata.get("resolution", "Unknown")
    num_issues = metadata.get("num_issues", "Unknown")
    csm_name = metadata.get("csm_name", "Unknown")
    mentions_contract_or_sla = metadata.get("mentions_contract_or_sla", "Unknown")
    mentions_data_volume = metadata.get("mentions_data_volume", "Unknown")
    mentions_file_format = metadata.get("mentions_file_format", "Unknown")
    mentions_billing_amount = metadata.get("mentions_billing_amount", "Unknown")
    is_feature_request = metadata.get("is_feature_request", "Unknown")
    transcript_excerpt = transcript[:3000]

    # Policy document goes first (static) for prefix caching across all calls.
    # Variable content (context, ground truth, transcript, summary) goes last.
    judge_prompt = f"""You are a compliance auditor evaluating a customer interaction summary against the company's summarization policy. The full policy is provided below. The summary is formatted as a JSON object — evaluate the CONTENT within each JSON field.

=== POLICY DOCUMENT ===
{_POLICY_DOCUMENT}
=== END POLICY DOCUMENT ===

Score how closely the summary adheres to the policy on each criterion (1=major violations, 5=fully compliant):

1. TONE: Adherence to Section 4 (Tone and Language Rules). Check sentiment neutralization (4.1), blameless language (4.2), and prohibited phrases (4.3). (1=emotional language, blame attribution, or prohibited phrases present, 5=fully compliant)
2. TIER_RULES: Adherence to Section 5 (Account Tier-Specific Rules) for a {tier} account. Check all tier-specific requirements from the applicable subsection. (1=missing all required tier elements, 5=fully compliant)
3. PRODUCT_RULES: Adherence to Section 6 (Product-Specific Rules) for {product} issues. Check all product-specific requirements from the applicable subsection. (1=missing all required product elements, 5=fully compliant)
4. ACCURACY: Adherence to Sections 1, 3, and 7. Does the summary accurately represent the transcript? Is the severity classification correct per Section 3? Are multi-issue rules from Section 7 followed? Use the GROUND TRUTH below to verify factual claims — penalize fabricated details or missing key facts. (1=fabricated/wrong, 5=fully accurate)
5. QUALITY: Adherence to Sections 1 and 8. Are all required sections present in order (Section 1)? Are length constraints met (Section 8)? Overall coherence and professionalism. (1=missing sections or grossly wrong length, 5=fully compliant)

INTERACTION CONTEXT:
- Customer sentiment: {sentiment}
- Account tier: {tier}
- Product area: {product}

GROUND TRUTH (use to verify factual accuracy):
- Resolution: {resolution}
- Number of distinct issues: {num_issues}
- CSM name: {csm_name}
- Customer mentioned contract/SLA: {mentions_contract_or_sla}
- Data volume mentioned: {mentions_data_volume}
- File format: {mentions_file_format}
- Billing amount mentioned: {mentions_billing_amount}
- Feature request: {is_feature_request}

ORIGINAL TRANSCRIPT (excerpt):
{transcript_excerpt}

SUMMARY TO EVALUATE:
{summary}

Return ONLY a JSON object with integer scores:
{{"tone": N, "tier": N, "product": N, "accuracy": N, "quality": N}}"""

    return {
        "model": _JUDGE_MODEL,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _parse_judge_response(content):
    """Parse judge response content into normalized scores dict."""
    # Strip <think>...</think> tags (Qwen3 thinking mode)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    match = re.search(r"\{[^}]+\}", content)
    if match:
        scores = json.loads(match.group())
        return {
            "tone": max(-1, min(1, (scores.get("tone", 3) - 3) / 2)),
            "tier": max(-1, min(1, (scores.get("tier", 3) - 3) / 2)),
            "product": max(-1, min(1, (scores.get("product", 3) - 3) / 2)),
            "accuracy": max(-1, min(1, (scores.get("accuracy", 3) - 3) / 2)),
            "quality": max(-1, min(1, (scores.get("quality", 3) - 3) / 2)),
        }
    return None


async def async_call_local_judge(summary, metadata, transcript):
    """Async version of call_local_judge using aiohttp.

    Runs natively in the asyncio event loop — no ProcessPoolExecutor needed.

    NO RETRIES: if the call fails or times out, raise immediately so the
    trajectory is discarded fast and the slot is freed for new requests.
    Retries hold a slot under the outer 300s timeout, starving fresh work.
    """
    global _aiohttp_session

    import aiohttp

    # Lazy-init session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout = aiohttp.ClientTimeout(total=60, sock_connect=5, sock_read=60)
        _aiohttp_session = aiohttp.ClientSession(timeout=timeout)
    payload = _build_judge_payload(summary, metadata, transcript)
    body = json.dumps(payload)

    # Route by transcript hash so all 8 samples from the same prompt hit the
    # same judge, enabling vLLM prefix cache reuse across the group.
    judge_idx = hash(transcript) % len(_JUDGE_URLS)
    judge_url = _JUDGE_URLS[judge_idx]

    try:
        async with _aiohttp_session.post(
            judge_url,
            data=body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Local judge HTTP {resp.status} from {judge_url}"
                )
            result = await resp.json()

        content = result["choices"][0]["message"]["content"]
        scores = _parse_judge_response(content)
        if scores is not None:
            return scores

        raise RuntimeError(
            f"Judge returned no parseable JSON scores. Content: {content[:200]}"
        )

    except aiohttp.ClientError as e:
        raise RuntimeError(f"Local judge connection error: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Local judge JSON parse error: {e}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Local judge unexpected {type(e).__name__}: {e}"
        )


async def async_policy_reward_fn(prompt, response, prompt_ids=None, completion_ids=None,
                                  ground_truth=None, data=None, **kwargs):
    """Async version of policy_reward_fn for direct use in asyncio event loop.

    Same scoring logic as policy_reward_fn but uses async_call_local_judge
    instead of the sync call_local_judge. This eliminates the need for
    ProcessPoolExecutor entirely.
    """
    global _CALL_COUNT, _JUDGE_SUCCESS, _JUDGE_FAIL
    _CALL_COUNT += 1

    metadata = _get_metadata(prompt, ground_truth, data)

    if not metadata:
        if _CALL_COUNT <= 3:
            print(f"[async_policy_reward] WARNING: No metadata found (call #{_CALL_COUNT})")
        answer_content = extract_answer_content(response)
        s1 = score_structure_compliance(answer_content, {})
        s2 = score_length_compliance(answer_content, {})
        fallback_reward = (s1 * 0.6 + s2 * 0.4) * 5.0
        return fallback_reward, {"structure_compliance": s1, "length_compliance": s2}

    if _CALL_COUNT <= 3:
        print(f"[async_policy_reward] Call #{_CALL_COUNT}: metadata has "
              f"tier={metadata.get('account_tier')}, "
              f"product={metadata.get('product_area')}, "
              f"pii_keys={list(metadata.get('pii', {}).keys())}")

    # Extract answer content for content-based scorers; full response for structure check
    answer_content = extract_answer_content(response)

    # --- Deterministic sub-scores ---
    scores = {
        "thinking_structure": score_thinking_structure(response, metadata),
        "json_validity": score_json_validity(answer_content, metadata),
        "pii_redaction": score_pii_redaction(answer_content, metadata),
        "structure_compliance": score_structure_compliance(answer_content, metadata),
        "severity_accuracy": score_severity_accuracy(answer_content, metadata),
        "prohibited_content": score_prohibited_content(answer_content, metadata),
        "length_compliance": score_length_compliance(answer_content, metadata),
    }

    # --- Async LLM-judged sub-scores ---
    if ENABLE_LLM_JUDGE:
        transcript = _extract_transcript(prompt)
        try:
            llm_scores = await async_call_local_judge(answer_content, metadata, transcript)
            _JUDGE_SUCCESS += 1
        except Exception:
            _JUDGE_FAIL += 1
            total = _JUDGE_SUCCESS + _JUDGE_FAIL
            pct = (_JUDGE_FAIL / total * 100) if total else 0
            print(f"[async_policy_reward] Local judge FAIL #{_JUDGE_FAIL} "
                  f"({pct:.1f}% fail rate, {_JUDGE_SUCCESS}/{total} success)")
            raise

        scores["tone_compliance"] = llm_scores["tone"]
        scores["tier_specific_rules"] = llm_scores["tier"]
        scores["product_specific_rules"] = llm_scores["product"]
        scores["factual_accuracy"] = llm_scores["accuracy"]
        scores["policy_justification_quality"] = llm_scores["quality"]
    else:
        # Phase 1: no judge, zero out judge components
        scores["tone_compliance"] = 0.0
        scores["tier_specific_rules"] = 0.0
        scores["product_specific_rules"] = 0.0
        scores["factual_accuracy"] = 0.0
        scores["policy_justification_quality"] = 0.0

    # --- Weighted composite ---
    composite = sum(ACTIVE_WEIGHTS[k] * scores[k] for k in ACTIVE_WEIGHTS)

    if _CALL_COUNT <= 3:
        phase_label = "phase2(judge)" if ENABLE_LLM_JUDGE else "phase1(deterministic)"
        print(f"[async_policy_reward] [{phase_label}] Scores: {json.dumps({k: round(v, 3) for k, v in scores.items()})}")
        print(f"[async_policy_reward] [{phase_label}] Composite: {composite:.4f} -> scaled: {composite * 10:.2f}")

    if _CALL_COUNT % _LOG_INTERVAL == 0:
        if ENABLE_LLM_JUDGE:
            total = _JUDGE_SUCCESS + _JUDGE_FAIL
            pct = (_JUDGE_FAIL / total * 100) if total else 0
            print(f"[async_policy_reward] Stats @ call {_CALL_COUNT}: "
                  f"judge {_JUDGE_SUCCESS}/{total} ok ({pct:.1f}% fail)")
        else:
            print(f"[async_policy_reward] Stats @ call {_CALL_COUNT}: "
                  f"phase1 deterministic-only (no judge)")

    return composite * 10.0, scores


# ============================================================================
# Composite Reward
# ============================================================================

WEIGHTS = {
    "thinking_structure": 0.05,
    "json_validity": 0.15,
    "pii_redaction": 0.20,
    "structure_compliance": 0.10,
    "severity_accuracy": 0.00,
    "prohibited_content": 0.10,
    "length_compliance": 0.05,
    "tone_compliance": 0.10,
    "tier_specific_rules": 0.05,
    "product_specific_rules": 0.05,
    "factual_accuracy": 0.10,
    "policy_justification_quality": 0.05,
}

# Phase 1 weights: judge components zeroed, deterministic redistributed to sum=1.0
PHASE1_WEIGHTS = {
    "thinking_structure": 0.10,
    "json_validity": 0.25,
    "pii_redaction": 0.25,
    "structure_compliance": 0.20,
    "severity_accuracy": 0.00,
    "prohibited_content": 0.10,
    "length_compliance": 0.10,
    "tone_compliance": 0.00,
    "tier_specific_rules": 0.00,
    "product_specific_rules": 0.00,
    "factual_accuracy": 0.00,
    "policy_justification_quality": 0.00,
}

ACTIVE_WEIGHTS = WEIGHTS if ENABLE_LLM_JUDGE else PHASE1_WEIGHTS

_CALL_COUNT = 0
_JUDGE_SUCCESS = 0
_JUDGE_FAIL = 0
_LOG_INTERVAL = 20


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
    global _CALL_COUNT, _JUDGE_SUCCESS, _JUDGE_FAIL
    _CALL_COUNT += 1

    metadata = _get_metadata(prompt, ground_truth, data)

    if not metadata:
        if _CALL_COUNT <= 3:
            print(f"[policy_reward] WARNING: No metadata found (call #{_CALL_COUNT})")
        answer_content = extract_answer_content(response)
        s1 = score_structure_compliance(answer_content, {})
        s2 = score_length_compliance(answer_content, {})
        return (s1 * 0.6 + s2 * 0.4) * 5.0

    if _CALL_COUNT <= 3:
        print(f"[policy_reward] Call #{_CALL_COUNT}: metadata has "
              f"tier={metadata.get('account_tier')}, "
              f"product={metadata.get('product_area')}, "
              f"pii_keys={list(metadata.get('pii', {}).keys())}")

    # Extract answer content for content-based scorers; full response for structure check
    answer_content = extract_answer_content(response)

    # --- Deterministic sub-scores ---
    scores = {
        "thinking_structure": score_thinking_structure(response, metadata),
        "json_validity": score_json_validity(answer_content, metadata),
        "pii_redaction": score_pii_redaction(answer_content, metadata),
        "structure_compliance": score_structure_compliance(answer_content, metadata),
        "severity_accuracy": score_severity_accuracy(answer_content, metadata),
        "prohibited_content": score_prohibited_content(answer_content, metadata),
        "length_compliance": score_length_compliance(answer_content, metadata),
    }

    # --- LLM-judged sub-scores (local vLLM judge) ---
    if ENABLE_LLM_JUDGE:
        transcript = _extract_transcript(prompt)
        try:
            llm_scores = call_local_judge(answer_content, metadata, transcript)
            _JUDGE_SUCCESS += 1
        except Exception:
            _JUDGE_FAIL += 1
            total = _JUDGE_SUCCESS + _JUDGE_FAIL
            pct = (_JUDGE_FAIL / total * 100) if total else 0
            print(f"[policy_reward] Local judge FAIL #{_JUDGE_FAIL} "
                  f"({pct:.1f}% fail rate, {_JUDGE_SUCCESS}/{total} success)")
            raise  # propagate → trajectory discarded

        scores["tone_compliance"] = llm_scores["tone"]
        scores["tier_specific_rules"] = llm_scores["tier"]
        scores["product_specific_rules"] = llm_scores["product"]
        scores["factual_accuracy"] = llm_scores["accuracy"]
        scores["policy_justification_quality"] = llm_scores["quality"]
    else:
        # Phase 1: no judge, zero out judge components
        scores["tone_compliance"] = 0.0
        scores["tier_specific_rules"] = 0.0
        scores["product_specific_rules"] = 0.0
        scores["factual_accuracy"] = 0.0
        scores["policy_justification_quality"] = 0.0

    # --- Weighted composite ---
    composite = sum(ACTIVE_WEIGHTS[k] * scores[k] for k in ACTIVE_WEIGHTS)

    if _CALL_COUNT <= 3:
        phase_label = "phase2(judge)" if ENABLE_LLM_JUDGE else "phase1(deterministic)"
        print(f"[policy_reward] [{phase_label}] Scores: {json.dumps({k: round(v, 3) for k, v in scores.items()})}")
        print(f"[policy_reward] [{phase_label}] Composite: {composite:.4f} -> scaled: {composite * 10:.2f}")

    if _CALL_COUNT % _LOG_INTERVAL == 0:
        if ENABLE_LLM_JUDGE:
            total = _JUDGE_SUCCESS + _JUDGE_FAIL
            pct = (_JUDGE_FAIL / total * 100) if total else 0
            print(f"[policy_reward] Stats @ call {_CALL_COUNT}: "
                  f"judge {_JUDGE_SUCCESS}/{total} ok ({pct:.1f}% fail)")
        else:
            print(f"[policy_reward] Stats @ call {_CALL_COUNT}: "
                  f"phase1 deterministic-only (no judge)")

    # Rescale from [-1, 1] to [-10, 10] for GRPO
    return composite * 10.0
