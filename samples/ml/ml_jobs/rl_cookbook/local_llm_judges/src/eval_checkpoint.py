#!/usr/bin/env python3
"""Evaluate a policy RL checkpoint by generating sample outputs.

Runs inside SPCS on 1 GPU. Downloads a checkpoint from Snowflake stage
via presigned URLs, generates summaries for eval prompts, and prints
outputs with deterministic scores.

Usage (inside container):
    python eval_checkpoint.py

Environment variables:
    CHECKPOINT_STAGE_PATH: full stage path to checkpoint dir, e.g.
        @RL_TRAINING_DB.RL_SCHEMA.RL_PAYLOAD_STAGE/run_policy_rl1c5fd0d8/output/model_output_local_judge_v17/checkpoints/root/policy-rl-summarization/local_judge_v17/default/epoch9epochstep14globalstep149
    NUM_SAMPLES: number of eval prompts to run (default: 10)
    BASE_MODEL: HuggingFace model ID to eval instead of a checkpoint (e.g. Qwen/Qwen3-1.7B)
"""
import json
import os
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_TABLE = "POLICY_RL_EVAL"
DATA_DATABASE = "RL_TRAINING_DB"
DATA_SCHEMA = "RL_SCHEMA"
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "10"))
CHECKPOINT_STAGE_PATH = os.environ.get("CHECKPOINT_STAGE_PATH", "")
BASE_MODEL = os.environ.get("BASE_MODEL", "")  # e.g. "Qwen/Qwen3-1.7B"
LOCAL_MODEL_DIR = "/tmp/eval_model"

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"


# ---------------------------------------------------------------------------
# Data loading (reuses pattern from run_policy_rl.py)
# ---------------------------------------------------------------------------
def _get_spcs_token():
    with open("/snowflake/session/token") as f:
        return f.read().strip()


def _query_eval_table(host):
    import gzip

    fq_table = f"{DATA_DATABASE}.{DATA_SCHEMA}.{EVAL_TABLE}"
    url = f"https://{host}/api/v2/statements"
    payload = {
        "statement": f"SELECT TRANSCRIPT, METADATA, PROMPT FROM {fq_table}",
        "timeout": 120,
        "resultSetMetaData": {"format": "jsonv2"},
        "warehouse": os.environ.get("CORTEX_WAREHOUSE", "ADMIN_WH"),
        "database": DATA_DATABASE,
        "schema": DATA_SCHEMA,
    }
    body = json.dumps(payload).encode("utf-8")
    ctx = ssl.create_default_context()

    def _read(resp):
        raw = resp.read()
        if raw[:2] == b'\x1f\x8b':
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))

    token = _get_spcs_token()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")

    resp = urllib.request.urlopen(req, context=ctx, timeout=120)
    result = _read(resp)

    columns = [col["name"] for col in result["resultSetMetaData"]["rowType"]]
    rows = []
    for row_data in result.get("data", []):
        rows.append(dict(zip(columns, row_data)))

    # Handle pagination
    partition_info = result.get("resultSetMetaData", {}).get("partitionInfo", [])
    statement_handle = result.get("statementHandle", "")
    for i in range(1, len(partition_info)):
        part_url = f"https://{host}/api/v2/statements/{statement_handle}?partition={i}"
        token = _get_spcs_token()
        part_req = urllib.request.Request(part_url, method="GET")
        part_req.add_header("Accept", "application/json")
        part_req.add_header("Authorization", f"Bearer {token}")
        part_resp = urllib.request.urlopen(part_req, context=ctx, timeout=120)
        part_result = _read(part_resp)
        for row_data in part_result.get("data", []):
            rows.append(dict(zip(columns, row_data)))

    print(f"  Loaded {len(rows)} eval rows")
    return rows


# ---------------------------------------------------------------------------
# SQL helper (for presigned URL downloads)
# ---------------------------------------------------------------------------
def _run_sql(statement):
    """Execute a SQL statement via Snowflake REST API and return result."""
    import gzip

    host = os.environ.get("SNOWFLAKE_HOST", "")
    url = f"https://{host}/api/v2/statements"
    payload = {
        "statement": statement,
        "timeout": 120,
        "resultSetMetaData": {"format": "jsonv2"},
        "warehouse": os.environ.get("CORTEX_WAREHOUSE", "ADMIN_WH"),
        "database": DATA_DATABASE,
        "schema": DATA_SCHEMA,
    }
    body = json.dumps(payload).encode("utf-8")
    ctx = ssl.create_default_context()
    token = _get_spcs_token()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")

    resp = urllib.request.urlopen(req, context=ctx, timeout=120)
    raw = resp.read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Checkpoint download via presigned URLs
# ---------------------------------------------------------------------------
def download_checkpoint(stage_path):
    """Download checkpoint files from Snowflake stage using presigned URLs.

    Args:
        stage_path: Full stage path like
            @RL_TRAINING_DB.RL_SCHEMA.RL_PAYLOAD_STAGE/run_policy_rl.../epoch9...

    Returns:
        Local directory path containing the downloaded model files.
    """
    if os.path.exists(LOCAL_MODEL_DIR):
        shutil.rmtree(LOCAL_MODEL_DIR)
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    print(f"  Listing files at: {stage_path}")
    list_result = _run_sql(f"LIST {stage_path}")
    files = [row[0] for row in list_result.get("data", [])]
    print(f"  Found {len(files)} files on stage")

    if not files:
        print("  ERROR: No files found at stage path")
        return None

    # Parse stage name for GET_PRESIGNED_URL
    # stage_path: @DB.SCHEMA.STAGE/path/to/dir
    # stage_name: @DB.SCHEMA.STAGE
    stage_parts = stage_path.split("/")
    stage_name = stage_parts[0]  # @DB.SCHEMA.STAGE
    stage_root = stage_name.lstrip("@").split(".")[-1].lower()  # e.g. "rl_payload_stage"

    ctx = ssl.create_default_context()
    downloaded = 0
    for stage_file in files:
        # stage_file looks like: rl_payload_stage/run_policy_rl.../model.safetensors
        # We need the relative path after the stage root
        lower_file = stage_file.lower()
        if lower_file.startswith(stage_root):
            rel_path = stage_file[len(stage_root) + 1:]
        else:
            rel_path = stage_file

        # Extract just the filename (last component)
        filename = os.path.basename(rel_path)
        if not filename:
            continue

        local_file = os.path.join(LOCAL_MODEL_DIR, filename)
        print(f"  Downloading: {filename}...", end=" ", flush=True)

        try:
            url_result = _run_sql(
                f"SELECT GET_PRESIGNED_URL({stage_name}, '{rel_path}')"
            )
            presigned_url = url_result["data"][0][0]

            # Download with SSL context
            req = urllib.request.Request(presigned_url)
            with urllib.request.urlopen(req, context=ctx, timeout=600) as resp:
                with open(local_file, "wb") as f:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)  # 8MB chunks
                        if not chunk:
                            break
                        f.write(chunk)

            size_mb = os.path.getsize(local_file) / 1e6
            print(f"{size_mb:.1f} MB")
            downloaded += 1
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"  Downloaded {downloaded}/{len(files)} files to {LOCAL_MODEL_DIR}")

    # Verify essential files exist
    essential = ["config.json"]
    has_weights = (
        os.path.exists(os.path.join(LOCAL_MODEL_DIR, "model.safetensors"))
        or any(
            f.startswith("model-") and f.endswith(".safetensors")
            for f in os.listdir(LOCAL_MODEL_DIR)
        )
    )
    has_config = os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json"))

    if not has_weights or not has_config:
        print(f"  WARNING: Missing essential files (weights={has_weights}, config={has_config})")
        print(f"  Files present: {os.listdir(LOCAL_MODEL_DIR)}")
        return None

    return LOCAL_MODEL_DIR


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_samples(model_path, prompts):
    """Generate summaries using vLLM with proper chat template."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n--- Loading model from {model_path} ---")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=6144,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Apply chat template with enable_thinking=True (matches training rollouts)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    templated_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        templated_prompts.append(text)
    print(f"  Applied chat template with enable_thinking=True to {len(templated_prompts)} prompts")

    sampling = SamplingParams(
        temperature=0.1,
        max_tokens=1200,
        top_p=0.95,
    )

    print(f"\n--- Generating {len(templated_prompts)} samples ---")
    t0 = time.time()
    outputs = llm.generate(templated_prompts, sampling)
    print(f"  Generated in {time.time() - t0:.1f}s")

    return [out.outputs[0].text for out in outputs]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_sample(response, metadata_str):
    """Run deterministic scorers on a single sample."""
    # Import scorers from policy_reward module
    sys.path.insert(0, "/mnt/job_stage/app")
    sys.path.insert(0, "/tmp/reward_modules")
    from policy_reward import (
        extract_answer_content,
        score_thinking_structure,
        score_json_validity,
        score_pii_redaction,
        score_structure_compliance,
        score_severity_accuracy,
        score_prohibited_content,
        score_length_compliance,
    )

    metadata = {}
    if metadata_str:
        try:
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        except (json.JSONDecodeError, TypeError):
            pass

    # Extract answer content (strips <think> tags, extracts <answer> block)
    answer_content = extract_answer_content(response)

    return {
        "thinking_structure": score_thinking_structure(response, metadata),
        "json_validity": score_json_validity(answer_content, metadata),
        "pii_redaction": score_pii_redaction(answer_content, metadata),
        "structure_compliance": score_structure_compliance(answer_content, metadata),
        "severity_accuracy": score_severity_accuracy(answer_content, metadata),
        "prohibited_content": score_prohibited_content(answer_content, metadata),
        "length_compliance": score_length_compliance(answer_content, metadata),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Policy RL Checkpoint Evaluation")
    print("=" * 70)
    print(f"  Checkpoint stage path: {CHECKPOINT_STAGE_PATH or '(none)'}")
    print(f"  Base model: {BASE_MODEL or '(none)'}")
    print(f"  Num samples: {NUM_SAMPLES}")

    # 1. Determine model path
    if BASE_MODEL:
        # Use a HuggingFace model directly (no checkpoint download)
        print(f"\n--- Using base model: {BASE_MODEL} ---")
        model_path = BASE_MODEL
    elif CHECKPOINT_STAGE_PATH:
        # Download checkpoint from Snowflake stage via presigned URLs
        print("\n--- Downloading Checkpoint from Stage ---")
        t0 = time.time()
        model_path = download_checkpoint(CHECKPOINT_STAGE_PATH)
        if not model_path:
            print("ERROR: Failed to download checkpoint.")
            print(f"  Stage path: {CHECKPOINT_STAGE_PATH}")
            sys.exit(1)
        print(f"  Download completed in {time.time() - t0:.1f}s")
    else:
        print("ERROR: Set CHECKPOINT_STAGE_PATH or BASE_MODEL env var.")
        sys.exit(1)

    # 2. Load eval data
    print("\n--- Loading Eval Data ---")
    host = os.environ.get("SNOWFLAKE_HOST", "")
    if not host:
        print("ERROR: SNOWFLAKE_HOST not set")
        sys.exit(1)
    rows = _query_eval_table(host)
    samples = rows[:NUM_SAMPLES]

    # 3. Generate
    prompts = [row["PROMPT"] for row in samples]
    responses = generate_samples(model_path, prompts)

    # 4. Score and display
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    all_scores = []
    for i, (row, response) in enumerate(zip(samples, responses)):
        metadata_str = row.get("METADATA", "{}")
        scores = score_sample(response, metadata_str)
        all_scores.append(scores)

        # Extract prompt excerpt (first 200 chars of transcript)
        prompt = row["PROMPT"]
        transcript_start = prompt.find("TRANSCRIPT:")
        if transcript_start >= 0:
            excerpt = prompt[transcript_start + 11:transcript_start + 211].strip()
        else:
            excerpt = prompt[:200]

        print(f"\n{'─' * 70}")
        print(f"SAMPLE {i + 1}/{len(samples)}")
        print(f"{'─' * 70}")
        print(f"PROMPT (excerpt): {excerpt}...")
        print(f"\nRESPONSE ({len(response)} chars):")
        print(response[:2000])
        if len(response) > 2000:
            print(f"  ... [{len(response) - 2000} more chars]")
        print(f"\nSCORES:")
        for k, v in scores.items():
            print(f"  {k:25s}: {v:+.3f}")

    # 5. Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    if all_scores:
        for key in all_scores[0]:
            vals = [s[key] for s in all_scores]
            avg = sum(vals) / len(vals)
            print(f"  {key:25s}: avg={avg:+.3f}  min={min(vals):+.3f}  max={max(vals):+.3f}")

    # Check JSON validity rate
    json_ok = sum(1 for s in all_scores if s["json_validity"] > -1.0)
    print(f"\n  JSON parse success: {json_ok}/{len(all_scores)} ({100 * json_ok / len(all_scores):.0f}%)")

    print(f"\n{'=' * 70}")
    print("Evaluation complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
