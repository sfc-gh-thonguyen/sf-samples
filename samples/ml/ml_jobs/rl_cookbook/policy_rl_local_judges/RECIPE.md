# Policy RL Summarization — Local vLLM Judge Variant

## Overview

Trains Qwen3-1.7B to summarize customer support interactions as **structured JSON** in compliance with a multi-rule policy using RL (PPO via AReaL). The reward function is decomposed into deterministic checks (~60%) and LLM-judged scores (~40%) via **4 local vLLM servers** running Qwen3-8B on dedicated GPUs with round-robin load balancing.

This variant eliminates the Cortex API dependency, reducing judge latency from ~10-30s to ~1-2s per call.

## GPU Layout (8x A100-40GB)

| GPU | Role |
|-----|------|
| 0-3 | AReaL: 2 vLLM rollout replicas (dp=2, tp=1) + 2 FSDP training (dp=2) for Qwen3-1.7B |
| 4-7 | 4x vLLM judge servers running Qwen3-8B (~16GB BF16 each) |

## Prerequisites

- **Snowflake connection**: `preprod8` (account: `NOTEBOOK_MLTEST`, org: `SFENGINEERING`)
- **Database/Schema**: `RL_TRAINING_DB.RL_SCHEMA`
- **Compute Pool**: `RL_LOCAL_JUDGE_POOL` (8x A100-40GB GPUs, single node)
- **Docker Image**: `areal-fresh:v7` in `/rl_training_db/rl_schema/rl_images/` (AReaL v1.0.2 at `/AReaL/src`)
- **External Access Integrations**: `RL_TRAINING_EAI`, `ALLOW_ALL_INTEGRATION`
- **W&B Secret**: `rl_training_db.rl_schema.wandb_api_key_secret`
- **Data tables**: `POLICY_RL_TRAIN` (1000 rows), `POLICY_RL_EVAL` (100 rows)
- **Data source**: `/code/users/thonguyen/datagen/data/` (`train.jsonl`, `eval.jsonl`)

## Files

| File | Purpose |
|------|---------|
| `run_policy_rl.py` | Entrypoint. Starts 4 vLLM judges on GPUs 4-7, queries tables, patches AReaL reward API, runs PPOTrainer, exports model. |
| `config_policy_rl.yaml` | AReaL config: Qwen3-1.7B model, 4-GPU allocation (`vllm:d2p1t1+d2p1t1`), 10 epochs. |
| `policy_reward.py` | Decomposed reward function: deterministic checks + local vLLM judge via OpenAI-compatible HTTP API. |
| `prepare_data.py` | Uploads JSONL data from local disk to Snowflake tables. **Always use this to update data.** |
| `submit_job.py` | Submits training as an ML Job via `snowflake.ml.jobs.submit_directory`. |
| `__init__.py` | Package init (makes `policy_rl_local_judges` importable for reward function). |

## Output Format (JSON)

The model is trained to produce structured JSON output. The prompt instructs the model to return a JSON object with these fields:

```json
{
  "issue_summary": "<concise description of the customer's issue(s)>",
  "customer_context": "<account tier, tenure, sentiment, relevant history>",
  "interaction_timeline": "<key events in chronological order>",
  "resolution_status": "<one of: RESOLVED, ESCALATED, PENDING_CUSTOMER, PENDING_ENGINEERING, UNRESOLVED>",
  "severity": "<one of: SEV1_CRITICAL, SEV2_HIGH, SEV3_MEDIUM, SEV4_LOW>",
  "action_items": ["<action 1>", "<action 2>"],
  "internal_notes": "<internal-only observations; PII redacted with [EMAIL], [PHONE], [SSN], [CARD], [ADDRESS], [DOB], [CREDENTIAL]>"
}
```

The `json_validity` reward component (weight 0.10) incentivizes valid JSON with all required fields and correct enum values. Other deterministic scorers (`structure_compliance`, `severity_accuracy`, `length_compliance`) are JSON-aware — they parse the response as JSON when possible and fall back to text-based checks during early training.

## Reward Components

| Component | Weight | Type | What it checks |
|-----------|--------|------|----------------|
| `json_validity` | 0.10 | Deterministic | Valid JSON, all 7 required keys, valid enums for `resolution_status` and `severity`, `action_items` is a list |
| `pii_redaction` | 0.20 | Deterministic | PII from metadata not leaked in raw form (emails, phones, cards, etc.) |
| `structure_compliance` | 0.10 | Deterministic | All required JSON fields present with non-empty content |
| `severity_accuracy` | 0.10 | Deterministic | Severity classification matches expected (derived from metadata) |
| `prohibited_content` | 0.10 | Deterministic | No internal names, prohibited phrases, raw SQL, or dollar amounts |
| `tone_compliance` | 0.10 | LLM Judge | Neutral, professional language — no blame, no naming individuals |
| `tier_specific_rules` | 0.05 | LLM Judge | Tier-appropriate elements (CSM name for Strategic, etc.) |
| `product_specific_rules` | 0.05 | LLM Judge | Product-appropriate details (volume/format for data loading, etc.) |
| `factual_accuracy` | 0.10 | LLM Judge | Accurately represents transcript without fabrication |
| `policy_justification_quality` | 0.10 | LLM Judge | Overall coherence, completeness, professionalism |

All components are logged to W&B as `rollout/reward/<component>`.

## How the local judge works

1. **`run_policy_rl.py`** starts 4 vLLM OpenAI-compatible servers on GPUs 4-7 (`CUDA_VISIBLE_DEVICES=4..7`, ports 38899-38902) before initializing AReaL. It polls `/health` on all 4 in parallel until ready (up to 10 minutes).

2. **`policy_reward.py`** sends judge requests via **native async HTTP** (aiohttp) with transcript-hash routing for prefix caching. Responses are parsed from `choices[0].message.content`.

3. **Ray sees only 4 GPUs** (`NUM_GPUS=4`), so AReaL uses GPUs 0-3 only. The port patch excludes all judge ports (38899-38902).

Key design decisions:
- **Ports 38899-38902**: Within SPCS safe range (12031-50000).
- **Transcript-hash routing**: All 8 samples from the same prompt hit the same judge server for prefix caching.
- **Native async**: No ProcessPoolExecutor — `async_policy_reward_fn` runs directly in the asyncio event loop via patched `AsyncRewardWrapper`.
- **No semaphore**: Removed to avoid bottleneck; vLLM handles queuing internally.
- **atexit cleanup**: All 4 judge processes are killed on exit or fatal error.

## Step 1: Prepare Data

Source data lives in `/code/users/thonguyen/datagen/data/` as JSONL files with keys: `transcript`, `metadata`, `prompt`.

**To update prompts or data, always edit the source JSONL files first, then re-upload:**

```bash
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python prepare_data.py \
  --train-file /code/users/thonguyen/datagen/data/train.jsonl \
  --eval-file /code/users/thonguyen/datagen/data/eval.jsonl
```

This truncates and re-creates two tables:
- `POLICY_RL_TRAIN` — 1000 rows (columns: `TRANSCRIPT`, `METADATA`, `PROMPT`)
- `POLICY_RL_EVAL` — 100 rows (same columns)

**Do not modify the Snowflake tables directly** — the JSONL files are the source of truth.

## Step 2: Submit the Job

```bash
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python submit_job.py --no-wait
```

This uses the ML Jobs API (`snowflake.ml.jobs.submit_directory`) which:
- Uploads the directory to `@RL_PAYLOAD_STAGE`, mounted at `/mnt/job_stage/app/` inside the container
- Requests 8 GPUs (4 for judges + 4 for AReaL)
- Sets env vars `LOCAL_JUDGE_MODEL`, `LOCAL_JUDGE_PORTS`, `LOCAL_JUDGE_BASE_PORT`

### Warm-starting from a checkpoint

To warm-start from a previous run's checkpoint, add `INIT_MODEL_STAGE_PATH` to `env_vars` in `submit_job.py`:

```python
"INIT_MODEL_STAGE_PATH": "@rl_training_db.rl_schema.rl_payload_stage/<job_prefix>/output/<model_output_dir>/checkpoints/root/policy-rl-summarization/<trial>/default/<checkpoint>",
```

This downloads the checkpoint via presigned URLs at startup and overrides `actor.path`, `ref.path`, `vllm.model`, and `tokenizer_path`.

## Step 3: Monitor

### Job status
```sql
SELECT SYSTEM$GET_SERVICE_STATUS('RL_TRAINING_DB.RL_SCHEMA.<JOB_NAME>');
```

### Event table logs
```sql
SELECT TIMESTAMP, VALUE
FROM snowflake.telemetry.events
WHERE RESOURCE_ATTRIBUTES:"snow.service.name" = '<JOB_NAME>'
  AND TIMESTAMP > DATEADD('minute', -10, CURRENT_TIMESTAMP())
  AND VALUE LIKE '%pattern%'
ORDER BY TIMESTAMP DESC
LIMIT 20;
```

Key log milestones:
1. `All 4 judges ready in Xs` — all 4 judge servers started on GPUs 4-7
2. `Metadata lookup: 1100 entries` — data loaded from Snowflake tables
3. `Patched AsyncRewardWrapper` — async reward pipeline active
4. `Ray started: 4.0 GPUs available` — Ray cluster up (4 GPUs, not 8)
5. `Loaded 1000 records from POLICY_RL_TRAIN` — HF Dataset created
6. `[async_policy_reward] Scores: {...}` — reward function active with per-component scores

### W&B dashboard
`https://snowflake.wandb.io/thongnguyen/spcs-policy-rl`

Run name is set by `trial_name` in `config_policy_rl.yaml` (e.g. `local_judge_v20`).

## Step 4: Retrieve the Trained Model

```sql
LIST @RL_TRAINING_DB.RL_SCHEMA.RL_PAYLOAD_STAGE PATTERN='.*model_output_local_judge.*';
```

## Version History

| Version | Key changes |
|---------|-------------|
| v8-v12 | Initial local judge setup, 5 judges on GPUs 3-7, 3 GPUs for AReaL |
| v13 | Removed retry mechanism, reduced timeouts |
| v14-v16 | Rearranged judge prompt for prefix caching, transcript-hash routing |
| v17 | Removed semaphore, native async HTTP reward pipeline, 4 judges on GPUs 4-7, 4 GPUs for AReaL |
| v18-v19 | Warm-start from v17 checkpoint, per-component reward logging to W&B |
| v20 | JSON output format, `json_validity` reward component, JSON-aware scoring, train from scratch |

## Troubleshooting

### vLLM judge server fails to start
Check logs for `vLLM judge process exited with code X`. Common causes: model download failure (needs external access integration), OOM (shouldn't happen — Qwen3-8B is ~16GB on A100-40GB).

### Judge returns no parseable JSON
The judge prompt asks for `{"tone": N, ...}` format. If the model wraps it in markdown or extra text, the regex extraction `\{[^}]+\}` should handle it. If scores are consistently missing, the model may need `temperature: 0.1` (already set).

### severity_accuracy flat at -0.5
Before v20, the prompt had no output format instructions, so the model never produced severity labels — `score_severity_accuracy` returned -0.5 (the `found is None` path). v20 fixes this with explicit JSON schema in the prompt.

### json_validity starts at -1.0
Expected for early training. The base Qwen3-1.7B model doesn't output valid JSON from the start. The reward signal should drive it to learn the format within the first few epochs.

### Job stuck at PENDING
Compute pool may be at capacity. Drop any previous running jobs first:
```sql
DROP SERVICE IF EXISTS RL_TRAINING_DB.RL_SCHEMA.<OLD_JOB_NAME>;
```
Then check: `DESCRIBE COMPUTE POOL RL_LOCAL_JUDGE_POOL;`
