# Policy RL Summarization — Cortex Judge Variant

## Overview

Trains Qwen3-0.6B to summarize customer support interactions in compliance with a multi-rule policy using RL (PPO via AReaL). The reward function is decomposed into deterministic checks (~60%) and LLM-judged scores (~40%) via the **Snowflake Cortex COMPLETE API**.

Data is stored in Snowflake tables and queried at runtime via the SPCS REST API — no custom stage mounts needed.

## Prerequisites

- **Snowflake connection**: `preprod8` (account: `NOTEBOOK_MLTEST`, org: `SFENGINEERING`)
- **Database/Schema**: `RL_TRAINING_DB.RL_SCHEMA`
- **Compute Pool**: `RL_A100_POOL` (8x A100-40GB GPUs, single node)
- **Docker Image**: `areal-fresh:v5` in `/rl_training_db/rl_schema/rl_images/`
- **External Access Integrations**: `RL_TRAINING_EAI`, `ALLOW_ALL_INTEGRATION`
- **W&B Secret**: `rl_training_db.rl_schema.wandb_api_key_secret`
- **Data tables**: `POLICY_RL_TRAIN` (1000 rows), `POLICY_RL_EVAL` (100 rows)

## Files

| File | Purpose |
|------|---------|
| `run_policy_rl.py` | Entrypoint. Queries tables via REST API, builds metadata lookup, patches AReaL reward timeout, runs PPOTrainer, exports model. |
| `config_policy_rl.yaml` | AReaL config: model (Qwen3-0.6B), 8-GPU allocation (`vllm:d4p1t1+d4p1t1`), 10 epochs, reward settings. |
| `policy_reward.py` | Decomposed reward function: deterministic checks + Cortex LLM judge (`SNOWFLAKE.CORTEX.COMPLETE`). |
| `prepare_data.py` | One-time script to upload JSONL data from local disk to Snowflake tables. |
| `submit_job.py` | Submits training as an ML Job via `snowflake.ml.jobs.submit_directory`. |
| `__init__.py` | Package init (makes `policy_rl_cortex_judges` importable for reward function). |

## How the Cortex judge works

The reward function (`policy_reward.py`) calls `SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b', ...)` via the Snowflake SQL REST API to score 5 subjective criteria (tone, tier rules, product rules, accuracy, quality) on a 1-5 scale. Scores are normalized to [-1, 1] and combined with deterministic sub-scores using a weighted composite.

Key design decisions:
- **Retry + discard**: Cortex API failures retry 3x with exponential backoff. If all retries fail, the trajectory is discarded (not scored as 0) to avoid training on bad rewards.
- **Pickle safety**: All exceptions are raised as `RuntimeError(...) from None` to break unpicklable exception chains from `urllib` (required for `ProcessPoolExecutor`).
- **Timeout patch**: `run_policy_rl.py` patches AReaL's `reward_api.py` at startup to increase timeout from 15s to 600s and change the timeout handler from `return 0` to `raise`.

## Step 1: Upload Data (one-time)

```bash
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python prepare_data.py \
  --train-file /path/to/train.jsonl --eval-file /path/to/eval.jsonl
```

This creates/overwrites two tables:
- `POLICY_RL_TRAIN` — 1000 rows (columns: `TRANSCRIPT`, `METADATA`, `PROMPT`)
- `POLICY_RL_EVAL` — 100 rows (same columns)

## Step 2: Submit the Job

```bash
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python submit_job.py --no-wait
```

This uses the ML Jobs API (`snowflake.ml.jobs.submit_directory`) which:
- Uploads the directory to `@RL_PAYLOAD_STAGE`, mounted at `/mnt/job_stage/app/` inside the container
- Overrides the container `command`/`args` to use the AReaL image's Python
- Sets env vars for Cortex judge (`CORTEX_JUDGE_MODEL`, `CORTEX_WAREHOUSE`)

## Step 3: Monitor

### Job status
```sql
SELECT SYSTEM$GET_SERVICE_STATUS('RL_TRAINING_DB.RL_SCHEMA.<JOB_NAME>');
```

### Live logs
```sql
CALL SYSTEM$GET_SERVICE_LOGS('RL_TRAINING_DB.RL_SCHEMA.<JOB_NAME>', '0', 'main', 500);
```

### W&B dashboard
`https://snowflake.wandb.io/thongnguyen/spcs-policy-rl`

Run name is set by `trial_name` in `config_policy_rl.yaml` (e.g. `policy_v11`).

## Step 4: Retrieve the Trained Model

```sql
LIST @RL_TRAINING_DB.RL_SCHEMA.RL_PAYLOAD_STAGE PATTERN='.*model_output.*';
```

## Troubleshooting

### Cortex API pickle errors (`cannot pickle 'BufferedReader'`)
All exceptions in `_execute_cortex_sql` must use `raise RuntimeError(...) from None`. The `from None` breaks the exception chain so `ProcessPoolExecutor` can pickle the error across process boundaries.

### Rollout timeout after 600s
Usually means too many trajectories failed reward computation. Check Cortex API fail rate in logs: `[policy_reward] Cortex API FAIL #N (X% fail rate)`.

### REST API returns fewer rows than expected
The Snowflake SQL API paginates results. `run_policy_rl.py` fetches all partitions automatically.

### Job stuck at PENDING
Compute pool may be at capacity. Check: `DESCRIBE COMPUTE POOL RL_A100_POOL;`
