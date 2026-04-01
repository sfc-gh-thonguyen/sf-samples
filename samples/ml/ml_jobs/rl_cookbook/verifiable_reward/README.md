# RL Fine-tuning on Snowpark Container Services (SPCS)

This cookbook demonstrates how to run GRPO (Group Relative Policy Optimization) training on SPCS using AReaL.

## Overview

- **Model**: Qwen3-0.6B (configurable)
- **Dataset**: GSM8K math problems
- **Algorithm**: GRPO with vLLM inference + FSDP training
- **Hardware**: 4x A100 GPUs (2 for inference, 2 for training)

## Prerequisites

1. SPCS compute pool with GPU nodes (GPU_NV_L recommended)
2. Container image with AReaL and flash-attn installed
3. External access integration for HuggingFace and W&B

## Quick Start

### Submit via ML Jobs API (recommended)

```bash
# Submit and return immediately (job runs in background on SPCS)
SNOWFLAKE_DEFAULT_CONNECTION_NAME=myconn python submit_job.py --no-wait

# Submit and wait for completion
SNOWFLAKE_DEFAULT_CONNECTION_NAME=myconn python submit_job.py
```

`submit_job.py` uses `snowflake.ml.jobs.submit_directory()` to upload the payload
directory to a Snowflake stage and launch a job on SPCS. It handles:

- Uploading all files in this directory to `@RL_PAYLOAD_STAGE`
- Overriding the container spec to use the custom AReaL image
- Setting environment variables (HF cache, W&B, NCCL)
- Injecting the W&B API key from a Snowflake secret

Monitor the job:

```sql
-- Check status
SELECT SYSTEM$GET_SERVICE_STATUS('RL_TRAINING_DB.RL_SCHEMA.<JOB_ID>');

-- View logs (last 500 lines)
SELECT SYSTEM$GET_SERVICE_LOGS('RL_TRAINING_DB.RL_SCHEMA.<JOB_ID>', '0', 'main', 500);
```

### Alternative: EXECUTE JOB SERVICE (manual)

```bash
# 1. Upload files to stage
snow stage copy config.yaml @MY_STAGE/rl_cookbook/verifiable_reward/
snow stage copy job_spec.yaml @MY_STAGE/rl_cookbook/verifiable_reward/
snow stage copy run_training.py @MY_STAGE/rl_cookbook/verifiable_reward/

# 2. Run the job (blocks until completion)
EXECUTE JOB SERVICE
  IN COMPUTE POOL MY_GPU_POOL
  FROM @MY_STAGE/rl_cookbook/verifiable_reward/
  SPECIFICATION_FILE = 'job_spec.yaml'
  EXTERNAL_ACCESS_INTEGRATIONS = (MY_EAI);
```

Note: `EXECUTE JOB SERVICE` blocks the SQL session for the entire training
duration (can be hours). Use `submit_job.py --no-wait` for non-blocking submission.

## Files

| File | Description |
|------|-------------|
| `submit_job.py` | ML Jobs submission script (recommended entry point) |
| `config.yaml` | GRPO training configuration |
| `run_training.py` | Training entrypoint script |
| `job_spec.yaml` | SPCS container spec (for EXECUTE JOB SERVICE) |

## Key Configuration

### Attention Implementation

**Important**: Use `flash_attention_2` when training with packed sequences:

```yaml
actor:
  attn_impl: flash_attention_2  # NOT sdpa or eager
```

Why: SDPA and eager attention ignore `cu_seqlens` boundaries in packed sequences, causing cross-sequence attention contamination. Only flash_attention_2 properly handles packed sequences via `flash_attn_varlen_func`.

### GPU Allocation

```yaml
# 4 GPUs: 2 for vLLM inference (dp=2, tp=1), 2 for FSDP training (dp=2)
allocation_mode: vllm:d2p1t1+d2p1t1
```

### Learning Rate

For bf16 training, use a learning rate that produces meaningful weight updates:
```yaml
optimizer:
  lr: 1.5e-5  # Not too small for bf16 precision
```

## Customization

### Using a Different Model

Update the model path in config.yaml:
```yaml
actor:
  path: your-org/your-model
```

### Using a Different Dataset

Modify the dataset section:
```yaml
train_dataset:
  path: your-dataset
  type: rl
```

### Adjusting GPU Count

For 8 GPUs (4 inference + 4 training):
```yaml
cluster:
  n_gpus_per_node: 8
allocation_mode: vllm:d4p1t1+d4p1t1
```

## Monitoring

### W&B Integration

`submit_job.py` injects the W&B API key from a Snowflake secret
(`rl_training_db.rl_schema.wandb_api_key_secret`). To use a different secret,
update the `secrets` section in `submit_job.py`'s `spec_overrides`.

The W&B base URL is set via `env_vars` in `submit_job.py`.

### Key Metrics to Watch

| Metric | Expected Value | Indicates |
|--------|----------------|-----------|
| `behave_imp_weight/avg` | ~1.0 | Correct logprob alignment |
| `task_reward/avg` | Increasing | Model learning |
| `entropy/avg` | Stable or slowly decreasing | Healthy exploration |
| `grad_norm` | 0.5-2.0 | Active gradient flow |

## Troubleshooting

### behave_imp_weight << 1.0

**Cause**: Using `sdpa` or `eager` attention with packed sequences.
**Fix**: Set `attn_impl: flash_attention_2` and ensure flash-attn is installed.

### W&B Not Logging / Resuming Old Run

**Cause**: Stale W&B state in fileroot directory.
**Fix**: Use a unique fileroot path for each new experiment:
```yaml
cluster:
  fileroot: /tmp/my_experiment_v2
```

### Config Error: Invalid recover mode 'False'

**Cause**: YAML interprets `off` as boolean.
**Fix**: Quote the value: `mode: "off"`
