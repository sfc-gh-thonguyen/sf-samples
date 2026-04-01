# RL Training Cookbook for SPCS

This cookbook provides two approaches for Reinforcement Learning (RL) training on Snowpark Container Services (SPCS) using AReaL.

## Approaches

### 1. [Verifiable Reward](./verifiable_reward/)

Uses **programmatic verification** to compute rewards. Best for tasks with objectively correct answers.

**Example**: GSM8K math problems
- Model generates step-by-step solution
- Reward function extracts and verifies the numerical answer
- Score: 1.0 if correct, 0.0 if wrong

**Pros**: Fast, deterministic, no additional compute needed
**Cons**: Only works for tasks with verifiable answers

### 2. [LLM-as-Judge](./llm_as_judges/)

Uses a **separate LLM** to evaluate output quality. Best for subjective or nuanced tasks.

**Example**: Medical SOAP note generation
- Model generates SOAP note from doctor-patient dialogue
- Judge LLM (Qwen3-8B) evaluates each section
- Score: Average of section pass/fail verdicts

**Pros**: Works for any task, captures nuanced quality
**Cons**: Requires additional GPU resources for judge

## Quick Comparison

| Aspect | Verifiable Reward | LLM-as-Judge |
|--------|------------------|--------------|
| Reward computation | Programmatic | LLM inference |
| Speed | Fast | Slower |
| GPU overhead | None | 2+ GPUs for judge |
| Task types | Math, code, factual | Open-ended, creative |
| Consistency | 100% deterministic | High (low temperature) |

## Prerequisites

1. **SPCS Compute Pool** with A100 GPUs
2. **AReaL Docker image** with flash-attn pre-installed
3. **External Access Integration** for HuggingFace/W&B

## Common Setup

### Create Compute Pool (if needed)

```sql
CREATE COMPUTE POOL RL_A100_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_L;  -- A100 80GB
```

### Create Stage for Code

```sql
CREATE STAGE IF NOT EXISTS RL_STAGE
  DIRECTORY = (ENABLE = TRUE)
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

### Upload Cookbook

```bash
# Upload verifiable reward cookbook
snow stage copy verifiable_reward @RL_STAGE/verifiable_reward --overwrite

# Upload LLM-as-judge cookbook
snow stage copy llm_as_judges @RL_STAGE/llm_as_judges --overwrite
```

## Key Technical Notes

### flash_attention_2 is Required

AReaL packs multiple sequences into a single batch for efficiency. This requires `flash_attention_2` to respect sequence boundaries:

```yaml
actor:
  attn_impl: flash_attention_2  # REQUIRED for packed sequences
```

**Do NOT use** `sdpa` or `eager` - they ignore `cu_seqlens` boundaries and cause cross-sequence attention contamination.

### SPCS Port Range

SPCS restricts cross-pod communication to ports 12031-50000. The training scripts automatically patch AReaL's port allocation.

### GPU Allocation

Both cookbooks use the allocation mode string to specify GPU distribution:

```yaml
# Format: sglang:d<data>p<pipe>t<tensor>+d<data>p<pipe>t<tensor>
allocation_mode: sglang:d2p1t1+d2p1t1  # 2 SGLang + 2 training = 4 GPUs
```

## Monitoring

Both approaches support W&B logging:

```yaml
stats_logger:
  wandb:
    mode: online
    project: spcs-areal
```

View at: https://snowflake.wandb.io/

## Directory Structure

```
rl_cookbook/
├── README.md                    # This file
├── verifiable_reward/           # GSM8K math verification
│   ├── README.md
│   ├── config.yaml              # 4 GPU config
│   ├── job_spec.yaml            # SPCS spec
│   └── run_training.py          # Entrypoint
└── llm_as_judges/               # LLM judge reward
    ├── README.md
    ├── config.yaml              # 8 GPU config (6 train + 2 judge)
    ├── job_spec.yaml            # SPCS spec
    ├── run_training.py          # Entrypoint with judge server
    ├── soap_workflow.py         # Custom workflow
    ├── prepare_data.py          # Data upload script
    └── data/
        ├── prompt_utils.py      # Judge prompts
        ├── synthetic_train_data.json
        └── synthetic_test_data.json
```
