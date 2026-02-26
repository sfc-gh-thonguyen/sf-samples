# Multi-Node RL Fine-tuning Pipeline for Snowflake SPCS

This pipeline implements GRPO (Group Relative Policy Optimization) training with LLM-as-judge evaluation for Medical SOAP note generation on Snowflake SPCS (Snowpark Container Services).

## Architecture

The training uses a 3-node architecture:
- **Node 1 (Judge)**: Runs Qwen3-8B as LLM judge for SOAP note evaluation
- **Node 2 (Rollout)**: Policy model inference using SGLang
- **Node 3 (Trainer)**: FSDP + LoRA training

## Prerequisites

1. **Snowflake Account** with:
   - GPU compute pool (e.g., `GPU_NV_M` or `GPU_NV_L`)
   - External Access Integration (EAI) for PyPI, HuggingFace, and GitHub

2. **Python Environment** with:
   - `snowflake-snowpark-python`
   - `snowflake-ml-python`

## Usage

### Step 1: Upload Training Data

Upload the synthetic training datasets to Snowflake stage:

```bash
python scripts/upload_data.py \
    --database YOUR_DATABASE \
    --schema YOUR_SCHEMA \
    --stage-name rl_payload_stage
```

This uploads `synthetic_train_data.json` and `synthetic_test_data.json` to `@rl_payload_stage/data/`.

### Step 2: Submit Training Job

Submit the multi-node RL training job:

```bash
python scripts/run_rl_train.py \
    --compute-pool GPU_NV_M \
    --external-access-integrations PYPI_EAI HF_EAI GITHUB_EAI \
    --database YOUR_DATABASE \
    --schema YOUR_SCHEMA \
    --num-nodes 3
```

### Configuration

The training configuration is in `src/grpo_lora_config.yaml`. Key settings:

- **Model**: `Qwen/Qwen3-1.7B` (policy model)
- **Judge Model**: `Qwen/Qwen3-8B`
- **LoRA**: rank=16, alpha=32, all-linear targets
- **Training**: 10 epochs, batch size 8, learning rate 3e-5

### Data Paths

On SPCS, data is available at:
- Training data: `/mnt/job_stage/data/synthetic_train_data.json`
- Test data: `/mnt/job_stage/data/synthetic_test_data.json`
- Checkpoints: `/mnt/job_stage/checkpoints/`
- Name resolve: `/mnt/job_stage/name_resolve/`

## Reward Structure

Total reward (max 5.0):
- **Format reward** (0-1): Valid JSON with S, O, A, P keys
- **Section rewards** (0-4): 1.0 each for S, O, A, P sections passing LLM judge

The LLM judge evaluates each section for:
1. Factual accuracy (no hallucination)
2. Completeness (key clinical information present)
3. Clinical appropriateness (correct section content)

## Files

```
rl_finetune/
├── scripts/
│   ├── upload_data.py       # Upload datasets to SPCS stage
│   └── run_rl_train.py      # Submit training job
├── src/
│   ├── setup_areal.sh       # Install AReaL from GitHub
│   ├── grpo_lora_config.yaml    # Training configuration
│   ├── judge_server_config.yaml # Judge server configuration
│   ├── soap_grpo_trainer.py     # Main training entry point
│   ├── dataset.py               # Dataset loading utilities
│   ├── reward.py                # Reward functions
│   └── prompt_utils.py          # Prompts for SOAP generation/evaluation
├── requirements.txt
└── README.md
```

## Monitoring

Training logs are available via:

```python
job.get_logs()
```

Statistics logged include:
- `reward`: Combined reward (0-5)
- `format_reward`: JSON format validity (0-1)
- `judge_S`, `judge_O`, `judge_A`, `judge_P`: Individual section scores
