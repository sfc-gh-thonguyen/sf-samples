# AReaL Reinforcement Learning Training on Snowflake SPCS

## Executive Summary

This document explains how to run multi-node reinforcement learning (RL) training for LLMs on Snowflake's Snowpark Container Services (SPCS). The implementation uses the **AReaL framework** (Asynchronous Reinforcement Learning) with **GRPO** (Group Relative Policy Optimization) to fine-tune models.

**Key Achievement**: Successfully running 2-node decoupled RL training where:
- **Node 1**: Runs SGLang inference servers for generating rollouts (tensor parallelism = 4)
- **Node 2**: Runs FSDP-based training with LoRA fine-tuning (data parallelism = 2)

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         SPCS Compute Pool (RL_GPU_POOL)                       │
│                         GPU_NV_M: 4× A10 GPUs per node                        │
├───────────────────────────────────┬───────────────────────────────────────────┤
│           NODE 1 (head)           │           NODE 2 (worker)                 │
│                                   │                                           │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────────────────┐  │
│  │      SGLang Inference       │  │  │        FSDP Training                │  │
│  │         (tp=4)              │  │  │          (dp=2)                     │  │
│  │                             │  │  │                                     │  │
│  │  • Tensor Parallel across   │  │  │  • Data Parallel across 4 GPUs     │  │
│  │    4 GPUs for inference     │  │  │  • LoRA fine-tuning (rank=16)      │  │
│  │  • Port 34000+ (SPCS safe)  │  │  │  • GRPO loss computation           │  │
│  │  • /generate endpoint       │  │  │  • Gradient accumulation           │  │
│  │  • /load_lora_adapter       │  │  │                                     │  │
│  └─────────────────────────────┘  │  └─────────────────────────────────────┘  │
│            ▲                      │                    │                      │
│            │                      │                    │                      │
│            │    Ray Object Store  │                    │                      │
│            └──────────────────────┼────────────────────┘                      │
│                                   │     Cross-Node Weight Sync                │
│                                   │     (LoRA weights via ray.put/get)        │
└───────────────────────────────────┴───────────────────────────────────────────┘
```

### Training Loop

1. **Rollout Generation**: Training node requests completions from SGLang via HTTP
2. **Reward Computation**: Compare model outputs against ground truth (math problems)
3. **Policy Gradient**: Compute GRPO loss using group-relative advantages
4. **Weight Update**: Trainer updates LoRA weights, syncs to SGLang via Ray object store
5. **Repeat**: SGLang uses updated weights for next rollout batch

---

## Key Components

### 1. Docker Image (`docker/Dockerfile`)

Custom image based on AReaL runtime v0.5.3 with these additions:

| Component | Purpose |
|-----------|---------|
| Ray wrapper script | Converts deprecated Ray 2.53.0 flags for ML Jobs compatibility |
| Snowflake runtime modules | Required by ML Jobs bootstrap (`snowflake.runtime`) |
| HF_HUB_DISABLE_XET=1 | Disables HuggingFace XET storage (incompatible with SPCS network) |
| cron + sysvinit-utils | Required by ML Jobs service management |

**Image Location**:
```
preprod8-notebook-mltest.awsuswest2preprod8.registry-dev.snowflakecomputing.com/
  rl_training_db/rl_schema/rl_images/areal-runtime:v0.5.3-ray253-fix6
```

### 2. Training Entry Point (`src/run_areal.py`)

Main script that:
1. Applies all patches at startup
2. Waits for Ray cluster formation
3. Registers node tags for deterministic scheduling
4. Launches AReaL training with proper configuration

### 3. Configuration (`src/fast_test_config.yaml`)

```yaml
allocation_mode: sglang:d1p1t4+d2p1t1  # Decoupled: SGLang tp=4, Training dp=2

sglang:
  enable_torch_compile: false  # Critical! Avoids 20+ min warmup
  disable_cuda_graph: true     # Speeds up startup

actor:
  weight_update_mode: disk     # Triggers Ray weight sync patch
  use_lora: true
  lora_rank: 16
```

**Allocation Mode Syntax**: `backend:d<dp>p<pp>t<tp>+d<dp>p<pp>t<tp>`
- First group: Inference (SGLang)
- Second group: Training
- `d` = data parallel, `p` = pipeline parallel, `t` = tensor parallel

### 4. Log Wrapper (`src/log_wrapper.py`)

Solves SPCS's 16KB log truncation by:
- Capturing ALL stdout/stderr to `/mnt/job_stage/logs/<job>.log`
- Filtering repetitive "Waiting for instances" messages
- Showing summary of last lines in truncated console output

---

## Critical Patches (5 Total)

### Patch 1: Ray Launcher (`ray_launcher_patched.py`)

**Problem**: AReaL's default scheduler doesn't guarantee deterministic node placement.

**Solution**: Custom Ray launcher using node tags:
```python
# Registration
ray.experimental.set_resource("node_tag:head", 1.0, head_node_id)
ray.experimental.set_resource("node_tag:worker", 1.0, worker_node_id)

# Scheduling
@ray.remote(resources={"node_tag:head": 0.01})
def sglang_server():
    ...  # Guaranteed to run on head node
```

### Patch 2: SGLang Server (`sglang_server_patched.py`)

**Problem**: SPCS blocks cross-node HTTP on ports < 30000.

**Solution**: Change base port from 10000 to 34000:
```python
SPCS_BASE_PORT = 34000  # Original was 10000
port_range = (server_idx * ports_per_server + SPCS_BASE_PORT, ...)
```

| Port Range | Cross-Node Status |
|------------|-------------------|
| < 30000    | ❌ BLOCKED        |
| 34000-50000| ✅ WORKS          |

### Patch 3: Remote Inference Engine (`remote_inf_engine_patched.py`)

**Problem**: External judge resolution times out after 1 second.

**Solution**: Increase timeout to 300s and fail explicitly (no silent fallback):
```python
JUDGE_RESOLVE_TIMEOUT = 300  # Was 1 second!
# If resolution fails, raise error instead of using policy model as judge
```

### Patch 4: Weight Update Port

**Problem**: NCCL/Gloo may use blocked ports for distributed training.

**Solution**: Force safe port range (12031-13000) for torch distributed.

### Patch 5: Ray Weight Sync (`ray_weight_sync.py`)

**Problem**: SPCS has no shared filesystem across nodes, so disk-based weight sync fails.

**Solution**: Use Ray's object store as cross-node memory transport:

```
┌─────────────────────────────┐          ┌─────────────────────────────┐
│       Trainer (Worker)      │          │      SGLang (Head)          │
│                             │          │                             │
│  1. Extract LoRA weights    │          │                             │
│  2. Convert to numpy bytes  │          │                             │
│  3. ray.put(weights) ─────────────────►│                             │
│                             │          │  4. ray.get(weights)        │
│                             │          │  5. Save to /tmp/lora/      │
│                             │          │  6. POST /load_lora_adapter │
│                             │          │                             │
│  7. Continue training       │          │  8. Use new weights for     │
│                             │          │     next rollout batch      │
└─────────────────────────────┘          └─────────────────────────────┘
```

---

## Deep Dive: Why XCCL Fails for LoRA Weight Sync

This section explains why we can't use SGLang's native XCCL (GPU-to-GPU) weight update for LoRA adapters, and why we built a Ray object store solution instead.

### Background: How LoRA Works

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to frozen pretrained weights:

```
Output = W_pretrained × input + (lora_B × lora_A) × input
                                 ↑         ↑
                              [d, r]    [r, d]   (r << d, typically r=16)
```

For a transformer's attention layer, we typically add LoRA to Q, K, V, and O projections:
- `q_proj.lora_A`, `q_proj.lora_B`
- `k_proj.lora_A`, `k_proj.lora_B`
- `v_proj.lora_A`, `v_proj.lora_B`
- `o_proj.lora_A`, `o_proj.lora_B`

### The Problem: QKV Merging in SGLang

Many modern LLMs (Qwen, Llama, Mistral) use **fused QKV projections** for efficiency. Instead of three separate weight matrices, they use one combined matrix:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINER (PEFT/HuggingFace)                           │
│                                                                         │
│    Stores Q, K, V as SEPARATE parameters:                               │
│                                                                         │
│    model.layers.0.self_attn.q_proj.lora_A  →  shape [16, 1024]         │
│    model.layers.0.self_attn.q_proj.lora_B  →  shape [1024, 16]         │
│    model.layers.0.self_attn.k_proj.lora_A  →  shape [16, 1024]         │
│    model.layers.0.self_attn.k_proj.lora_B  →  shape [1024, 16]         │
│    model.layers.0.self_attn.v_proj.lora_A  →  shape [16, 1024]         │
│    model.layers.0.self_attn.v_proj.lora_B  →  shape [1024, 16]         │
└─────────────────────────────────────────────────────────────────────────┘

                              ≠  NAME MISMATCH!

┌─────────────────────────────────────────────────────────────────────────┐
│                    SGLANG (Optimized Inference)                         │
│                                                                         │
│    Merges Q, K, V into SINGLE fused parameter:                          │
│                                                                         │
│    model.layers.0.self_attn.qkv_proj.lora_A  →  shape [16, 1024]       │
│    model.layers.0.self_attn.qkv_proj.lora_B  →  shape [3072, 16]       │
│                                                  ↑                      │
│                                          Q+K+V output dims              │
│                                          (1024 + 1024 + 1024)           │
└─────────────────────────────────────────────────────────────────────────┘
```

### XCCL Weight Update: How It Works (And Fails)

SGLang's XCCL endpoint (`/update_weights_from_distributed`) uses NCCL for GPU-direct weight transfer:

```python
# SGLang's XCCL update logic (simplified)
def update_weights_from_distributed(self, tensor_name, tensor_data):
    # Find parameter by EXACT name match
    for name, param in self.model.named_parameters():
        if name == tensor_name:  # <-- EXACT MATCH REQUIRED!
            param.data.copy_(tensor_data)
            return
    raise ValueError(f"Parameter {tensor_name} not found!")
```

**The failure**:
1. Trainer broadcasts tensor named `q_proj.lora_A`
2. SGLang searches for `q_proj.lora_A` in its parameters
3. SGLang only has `qkv_proj.lora_A` → **NAME NOT FOUND**
4. Weight update fails silently or with error

### Shape Mismatch: Why Simple Renaming Doesn't Work

Even if we rename `q_proj → qkv_proj`, the shapes don't match:

```
Trainer's q_proj.lora_B:     [1024, 16]   (Q output dimension only)
SGLang's qkv_proj.lora_B:    [3072, 16]   (Q+K+V fused output dimension)
```

To make XCCL work, we'd need to:
1. Concatenate Q, K, V lora_B matrices: `[1024,16] + [1024,16] + [1024,16] → [3072,16]`
2. Handle models with GQA (Grouped Query Attention) where K,V have different sizes
3. Know the exact concat order (varies by model architecture)

This is fragile and model-specific.

### Disk Mode: Why It Works

SGLang's `/load_lora_adapter` endpoint uses PEFT's standard loading logic:

```python
# SGLang disk loading (simplified)
def load_lora_adapter(self, path):
    # PEFT handles the name translation!
    adapter = PeftModel.from_pretrained(self.model, path)
    
    # PEFT internally maps:
    #   q_proj.lora_A → applies to Q slice of qkv_proj
    #   k_proj.lora_A → applies to K slice of qkv_proj
    #   v_proj.lora_A → applies to V slice of qkv_proj
```

PEFT knows how to map separate Q/K/V LoRA weights to fused QKV projections because:
1. It reads `target_modules` from `adapter_config.json`
2. It understands the model architecture
3. It applies LoRA to the correct slices of the fused weight

### Our Solution: Ray Object Store + Disk Mode

Since SPCS has no shared filesystem, we use Ray's object store as a transport layer, then leverage disk mode's proven name translation:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Trainer extracts LoRA weights                                       │
│                                                                              │
│    weights = {                                                               │
│      "model.layers.0.self_attn.q_proj.lora_A": tensor([16, 1024]),          │
│      "model.layers.0.self_attn.q_proj.lora_B": tensor([1024, 16]),          │
│      ...                                                                     │
│    }                                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Convert to numpy bytes (Ray serialization)                          │
│                                                                              │
│    # BFloat16 → Float32 (numpy doesn't support bf16)                         │
│    # DTensor → regular tensor (FSDP distributed tensors)                     │
│    numpy_weights = {k: tensor.numpy().tobytes() for k, tensor in weights}   │
│                                                                              │
│    ref = ray.put({"numpy_weights": numpy_weights, "adapter_config": config})│
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                        Ray Object Store Transfer
                        (automatic cross-node)
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Remote function on SGLang node                                      │
│                                                                              │
│    @ray.remote(resources={"node_tag:head": 0.01})                            │
│    def load_lora_on_sglang_node(weights_data, sglang_addr):                 │
│        # Ray auto-dereferences the ObjectRef                                 │
│        weights = reconstruct_tensors(weights_data["numpy_weights"])          │
│                                                                              │
│        # Save to local filesystem                                            │
│        safetensors.save_file(weights, "/tmp/lora/adapter_model.safetensors")│
│        json.dump(config, "/tmp/lora/adapter_config.json")                   │
│                                                                              │
│        # Call SGLang's disk loading endpoint                                 │
│        requests.post(f"http://{sglang_addr}/load_lora_adapter",             │
│                      json={"lora_path": "/tmp/lora/"})                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: PEFT handles name translation                                       │
│                                                                              │
│    q_proj.lora_A ──┐                                                         │
│    k_proj.lora_A ──┼──► PEFT applies to qkv_proj slices correctly           │
│    v_proj.lora_A ──┘                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Comparison: XCCL vs Ray Object Store + Disk

| Aspect | XCCL (GPU-Direct) | Ray + Disk Mode |
|--------|-------------------|-----------------|
| Speed | ⚡ Fastest (GPU-to-GPU) | 🐢 Slower (serialize → transfer → disk → load) |
| Name Translation | ❌ None (exact match) | ✅ PEFT handles it |
| Shape Translation | ❌ None | ✅ PEFT handles it |
| Cross-Node in SPCS | ❓ Requires NCCL setup | ✅ Ray handles it |
| Shared Filesystem | Not needed | Not needed (local /tmp/) |
| Model Compatibility | Only unfused Q/K/V | ✅ All models |

### Why Not Patch SGLang's XCCL?

We considered patching SGLang to add name translation to XCCL:

```python
# Hypothetical XCCL patch (NOT implemented)
def translate_name(trainer_name):
    if "q_proj" in trainer_name:
        return trainer_name.replace("q_proj", "qkv_proj")
    ...
```

**Problems**:
1. Still need to handle shape concatenation
2. GQA models have different K/V sizes than Q
3. Different models use different fusion strategies
4. Very fragile, breaks with model updates

The Ray + Disk approach is more robust because it delegates all the complexity to PEFT, which is already battle-tested.

---

## Challenges and Solutions

### Challenge 1: Ray 2.53.0 Flag Changes

**Symptom**: `No such option: --dashboard-grpc-port`

**Cause**: ML Jobs bootstrap uses deprecated Ray flags.

**Solution**: Wrapper script at `/AReaL/.venv/bin/ray` that:
- Converts `--dashboard-grpc-port` → `--dashboard-agent-grpc-port`
- Handles empty `--dashboard-port=` and `--dashboard-host=` flags

### Challenge 2: SPCS Port Restrictions

**Symptom**: Worker node times out reaching SGLang: `Timeout: ['10.244.x.x:17436']`

**Cause**: SPCS blocks cross-node HTTP on ports < 30000.

**Discovery Method**: Diagnostic script tested ports 15000-50000 systematically.

**Solution**: `sglang_server_patched.py` changes base port to 34000.

### Challenge 3: 16KB Log Truncation

**Symptom**: Logs show middle of execution, missing both start and end.

**Cause**: SPCS truncates from BEGINNING when logs exceed 16KB.

**Solution**: `log_wrapper.py` captures full logs to stage file:
```
/mnt/job_stage/logs/<timestamp>_<script>_<hostname>.log
```

### Challenge 4: torch.compile Timeout

**Symptom**: SGLang servers never become healthy, trainers time out.

**Cause**: torch.compile warmup takes 20+ minutes per server.

**Solution**: Disable in config:
```yaml
sglang:
  enable_torch_compile: false
  disable_cuda_graph: true
```

### Challenge 5: Cross-Node LoRA Weight Sync

**Symptom**: Training runs but SGLang never updates its LoRA weights.

**Cause**: 
1. No shared filesystem in SPCS (disk mode fails)
2. XCCL mode has parameter name mismatches (Q/K/V vs QKV)

**Solution**: Ray object store based sync:
1. Trainer extracts LoRA weights → numpy bytes
2. `ray.put()` stores in object store
3. Remote function on SGLang node does `ray.get()`
4. Saves to local `/tmp/`, calls `/load_lora_adapter`
5. SGLang's PEFT loader handles name translation

### Challenge 6: Worker Node OOM

**Symptom**: Exit code 137 after 20 minutes of training.

**Cause**: dp=4 with FSDP uses too much memory.

**Solution**: Reduce data parallelism:
```yaml
allocation_mode: sglang:d1p1t4+d2p1t1  # Changed from d4p1t1
```

---

## Job Submission

### Quick Start

```bash
# 2-node decoupled training
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python scripts/run_rl_train.py \
    --compute-pool RL_GPU_POOL \
    --external-access-integrations PYPI_HF_EAI \
    --database RL_TRAINING_DB \
    --schema RL_SCHEMA \
    --num-nodes 2 \
    --no-wait \
    --memory 150G \
    --config fast_test_config.yaml \
    --trainer simple_math_trainer.py \
    --runtime '<image-url>'
```

### Monitoring

```sql
-- View job status
SELECT SYSTEM$GET_SERVICE_STATUS('RL_TRAINING_DB.RL_SCHEMA.<JOB_ID>');

-- View logs (last 1000 lines)
SELECT SYSTEM$GET_SERVICE_LOGS('RL_TRAINING_DB.RL_SCHEMA.<JOB_ID>', 0, 'main', 1000);

-- List all jobs
SHOW SERVICES LIKE '%AREAL%' IN COMPUTE POOL RL_GPU_POOL;
```

### Access Full Logs

```sql
-- List log files on stage
LIST @RL_TRAINING_DB.RL_SCHEMA.RL_PAYLOAD_STAGE/app/logs/;

-- Download specific log
GET @RL_TRAINING_DB.RL_SCHEMA.RL_PAYLOAD_STAGE/app/logs/<filename>.log file://./
```

---

## File Structure

```
rl_finetune/
├── docker/
│   └── Dockerfile              # Custom AReaL image with all fixes
├── scripts/
│   └── run_rl_train.py         # Job submission script
├── src/
│   ├── run_areal.py            # Main entry point with patch application
│   ├── log_wrapper.py          # Log capture for SPCS
│   ├── fast_test_config.yaml   # Training configuration
│   ├── simple_math_trainer.py  # GRPO trainer for math problems
│   └── areal_patches/
│       ├── ray_launcher_patched.py       # Node tag scheduling
│       ├── sglang_server_patched.py      # SPCS port fix
│       ├── remote_inf_engine_patched.py  # Judge timeout fix
│       └── ray_weight_sync.py            # Cross-node LoRA sync
├── AGENT.md                    # Detailed technical documentation
└── RL_TRAINING_OVERVIEW.md     # This document
```

---

## Snowflake Resources

| Resource | Value |
|----------|-------|
| Connection | `preprod8` |
| Database | `RL_TRAINING_DB` |
| Schema | `RL_SCHEMA` |
| Compute Pool | `RL_GPU_POOL` (GPU_NV_M: 4× A10, 44 vCPUs, 178GB RAM per node) |
| Stage | `@RL_TRAINING_DB.RL_SCHEMA.RL_PAYLOAD_STAGE` |
| External Access | `PYPI_HF_EAI` (PyPI + HuggingFace) |

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| "Waiting for X jobs" forever | trainer_n_nodes = 0 | Use 2-node decoupled mode |
| Worker timeout to SGLang | Port < 30000 | Apply sglang_server patch |
| torch.compile timeout | Warmup too slow | Set `enable_torch_compile: false` |
| Exit code 137 | OOM | Reduce dp or add `--memory 150G` |
| "No module 'snowflake'" | Missing runtime | Use custom Docker image |
| Log truncation | 16KB limit | Use log_wrapper.py |
| LoRA weights not syncing | No shared filesystem | Ray weight sync patch |

---

## Next Steps

1. **Scale Testing**: Test with larger models (Qwen-1.7B, Qwen-7B)
2. **3-Node Mode**: Add external judge server for better reward signal
3. **Checkpointing**: Add model checkpoint export to Snowflake stage
4. **W&B Integration**: Enable experiment tracking with Weights & Biases

---

## References

- [AReaL GitHub Repository](https://github.com/inclusionAI/AReaL)
- [Snowflake ML Jobs Documentation](https://docs.snowflake.com/en/developer-guide/snowpark-ml/jobs)
- [Ray Custom Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- [SGLang Documentation](https://github.com/sgl-project/sglang)
