## 🚧 CURRENT WORK IN PROGRESS (Feb 2026)

### Active Task: E2E Test for Unified Multi-Node RL Training

**Job Running:** `LOG_WRAPPER_3Y6Z7X4C4MN9` (PENDING - waiting for GPU nodes)

**What Was Fixed:**
1. ✅ **Ray 2.53.0 `set_resource()` deprecation** - Updated to use pre-registered SPCS tags (`head`/`worker`) instead of dynamic `rollout`/`trainer` tags
2. ✅ **SGLang LoRA config** - Added `--max-lora-rank 16` and `--lora-target-modules` params required by newer SGLang

**Files Modified:**
- `src/areal_patches/ray_launcher_patched.py` - `get_rollout_node_tag()` and `get_trainer_node_tag()` now return `head`/`worker`
- `src/areal_patches/sglang_server_patched.py` - Added LoRA initialization params
- `src/run_areal.py` - Removed `ray.shutdown()`, set `RAY_ADDRESS`, fixed `ray.init(address=...)`

**To Continue:**
```bash
# Check job status
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 snow sql -q "SELECT SYSTEM\$GET_SERVICE_STATUS('RL_TRAINING_DB.RL_SCHEMA.LOG_WRAPPER_3Y6Z7X4C4MN9')"

# Get logs
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 snow sql -q "SELECT SYSTEM\$GET_SERVICE_LOGS('RL_TRAINING_DB.RL_SCHEMA.LOG_WRAPPER_3Y6Z7X4C4MN9', 0, 'main', 100)"

# If job failed, run again:
cd samples/ml/ml_jobs/rl_finetune
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python scripts/e2e_test_multinode.py \
    --compute-pool RL_GPU_POOL \
    --database RL_TRAINING_DB \
    --schema RL_SCHEMA \
    --external-access-integrations PYPI_HF_EAI \
    --runtime "preprod8-notebook-mltest.awsuswest2preprod8.registry-dev.snowflakecomputing.com/rl_training_db/rl_schema/rl_images/areal-runtime:v0.5.3-ray253-fix6" \
    --test 2node_basic
```

**Expected Success Indicators:**
1. Logs show: `Scheduling SGLang server 0 to node_tag:head` (NOT `node_tag:rollout`)
2. Logs show: `SPCS FIX: Added LoRA initialization params`
3. SGLang starts without `AssertionError` about LoRA params
4. Training progresses with loss decreasing

**If It Fails Again:**
- Check logs for new error messages
- Common issues documented in sections 14 and 15 below

---

## Unified Multi-Node Architecture (NEW!)

### Overview

The codebase has been refactored to support **N-node configurations** with a unified node allocation schema. This replaces the legacy `AREAL_3NODE_MODE` environment variable approach.

### Key Changes

| Old Approach | New Approach |
|--------------|--------------|
| `AREAL_3NODE_MODE=1` env var | `nodes` section in config YAML |
| Separate scripts (`run_rl_train.py`, `run_3node_training.py`) | Unified `run_multinode_training.py` |
| Hardcoded `use_2node_mode`/`use_3node_mode` conditionals | Dynamic role-based scheduling via `NodeAllocation` |

### Config Schema

```yaml
nodes:
  total: 2                    # Total nodes in this job
  roles:
    rollout: 1                # Nodes for SGLang inference
    trainer: 1                # Nodes for GRPO training
  external_judge:
    enabled: false            # Launch separate judge job?
    experiment_name: null     # Judge's experiment name (for name_resolve)
    trial_name: null
```

### Node Tag Generation

| Config | Node Tags Generated |
|--------|---------------------|
| `{rollout: 1, trainer: 1}` | `rollout`, `trainer` |
| `{rollout: 2, trainer: 2}` | `rollout_0`, `rollout_1`, `trainer_0`, `trainer_1` |
| `{rollout: 1, trainer: 3}` | `rollout`, `trainer_0`, `trainer_1`, `trainer_2` |

### Key Files

| File | Purpose |
|------|---------|
| `src/node_allocation.py` | NodeAllocation dataclass, parsing, validation |
| `src/run_areal.py` | Updated to use `register_node_tags_unified()` |
| `src/areal_patches/ray_launcher_patched.py` | Unified scheduling helpers |
| `src/grpo_multinode_template.yaml` | Template config with `nodes` section |
| `scripts/run_multinode_training.py` | Unified submission script |
| `src/test_node_allocation.py` | Unit tests (19 passing) |

### Backward Compatibility

Legacy configs (without `nodes` section) are automatically handled:
1. `derive_from_legacy_config()` infers roles from `cluster.n_nodes`
2. Deprecation warning is emitted
3. Legacy tags (`head`/`worker`) registered alongside new tags

### Usage Examples

**2-Node (default):**
```bash
python scripts/run_multinode_training.py \
    --compute-pool GPU_POOL \
    --external-access-integrations RL_EAI \
    --config grpo_multinode_template.yaml
```

**3-Node with External Judge:**
```bash
python scripts/run_multinode_training.py \
    --compute-pool GPU_POOL \
    --external-access-integrations RL_EAI \
    --config grpo_3node_config.yaml \
    --with-judge \
    --judge-config judge_server_config.yaml
```

**4-Node Scaling (2 rollout + 2 trainer):**
```yaml
# In config:
nodes:
  total: 4
  roles:
    rollout: 2
    trainer: 2
allocation_mode: sglang:d2p1t4+d8p1t1  # 2 inference servers, 8 trainers
```

---

# AReaL RL Fine-tuning on Snowflake SPCS

## Project Overview

This project runs AReaL (Asynchronous Reinforcement Learning) GRPO training on Snowflake Snowpark Container Services (SPCS) using ML Jobs. The goal is to fine-tune LLMs using reinforcement learning with a medical SOAP note generation task.

## Current Status

**Phase: 2-Node Decoupled Training WORKING! ✅**

- ✅ Custom Docker image builds and runs on SPCS
- ✅ Ray 2.53.0 cluster starts successfully
- ✅ HuggingFace model download works (Qwen/Qwen3-0.6B)
- ✅ **torch.compile disabled** (`enable_torch_compile: false`) - KEY FIX!
- ✅ CUDA graphs disabled (`disable_cuda_graph: true`)
- ✅ SGLang rollout servers **healthy** (~20 sec startup)
- ✅ **SPCS Port Fix Applied** - SGLang uses ports 34000-50000
- ✅ **Cross-Node HTTP WORKING** - Worker can reach SGLang on head node
- ✅ **2-Node Decoupled Training COMPLETED** - Job `RUN_AREAL_12RMR54QV0OQ0`
- ⚠️ Single-node colocated mode still broken (trainer_n_nodes=0 bug)

## Snowflake Configuration

| Resource | Value |
|### 9. SPCS Cross-Node HTTP Port Restrictions (CRITICAL!)

**Problem:** SPCS blocks cross-node HTTP on ports below ~30000. AReaL's default port range starts at 10000, causing SGLang servers to be unreachable from worker nodes.

**Symptoms:**
- SGLang servers start successfully on head node
- Worker node times out trying to reach SGLang HTTP endpoints
- Logs show: `Worker→SGLang: 0/N ready. Not ready: ['IP:PORT(Timeout)']`
- Training never starts due to rollout server connectivity failure

**Root Cause:** AReaL's `sglang_server.py` uses port range starting at 10000:
```python
port_range = (server_local_idx * ports_per_server + 10000, ...)
```

**Solution:** Created `src/areal_patches/sglang_server_patched.py` that changes base port to 34000:
```python
SPCS_BASE_PORT = 34000  # Minimum port that works cross-node
port_range = (server_local_idx * ports_per_server + SPCS_BASE_PORT, ...)
```

**Patch Application:** Updated `src/run_areal.py` to apply both patches:
```python
def apply_areal_patches():
    patches = [
        {"name": "Ray Launcher", "file": "ray_launcher_patched.py", ...},
        {"name": "SGLang Server (SPCS port fix)", "file": "sglang_server_patched.py", ...},
    ]
```

**Verification:** Job `RUN_AREAL_12RMR54QV0OQ0` completed successfully with SGLang on port 39948.

----------|-------|
| Connection | `preprod8` |
| Database | `RL_TRAINING_DB` |
| Schema | `RL_SCHEMA` |
| Compute Pool | `RL_GPU_POOL` (GPU_NV_M: 4xA10 GPUs, 44 vCPUs, 178GB RAM) |
| Image Repository | `RL_TRAINING_DB.RL_SCHEMA.RL_IMAGES` |
| External Access | `PYPI_HF_EAI` |

---

## AReaL Allocation Mode Deep Dive

### Allocation Mode Syntax

Format: `[backend:]d<dp>p<pp>t<tp>[+d<dp>p<pp>t<tp>]`

| Mode | Example | Description |
|------|---------|-------------|
| **Colocated** | `sglang:d4p1t1` | Single allocation - inference + training share same GPUs |
| **Decoupled** | `sglang:d2p1t1+d2p1t1` | Two allocations - inference and training on separate GPUs |
| **Training-only** | `d4p1t1` | No inference backend (SFT, not RL) |

### Key Code Locations

| File | Purpose |
|------|---------|
| `areal/api/alloc_mode.py` | AllocationMode parsing, AllocationType enum |
| `areal/infra/launcher/ray.py` | Ray job submission, placement groups |

### The Colocated Mode Bug (v0.5.3)

**Problem**: For `sglang:d4p1t1` on 1 node:

```python
# In ray.py lines 397, 527-529:
n_sglang_nodes = max(1, allocation_mode.gen.world_size // n_gpus_per_node)
# = max(1, 4 // 4) = 1

trainer_n_nodes = n_nodes - n_sglang_nodes  
# = 1 - 1 = 0  <-- BUG!

n_trainer_processes = trainer_n_nodes * config.cluster.n_gpus_per_node  
# = 0 * 4 = 0  <-- No trainer processes!
```

**Root Cause**: The code assumes inference and training need **separate nodes**, but colocated mode should **share the same nodes**.

**Symptoms**:
- SGLang servers launch successfully
- Logs show "Waiting for 1 jobs" forever
- Trainer never starts because `n_trainer_processes = 0`

### Decoupled Mode Bug (Single Node)

**Problem**: For `sglang:d2p1t1+d2p1t1` on 1 node:

```python
n_sglang_nodes = max(1, 2 // 4) = max(1, 0) = 1
trainer_n_nodes = 1 - 1 = 0  # ZeroDivisionError at line 162!
```

This causes `count % nodes` → `count % 0` → ZeroDivisionError.

### Working Configurations

| Nodes | Allocation Mode | Status | Notes |
|-------|-----------------|--------|-------|
| 1 | `sglang:d4p1t1` | ❌ Stuck | Colocated bug - trainer_n_nodes=0 |
| 1 | `sglang:d2p1t1+d2p1t1` | ❌ ZeroDivision | Decoupled requires ≥2 nodes |
| 2 | `sglang:d4p1t1+d4p1t1` | Should work | Separate nodes for inference/training |
| 2 | Patched ray.py | Testing | Custom node affinity fix |

---

## Issues Solved

### 1. Ray 2.53.0 Flag Compatibility

**Problem:** ML Jobs bootstrap uses `--dashboard-grpc-port` which is deprecated in Ray 2.53.0.

**Solution:** Created wrapper script at `/AReaL/.venv/bin/ray` that converts deprecated flags.

### 2. Empty Dashboard Flags

**Problem:** ML Jobs bootstrap passes empty `--dashboard-port=` and `--dashboard-host=` flags.

**Solution:** Wrapper sets Ray default values when empty.

### 3. HuggingFace XET Storage CAS Errors

**Problem:** HuggingFace's new XET requires endpoints not in standard network rules.

**Solution:** Disabled XET with `HF_HUB_DISABLE_XET=1` in Dockerfile.

### 4. Missing Snowflake Runtime Modules

**Problem:** ML Jobs bootstrap requires `snowflake.runtime` module.

**Solution:** Added `mlruntimes_service-2.2.26-py3-none-any.whl` to the image.

### 5. SGLang Server Startup Timeout

**Problem:** `TimeoutError: server launch failed` - torch.compile warmup taking too long.

**Solution:** Three-part fix in config:
```yaml
sglang:
  enable_torch_compile: false  # Master switch - KEY FIX!
  disable_cuda_graph: true     # Speeds up startup
rollout:
  setup_timeout: 600.0         # Now only needs 10 min
```

### 6. HTTP Cross-Node Issue

**Problem:** Ray schedules SGLang servers on wrong node (head instead of worker).

**Solution:** Use `NodeAffinitySchedulingStrategy` to pin tasks to specific nodes:
```python
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

server_ref = start_server.options(
    scheduling_strategy=NodeAffinitySchedulingStrategy(
        node_id=worker_node_id,
        soft=False  # Hard constraint
    )
).remote()
```

### 7. AReaL v0.5.3 Import Paths

**Problem:** PPOTrainer moved between versions.

**Solution:** Use fallback import:
```python
try:
    from areal import PPOTrainer
except ImportError:
    from areal.experimental.trainer import PPOTrainer
```

### 8. Config Schema Differences (v0.5.3)

**Problem:** v0.5.3 has different config schema than newer versions.

**Key differences:**
- No `model_path` top-level key - use `actor.path` directly
- No `model_name` key
- Must have `sglang:` section when using `sglang:` allocation mode
- Must have `vllm:` section when using `vllm:` allocation mode

---

## Port Diagnostic Results

### SPCS Cross-Node HTTP Port Restrictions (CRITICAL!)

**Discovery**: SPCS blocks cross-node HTTP traffic on ports below ~30000.

| Port | Cross-Node HTTP | Notes |
|------|-----------------|-------|
| 42106 | ✅ WORKS | In working range |
| 39948 | ✅ WORKS | Successfully used by SGLang |
| 34517 | ✅ WORKS | In working range |
| 22573 | ❌ BLOCKED | Timeout from worker node |
| 21381 | ❌ BLOCKED | Timeout from worker node |
| 20630 | ❌ BLOCKED | TCP CLOSED |
| 20511 | ❌ BLOCKED | Timeout from worker node |
| 17436 | ❌ BLOCKED | Timeout from worker node |
| 15849 | ❌ BLOCKED | TCP CLOSED |

**Pattern**: Ports ≥ 34000 work, ports < 30000 are blocked for cross-node HTTP.

### Original AReaL Port Selection (BROKEN on SPCS)

```python
# In areal/launcher/sglang_server.py
ports_per_server = 40000 // n_servers_per_node
port_range = (
    server_local_idx * ports_per_server + 10000,  # Base = 10000
    (server_local_idx + 1) * ports_per_server + 10000,
)
```

With 4 SGLang servers per node:
- Server 0: ports 10000-20000 ❌ BLOCKED
- Server 1: ports 20000-30000 ❌ BLOCKED  
- Server 2: ports 30000-40000 ⚠️ Partially works
- Server 3: ports 40000-50000 ✅ WORKS

### Fixed Port Selection (WORKS on SPCS)

```python
# In src/areal_patches/sglang_server_patched.py
SPCS_BASE_PORT = 34000  # Minimum port that works cross-node in SPCS
SPCS_MAX_PORT = 50000
SPCS_PORT_RANGE = SPCS_MAX_PORT - SPCS_BASE_PORT  # 16000 ports

ports_per_server = SPCS_PORT_RANGE // n_servers_per_node
port_range = (
    server_local_idx * ports_per_server + SPCS_BASE_PORT,  # Base = 34000
    (server_local_idx + 1) * ports_per_server + SPCS_BASE_PORT,
)
```

With 1 SGLang server (TP=4): port range 34000-50000 ✅ ALL WORK

---

### 10. AReaL Module Import Compatibility

**Problem:** Docker image uses `areal.platforms`, `areal.utils.*` while GitHub source uses `areal.infra.*`.

**Solution:** Use try/except fallback imports in patch files:
```python
try:
    from areal.infra.platforms import current_platform
    from areal.infra.utils.launcher import TRITON_CACHE_PATH, get_scheduling_spec
except ImportError:
    from areal.platforms import current_platform
    from areal.utils.launcher import TRITON_CACHE_PATH, get_scheduling_spec
```

---

## Job History

| Job ID | Status | Error | Notes |
|--------|--------|-------|-------|
| `RUN_AREAL_12RMR54QV0OQ0` | ✅ DONE | - | **2-node decoupled SUCCESS!** SGLang port 39948, ~2.6 min |
| `RUN_AREAL_ZTA366R3ZF5L` | FAILED | ModuleNotFoundError | Import path mismatch - fixed with try/except |
| `RUN_AREAL_1OVHF1UJYNPQK` | STUCK | Trainer never started | Colocated mode bug |
| `RUN_AREAL_1KV76LJVNIOAH` | FAILED | ConfigAttributeError: sglang not in struct | Missing sglang config |
| `RUN_AREAL_IMEAHMWAB3MM` | FAILED | ValueError: Invalid backend: None | Used `d4p1t1` without backend |
| `RUN_AREAL_DY1MJQJW78HM` | ✅ DONE | - | Node affinity test passed |
| `RUN_AREAL_QWJOBZH08TCB` | ✅ DONE | - | Port diagnostic passed |

---

## Key Files

| File | Purpose |
|------|---------|
| `docker/Dockerfile` | Custom AReaL image with all fixes |
| `src/run_areal.py` | Main training entry point with patch application |
| `src/node_allocation.py` | **NEW:** Unified node allocation schema |
| `src/soap_grpo_trainer.py` | GRPO trainer implementation |
| `src/grpo_lora_config.yaml` | Legacy 2-node training configuration |
| `src/grpo_multinode_template.yaml` | **NEW:** Template config with `nodes` section |
| `src/areal_patches/ray_launcher_patched.py` | Custom Ray launcher with unified node scheduling |
| `src/areal_patches/sglang_server_patched.py` | SPCS port fix (see below) |
| `src/areal_patches/remote_inf_engine_patched.py` | Judge resolution fix - 5min timeout, no silent fallback |
| `scripts/run_multinode_training.py` | **NEW:** Unified job submission script |
| `scripts/run_rl_train.py` | **DEPRECATED:** Legacy 2-node submission |
| `scripts/run_3node_training.py` | **DEPRECATED:** Legacy 3-node submission |
| `src/test_node_allocation.py` | **NEW:** Unit tests for node allocation |

---

## AReaL Patches Detailed

### Patch 1: `ray_launcher_patched.py`

**Purpose:** Custom Ray launcher for 2-node decoupled mode with deterministic node placement.

**Key Changes:**

1. **Node Tag Resource Scheduling**: Uses Ray custom resources (`node_tag:head`, `node_tag:worker`) for deterministic task placement instead of placement groups.
   ```python
   def submit_with_node_tag(self, job_name, ..., node_tag: str, node_tag_amount: float = 0.01):
       resource_key = f"node_tag:{node_tag}"
       future = ray.remote(
           resources={resource_key: node_tag_amount},
           ...
       )(run_func).remote(...)
   ```

2. **2-Node Decoupled Mode Detection**: Automatically detects when to use node tags vs placement groups.
   ```python
   use_node_tag = (n_nodes == 2 and n_sglang_nodes == 1 and not cross_nodes)
   if use_node_tag:
       launcher.submit_array_with_node_tag(..., node_tag="head")  # SGLang on head
   ```

3. **Cross-Node HTTP Health Check**: Before launching trainers, verifies SGLang is reachable from worker node.
   ```python
   @ray.remote(resources={"node_tag:worker": 0.01})
   def test_sglang_from_worker(addrs):
       # Polls http://{addr}/health until all servers respond HTTP 200
   ```

4. **Torch Distributed Env Hook for Node Tags**: Discovers worker node IP for MASTER_ADDR.
   ```python
   def torch_env_hook_for_node_tag(n_tasks):
       for node in ray.nodes():
           if "node_tag:worker" in node.get("Resources", {}):
               worker_ip = node["NodeManagerAddress"]
       return [{"RANK": i, "MASTER_ADDR": worker_ip, ...} for i in range(n_tasks)]
   ```

5. **Colocated Mode Fix**: Sets `trainer_n_nodes = n_nodes` for colocated mode (sharing GPUs).
   ```python
   if allocation_mode.type_ == AllocationType.COLOCATE:
       trainer_n_nodes = n_nodes  # Not n_nodes - n_llm_nodes (which was 0!)
   ```

---

### Patch 2: `sglang_server_patched.py`

**Purpose:** Fix SGLang port selection for SPCS cross-node HTTP restrictions.

**Key Changes:**

1. **Port Range Constants**: Defines SPCS-safe port range.
   ```python
   SPCS_BASE_PORT = 34000  # Minimum port that works cross-node in SPCS
   SPCS_MAX_PORT = 50000   # Maximum port
   SPCS_PORT_RANGE = 16000  # 16000 ports available
   ```

2. **Port Selection Fix**: Replaces original 10000-based port range.
   ```python
   # Original (BROKEN on SPCS):
   port_range = (server_local_idx * ports_per_server + 10000, ...)
   
   # Patched (WORKS on SPCS):
   port_range = (server_local_idx * ports_per_server + SPCS_BASE_PORT, ...)
   ```

3. **Import Compatibility**: Handles both `areal.infra.*` (GitHub) and `areal.*` (Docker) paths.
   ```python
   try:
       from areal.infra.platforms import current_platform
       from areal.infra.utils.launcher import TRITON_CACHE_PATH
   except ImportError:
       from areal.platforms import current_platform
       from areal.utils.launcher import TRITON_CACHE_PATH
   ```

4. **Enhanced Logging**: Logs port selection for debugging.
   ```python
   logger.info(f"SPCS PORT FIX: Using port range {SPCS_BASE_PORT}-{SPCS_MAX_PORT}")
   logger.info(f"Server {server_local_idx}: port_range={port_range}")
   logger.info(f"Server {server_local_idx}: selected port={server_port}")
   ```

---

### Patch 3: `remote_inf_engine_patched.py`

**Purpose:** Fix judge server resolution timeout and eliminate silent fallback to policy model.

**Problem with Original Code:**
```python
# In areal/infra/remote_inf_engine.py
# When experiment_name/trial_name is set (for external judge):
try:
    self.addresses = wait_llm_server_addrs(
        experiment_name=self.config.experiment_name,
        trial_name=self.config.trial_name,
        timeout=1,  # Only 1 second! WAY TOO SHORT
    )
except (TimeoutError, RuntimeError):
    # SILENT FALLBACK - uses rollout servers instead of judge!
    addrs_str = os.getenv("AREAL_LLM_SERVER_ADDRS")
    if addrs_str:
        self.addresses = addrs_str.split(",")  # BAD: Judge now uses policy model!
```

**Why This Is Critical:**
- When training with an external judge (for reward model), the judge needs time to start
- 1 second is not enough - SGLang servers can take 30+ seconds to initialize
- Silent fallback means **the policy model judges itself** → defeats the purpose!
- Training appears to work but produces meaningless results

**Key Changes:**

1. **5-Minute Timeout**: Increases timeout from 1 second to 300 seconds.
   ```python
   JUDGE_RESOLVE_TIMEOUT = 300  # 5 minutes instead of 1 second
   
   self.addresses = wait_llm_server_addrs(
       experiment_name=self.config.experiment_name,
       trial_name=self.config.trial_name,
       timeout=JUDGE_RESOLVE_TIMEOUT,
   )
   ```

2. **Error on Failure (No Silent Fallback)**:
   ```python
   except (TimeoutError, RuntimeError) as e:
       # CRITICAL: Do NOT silently fallback to rollout servers!
       self.logger.error(
           f"SPCS PATCH: FAILED to resolve server via name_resolve after {JUDGE_RESOLVE_TIMEOUT}s!"
       )
       raise RuntimeError(
           f"External server resolution failed for experiment={self.config.experiment_name}, "
           f"trial={self.config.trial_name} after {JUDGE_RESOLVE_TIMEOUT}s."
       ) from e
   ```

3. **Applied via Monkey-Patch**: Unlike file replacement patches, this one uses Python monkey-patching since only the `initialize()` method needs modification.
   ```python
   def apply_remote_inf_engine_patch():
       from areal.infra.remote_inf_engine import RemoteInfEngine
       RemoteInfEngine._original_initialize = RemoteInfEngine.initialize
       RemoteInfEngine.initialize = patched_initialize
   ```

**Use Case: 3-Node Setup (Judge + Rollout + Trainer)**
- Node 1: Judge server (separate model like GPT-4 or fine-tuned reward model)
- Node 2: Rollout server (policy model for generating responses)  
- Node 3: Trainer (GRPO training with policy gradients)

The judge server registers via `name_resolve.add_subentry()`, and this patch ensures the trainer waits long enough to find it via `wait_llm_server_addrs()`.

---

### How Patches Are Applied

In `src/run_areal.py`:
```python
def apply_areal_patches():
    patches = [
        {
            "name": "Ray Launcher",
            "file": "ray_launcher_patched.py",
            "module": "areal.launcher.ray",
            "target": "/AReaL/.venv/lib/python3.11/site-packages/areal/launcher/ray.py",
        },
        {
            "name": "SGLang Server (SPCS port fix)",
            "file": "sglang_server_patched.py",
            "module": "areal.launcher.sglang_server",
            "target": "/AReaL/.venv/lib/python3.11/site-packages/areal/launcher/sglang_server.py",
        },
    ]
    for patch in patches:
        shutil.copy(patch_src, patch["target"])
```

---

## Docker Image

### Current Working Image
```
preprod8-notebook-mltest.awsuswest2preprod8.registry-dev.snowflakecomputing.com/rl_training_db/rl_schema/rl_images/areal-runtime:v0.5.3-ray253-fix6
```

### Build Commands
```bash
cd docker/
docker buildx build --platform linux/amd64 -t <registry>/areal-runtime:<tag> --load .
docker push <registry>/areal-runtime:<tag>
```

---

## Job Submission

**IMPORTANT: Always use `--no-wait` flag** to avoid blocking and allow immediate status checks:
```bash
# Submit job and immediately return (non-blocking)
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python scripts/run_rl_train.py \
    -p RL_GPU_POOL -e PYPI_HF_EAI -n 2 --no-wait ...

# Then check status and logs:
SELECT SYSTEM$GET_SERVICE_LOGS('RL_TRAINING_DB.RL_SCHEMA.<JOB_ID>', 0, 'main', 1000);
```

### Single Node (Currently Broken)
```bash
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python scripts/run_rl_train.py \
    --compute-pool RL_GPU_POOL \
    --external-access-integrations PYPI_HF_EAI \
    --database RL_TRAINING_DB \
    --schema RL_SCHEMA \
    --num-nodes 1 \
    --no-wait \
    --runtime "<registry>/areal-runtime:v0.5.3-ray253-fix6"
```

### Two Nodes (Decoupled Mode)
```bash
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python scripts/run_rl_train.py \
    --compute-pool RL_GPU_POOL \
    --external-access-integrations PYPI_HF_EAI \
    --database RL_TRAINING_DB \
    --schema RL_SCHEMA \
    --num-nodes 2 \
    --no-wait \
    --runtime "<registry>/areal-runtime:v0.5.3-ray253-fix6"
```

### Three Nodes (External Judge Mode)

**Architecture:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Node 1        │    │   Node 2        │    │   Node 3        │
│   (judge)       │    │   (rollout)     │    │   (trainer)     │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Judge Model │ │    │ │ Policy Model│ │    │ │ GRPO Trainer│ │
│ │ (Qwen-8B)   │ │    │ │ (Qwen-0.6B) │ │    │ │             │ │
│ │             │◄├────┤►│ SGLang      │◄├────┤►│ LoRA weights│ │
│ │ Reward calc │ │    │ │ Rollout     │ │    │ │ Policy grad │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ node_tag:judge  │    │ node_tag:rollout│    │ node_tag:trainer│
└─────────────────┘    └─────────────────┘    └─────────────────┘
     (Job 1)                        (Job 2)
```

**Step 1: Launch Judge Server (separate job)**
```bash
# Judge must start FIRST and register with name_resolve
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python scripts/run_judge_server.py \
    --compute-pool RL_GPU_POOL \
    --external-access-integrations PYPI_HF_EAI \
    --database RL_TRAINING_DB \
    --schema RL_SCHEMA \
    --num-nodes 1 \
    --config judge_server_config.yaml \
    --runtime "<registry>/areal-runtime:v0.5.3-ray253-fix6"
```

**Step 2: Launch Training Job (waits for judge)**
```bash
# Training job will resolve judge address via name_resolve (5 min timeout)
SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 \
AREAL_3NODE_MODE=1 \
AREAL_NUM_NODES=2 \
python scripts/run_rl_train.py \
    --compute-pool RL_GPU_POOL \
    --external-access-integrations PYPI_HF_EAI \
    --database RL_TRAINING_DB \
    --schema RL_SCHEMA \
    --num-nodes 2 \
    --config grpo_3node_config.yaml \
    --runtime "<registry>/areal-runtime:v0.5.3-ray253-fix6"
```

**Key Environment Variables:**
| Variable | Description |
|----------|-------------|
| `AREAL_3NODE_MODE=1` | Enable 3-node scheduling (rollout/trainer tags) |
| `AREAL_NUM_NODES=2` | Number of nodes for training job (excluding judge) |

**Why Use 3-Node Setup?**
- **Better reward signal**: Judge model (larger/smarter) evaluates policy output
- **Avoids self-judging**: Policy model doesn't evaluate its own responses
- **Scalable**: Judge can be shared across multiple training jobs
- **Quality**: Use GPT-4, Claude, or fine-tuned reward model as judge

---

## Node Tag Scheduling System

### Overview

Ray's default placement groups don't guarantee deterministic node assignment. We use **custom Ray resources** (node tags) to pin tasks to specific nodes.

### How It Works

1. **Registration** (`run_areal.py`): At startup, assigns `node_tag:X = 1.0` resource to each node
2. **Scheduling** (`ray_launcher_patched.py`): Tasks request fractional amounts (0.01) of node tags
3. **Multiplexing**: 100 tasks can share one node (100 × 0.01 = 1.0)

### Node Tags by Mode

| Mode | Node 0 | Node 1 | Node 2 |
|------|--------|--------|--------|
| 2-node | `node_tag:head` | `node_tag:worker` | - |
| 3-node | `node_tag:judge`* | `node_tag:rollout` | `node_tag:trainer` |

*Judge runs as separate job

### Implementation Details

```python
# Registration (run_areal.py)
ray.experimental.set_resource("node_tag:rollout", 1.0, node_id)

# Scheduling (ray_launcher_patched.py)  
@ray.remote(resources={"node_tag:trainer": 0.01})
def trainer_task():
    ...
```

### Extending to N Nodes

To add more nodes (e.g., multiple rollout servers):

1. Add new tag constants in `ray_launcher_patched.py`:
   ```python
   NODE_TAG_ROLLOUT_0 = "rollout_0"
   NODE_TAG_ROLLOUT_1 = "rollout_1"
   ```

2. Update `register_node_tags()` in `run_areal.py` to handle additional nodes

3. Update scheduling logic to distribute tasks across rollout nodes

---

## Monitoring Commands

```sql
-- Check job status
SELECT SYSTEM$GET_SERVICE_STATUS('RL_TRAINING_DB.RL_SCHEMA.<JOB_ID>');

-- View logs
SELECT TIMESTAMP, VALUE::STRING as message
FROM SNOWFLAKE.TELEMETRY.EVENTS 
WHERE RESOURCE_ATTRIBUTES:"snow.service.name"::STRING = '<JOB_ID>'
AND LENGTH(VALUE::STRING) > 30
ORDER BY TIMESTAMP DESC
LIMIT 50;

-- List all jobs
SHOW SERVICES LIKE '%AREAL%' IN COMPUTE POOL RL_GPU_POOL;

-- Check compute pool
SHOW COMPUTE POOLS LIKE 'RL_GPU_POOL';
```

---

## Next Steps

1. **Scale Testing**
   - Test with larger models (Qwen-1.8B, Qwen-7B)
   - Measure throughput and GPU utilization
   - Compare single-node vs multi-node performance

2. **Production Hardening**
   - Add checkpointing for long training runs
   - Implement model export to Snowflake stage

---

## Repository Setup (For New Machine)

### Essential Files (Committed)

```
rl_finetune/
├── AGENT.md                          # This file - critical context
├── .gitignore                        # Excludes large files
├── docker/
│   ├── Dockerfile                    # Image build instructions
│   └── mlruntimes_service-*.whl      # Required runtime wheel
├── scripts/
│   ├── e2e_test_multinode.py         # E2E test script
│   └── run_multinode_training.py     # Production job submission
└── src/
    ├── run_areal.py                  # Main entry point
    ├── log_wrapper.py                # Log wrapper for SPCS
    ├── node_allocation.py            # Unified node allocation
    ├── soap_grpo_trainer.py          # GRPO trainer
    ├── e2e_test_2node.yaml           # Test config
    ├── grpo_multinode_template.yaml  # Template config
    └── areal_patches/
        ├── ray_launcher_patched.py   # Ray scheduler fixes
        ├── sglang_server_patched.py  # Port & LoRA fixes
        ├── remote_inf_engine_patched.py
        └── ray_weight_sync.py
```

### Excluded (Regenerate Locally)

| Item | How to Regenerate |
|------|-------------------|
| `docker/areal-runtime.tar` | `cd docker && docker build -t areal-runtime .` |
| `src/data/*.json` | Run `scripts/upload_data.py` or use existing stage data |
| `.venv/` | `python -m venv .venv && pip install -r requirements.txt` |

### Quick Start on New Machine

```bash
# 1. Clone and setup
git clone <repo>
cd samples/ml/ml_jobs/rl_finetune

# 2. Set connection
export SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8

# 3. Check existing job (if any)
snow sql -q "SHOW SERVICES IN SCHEMA RL_TRAINING_DB.RL_SCHEMA"

# 4. Run E2E test
python scripts/e2e_test_multinode.py \
    --compute-pool RL_GPU_POOL \
    --database RL_TRAINING_DB \
    --schema RL_SCHEMA \
    --external-access-integrations PYPI_HF_EAI \
    --runtime "preprod8-notebook-mltest.awsuswest2preprod8.registry-dev.snowflakecomputing.com/rl_training_db/rl_schema/rl_images/areal-runtime:v0.5.3-ray253-fix6" \
    --test 2node_basic
```
   - Add Weights & Biases integration for tracking

3. **Upstream Fixes**
   - Report SPCS port restriction issue to SPCS team
   - Consider contributing port fix to AReaL repo
   - Document SPCS networking constraints

---

## Troubleshooting Tips

1. **Job stuck at "Waiting for X jobs"**: Check if trainer_n_nodes > 0 in allocation mode
2. **ZeroDivisionError**: Decoupled mode requires n_nodes > n_inference_nodes
3. **ConfigAttributeError: sglang not in struct**: Add `sglang:` section to config
4. **ValueError: Invalid backend**: Use `sglang:d4p1t1` not just `d4p1t1` for RL
5. **TimeoutError: server launch failed**: Add `enable_torch_compile: false` to sglang config
6. **HuggingFace download fails**: Verify EAI is attached, check network rules
7. **Ray flag errors**: Ensure using image with wrapper script
8. **Worker can't reach SGLang (Timeout)**: **SPCS port issue** - ensure SGLang uses ports ≥34000
9. **ModuleNotFoundError: areal.infra**: Use try/except fallback imports for `areal.*` paths
10. **Insufficient GPU resources**: Check for zombie jobs with `SHOW SERVICES LIKE '%AREAL%'` and drop them

---

## References

- [AReaL GitHub](https://github.com/inclusionAI/AReaL)
- [Snowflake ML Jobs Documentation](https://docs.snowflake.com/en/developer-guide/snowpark-ml/jobs)
- [Ray Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)

---

### 11. LoRA Weight Update Issues (CRITICAL FOR LORA TRAINING!)

**Problem:** XCCL (GPU-to-GPU) weight update fails for LoRA due to parameter name AND shape mismatch.

**Root Cause:** SGLang merges QKV projections internally, creating different structures:

| Component | Trainer (Separate) | SGLang (Merged) |
|-----------|-------------------|-----------------|
| Q LoRA | `q_proj.lora_A` [16, 1024] | N/A (merged) |
| K LoRA | `k_proj.lora_A` [16, 1024] | N/A (merged) |
| V LoRA | `v_proj.lora_A` [16, 1024] | N/A (merged) |
| QKV LoRA | N/A | `qkv_proj.lora_A` [16, 1024] + `qkv_proj.lora_B` [3072, 16] |

**What XCCL Expects:**
- **Exact** parameter name match (uses `model.named_parameters()`)
- **Exact** tensor shape match
- SGLang's XCCL endpoint does NO name translation - it expects the incoming tensor names to match internal model param names exactly
- For merged QKV models (Qwen, Llama), SGLang has `qkv_proj` internally, not separate `q/k/v_proj`

**Why Simple Name Translation Fails:**
Even if we rename `q_proj → qkv_proj`, the shapes don't match:
- Trainer's Q lora_B: `[1024, 16]` (Q output dim)
- SGLang's qkv_proj lora_B: `[3072, 16]` (Q+K+V output dim)

**Why Full Tensor Merging is Complex:**
1. Different models merge QKV differently (concat order varies)
2. GQA models have different K/V sizes than Q
3. SGLang's internal LoRA structure may differ from trainer's

**Disk Mode Is Different:**
When loading LoRA from disk (`/load_lora_adapter`), SGLang:
1. Reads the adapter config and weights
2. Uses the `peft` library's loading logic
3. `peft` handles name translation via `target_modules` config
4. This works because the adapter was saved with separate Q/K/V names

**SPCS Constraint:** No shared filesystem across nodes!
- `/mnt/job_stage/` is read-only
- Runtime storage is local ephemeral (not shared)
- Disk mode can't work cross-node without Ray object store

---

### 13. Cross-Node Weight Sync with Ray Object Store (RECOMMENDED SOLUTION)

**Problem:** Disk mode requires shared filesystem, which SPCS doesn't have cross-node.

**Solution:** Use Ray's object store as a shared memory layer:

```
[Trainer Node]                      [SGLang Node]
     |                                    |
     | 1. Save LoRA checkpoint to         |
     |    Ray object store                |
     |    ref = ray.put(weights_dict)     |
     |                                    |
     | 2. Call remote function on         |
     |    SGLang node with ref            |
     |----------------------------------->|
     |                                    | 3. ray.get(ref) to retrieve
     |                                    | 4. Save to local /tmp/lora/
     |                                    | 5. Call /load_lora_adapter
     |                                    |    with local path
     |                                    |
```

**Why This Works:**
1. Ray object store shares data across nodes without shared filesystem
2. SGLang's `/load_lora_adapter` handles name translation automatically
3. Uses disk mode's proven Q/K/V → qkv_proj translation logic

**Implementation:** See `areal_patches/ray_weight_sync.py`

**Setup:**
1. Config: `weight_update_mode: disk` in grpo_lora_config.yaml
2. Patch application: Applied automatically in `run_func()` in ray_launcher_patched.py
3. Auto-initialization: Patch reads `AREAL_LLM_SERVER_ADDRS` env var to find SGLang servers

**How it works:**
1. Patch intercepts `FSDPEngine._update_weights_from_disk()` calls
2. Extracts LoRA weights from the trainer's PEFT model
3. Saves weights + adapter config to Ray object store (`ray.put()`)
4. Spawns a Ray remote task on each SGLang node
5. Remote task does `ray.get()` to retrieve weights
6. Saves to local `/tmp/ray_lora_adapter/<name>/` as safetensors
7. Calls SGLang's `/load_lora_adapter` endpoint with local path
8. SGLang handles Q/K/V → qkv_proj name translation during load

---

### 13b. Why Disk Mode Works But XCCL Doesn't (LoRA + Merged QKV)

**Understanding the difference:**

| Mode | SGLang Endpoint | Name Translation |
|------|-----------------|------------------|
| Disk | `/load_lora_adapter` | ✅ PEFT handles q_proj→qkv_proj |
| XCCL | `/update_weights_from_distributed` | ❌ Expects exact param name match |

**Disk mode flow:**
1. Trainer saves `q_proj.lora_A`, `k_proj.lora_A`, `v_proj.lora_A` to disk
2. SGLang calls PEFT's `load_adapter()` 
3. PEFT uses `target_modules` config to map names
4. Internally, SGLang keeps Q/K/V LoRA SEPARATE (just applies to different slices of qkv output)

**XCCL mode flow:**
1. Trainer broadcasts tensor with name `q_proj.lora_A`
2. SGLang receives and tries to match via `model.named_parameters()`
3. SGLang has `qkv_proj.lora_A` internally → **NAME MISMATCH**

**Could we make XCCL work?**

Option A: **Patch SGLang's XCCL endpoint** to do name translation (invasive, version-specific)

Option B: **Use `/update_weights_from_tensor`** endpoint instead of NCCL broadcast
- Serialize tensors to JSON/bytes
- Send via HTTP 
- SGLang applies name translation on receive
- Pro: No NCCL setup needed
- Con: Slower than GPU-direct

Option C: **Ray object store + `/load_lora_adapter`** (current implementation)
- Uses proven PEFT loading path
- Works cross-node without shared filesystem
- Leverages disk mode's name translation

---

### 11b. Disk Mode Bug (AReaL)

There's a race condition in `_update_weights_from_disk()`:
1. Sends HTTP request to SGLang to load LoRA
2. THEN saves the checkpoint
This causes 400 error on first weight update since file doesn't exist yet.

**Workaround Options:**
1. Skip initial weight update (requires AReaL code change)
2. Pre-create initial LoRA checkpoint before training starts
3. Patch AReaL to save before triggering load

---

### 12. NCCL Configuration for SPCS (CRITICAL!)

**Problem:** NCCL tries to use sandbox network interfaces (172.16.x.x, 100.64.x.x) which are not routable cross-node.

**Solution:** Set these environment variables BEFORE any NCCL operations:
```bash
export NCCL_SOCKET_IFNAME=eth0    # Force pod network (10.244.x.x)
export NCCL_IB_DISABLE=1          # No InfiniBand in SPCS
```

**Applied in:**
- `run_areal.py` (at module level, before imports)
- `ray_launcher_patched.py` (in trainer env vars)

---

### 14. Ray 2.53.0 Deprecated `set_resource()` (CRITICAL!)

**Problem:** Ray 2.53.0 deprecated `ray.experimental.set_resource()` for dynamically creating custom resources.

**Error Message:**
```
Dynamic custom resources are deprecated. Consider using placement groups instead 
(docs.ray.io/en/master/placement-group.html). You can also specify resources at 
Ray start time with the 'resources' field in the cluster autoscaler.
```

**Impact:** The original node tag registration code tried to create `node_tag:rollout` and `node_tag:trainer` tags, but they fail to register.

**Solution:** Use the **pre-registered** SPCS ML Jobs tags instead:
- `node_tag:head` - Automatically registered on instance 0 (head node)
- `node_tag:worker` - Automatically registered on instance 1+ (worker nodes)

**Code Fix in `ray_launcher_patched.py`:**
```python
def get_rollout_node_tag(node_allocation, node_idx: int = 0) -> str:
    """Get the appropriate rollout node tag for a given index.
    
    For SPCS ML Jobs, we use the pre-registered tags:
    - node_tag:head for rollout (first node)
    - node_tag:worker for trainer (second node)
    
    This is because Ray 2.53+ deprecated ray.experimental.set_resource()
    so we can't dynamically create new tags.
    """
    # In 2-node setup, rollout goes on head node
    if node_allocation.total == 2 and node_idx == 0:
        return NODE_TAG_HEAD  # "head" - pre-registered by SPCS
    
    rollout_count = node_allocation.roles.get('rollout', 1)
    if rollout_count == 1:
        return NODE_TAG_HEAD  # Use head for single rollout
    return f"{NODE_TAG_ROLLOUT}_{node_idx}"


def get_trainer_node_tag(node_allocation, node_idx: int = 0) -> str:
    """Get the appropriate trainer node tag for a given index.
    
    For SPCS ML Jobs, we use the pre-registered tags:
    - node_tag:head for rollout (first node)
    - node_tag:worker for trainer (second node)
    """
    # In 2-node setup, trainer goes on worker node
    if node_allocation.total == 2:
        return NODE_TAG_WORKER  # "worker" - pre-registered by SPCS
    
    trainer_count = node_allocation.roles.get('trainer', 1)
    if trainer_count == 1:
        return NODE_TAG_WORKER  # Use worker for single trainer
    return f"{NODE_TAG_TRAINER}_{node_idx}"
```

**Key Changes in `run_areal.py`:**
1. Removed `ray.shutdown()` after tag registration to preserve cluster state
2. Set `RAY_ADDRESS` environment variable before launching AReaL
3. Changed `ray.init(address="auto")` to `ray.init(address=ray_address)` with actual IP:port

**Verification:** Job logs now show:
```
Scheduling SGLang server 0 to node_tag:head
```
Instead of the failing:
```
Scheduling SGLang server 0 to node_tag:rollout
```

**N>2 Node Limitation:** For setups with more than 2 nodes, the pre-registered tags won't be sufficient. Options:
1. Use Ray placement groups instead of custom resources
2. Request SPCS team to pre-register additional tags
3. Use `resources` field at Ray start time (requires container modification)

---

### 15. SGLang LoRA Configuration Change

**Problem:** Recent SGLang versions require additional parameters when using `--enable-lora` without `--lora-paths`.

**Error Message:**
```
AssertionError: When no initial --lora-paths is provided, you need to specify 
both --max-lora-rank and --lora-target-modules for LoRA initialization.
```

**Solution:** Add these parameters to SGLang launch command or config:
```yaml
sglang:
  extra_args:
    - "--max-lora-rank=16"
    - "--lora-target-modules=q_proj,k_proj,v_proj,o_proj"
```

Or in the patched SGLang server, add to the launch command:
```python
cmd.extend([
    "--max-lora-rank", "16",
    "--lora-target-modules", "q_proj,k_proj,v_proj,o_proj",
])
```

---

### Port Summary for SPCS

| Port Range | Usage | Status |
|------------|-------|--------|
| 34000-50000 | SGLang HTTP | ✅ Works cross-node |
| 12031-13000 | Weight sync / torch distributed | ✅ Safe range |
| < 30000 | HTTP | ❌ BLOCKED cross-node |
| 29500 (default) | torch distributed | ⚠️ May be blocked |

