#!/usr/bin/env python3
"""
Policy RL Training with Local vLLM Judge on SPCS

Trains Qwen3-1.7B to summarize customer support interactions in compliance with
a multi-rule policy, using a decomposed reward function (deterministic + local LLM judge).

GPU layout (8x A100-40GB):
  GPUs 4-7: 4x vLLM judge servers (Qwen3-8B, ~16GB BF16 each)
  GPUs 0-3: AReaL (2 vLLM rollout dp=2 + 2 FSDP training dp=2) for Qwen3-1.7B

Key differences from policy_rl_cortex_judges/run_policy_rl.py:
  1. Starts 4 local vLLM judges on GPUs 4-7 before AReaL
  2. Ray sees only 4 GPUs (NUM_GPUS=4)
  3. No Cortex API dependency for judging
  4. Reward calls routed by transcript hash for prefix cache reuse
"""
import asyncio
import atexit
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

# Force unbuffered output for SPCS logs
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Suppress verbose output
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ============================================================================
# SPCS Environment Configuration
# ============================================================================
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_cache"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Phase 1 (ENABLE_LLM_JUDGE=0): all 8 GPUs for AReaL, no judges
# Phase 2 (ENABLE_LLM_JUDGE=1): GPUs 0-3 for AReaL, GPUs 4-7 for judges
ENABLE_LLM_JUDGE = os.environ.get("ENABLE_LLM_JUDGE", "1") == "1"

if ENABLE_LLM_JUDGE:
    JUDGE_GPUS = [4, 5, 6, 7]
    NUM_AREAL_GPUS = 4
else:
    JUDGE_GPUS = []
    NUM_AREAL_GPUS = 8

JUDGE_BASE_PORT = int(os.environ.get("LOCAL_JUDGE_BASE_PORT", "38899"))
JUDGE_PORTS = [JUDGE_BASE_PORT + i for i in range(len(JUDGE_GPUS))]
os.environ["NUM_GPUS"] = str(NUM_AREAL_GPUS)

JUDGE_MODEL = os.environ.get("LOCAL_JUDGE_MODEL", "Qwen/Qwen3-8B")

phase_label = "Phase 2 (with judge)" if ENABLE_LLM_JUDGE else "Phase 1 (deterministic only)"
print("=" * 60)
print(f"Policy RL Training — {phase_label}")
print("=" * 60)
if ENABLE_LLM_JUDGE:
    print(f"  Judge model:  {JUDGE_MODEL}")
    print(f"  Judge GPUs:   {JUDGE_GPUS} ({len(JUDGE_GPUS)} judges)")
    print(f"  Judge ports:  {JUDGE_PORTS}")
print(f"  AReaL GPUs:   0-{NUM_AREAL_GPUS - 1} ({NUM_AREAL_GPUS} total)")

# ============================================================================
# Local vLLM Judge Server
# ============================================================================
_judge_processes = []


def start_judge_servers():
    """Spawn vLLM OpenAI-compatible servers on dedicated GPUs.

    Launches all servers in parallel, then waits for all /health endpoints
    to return 200 (up to 10 minutes). Registers atexit handler.
    """
    global _judge_processes

    print(f"\n--- Starting {len(JUDGE_GPUS)} vLLM Judge Servers ---")

    for gpu, port in zip(JUDGE_GPUS, JUDGE_PORTS):
        print(f"  Launching judge on GPU {gpu}, port {port}...")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", JUDGE_MODEL,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
            "--trust-remote-code",
            "--disable-log-requests",
            "--enable-prefix-caching",
        ]

        proc = subprocess.Popen(
            cmd,
            env=env,
        )
        _judge_processes.append((gpu, port, proc))

    atexit.register(_kill_judges)

    # Wait for all judges to become healthy
    max_wait = 600  # 10 minutes
    start_time = time.time()
    last_log = 0
    healthy = set()

    while time.time() - start_time < max_wait and len(healthy) < len(JUDGE_GPUS):
        for i, (gpu, port, proc) in enumerate(_judge_processes):
            if i in healthy:
                continue
            # Check if process died
            if proc.poll() is not None:
                output = proc.stdout.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"vLLM judge on GPU {gpu} port {port} exited with code "
                    f"{proc.returncode}. Last output:\n{output[-2000:]}"
                )
            try:
                req = urllib.request.Request(
                    f"http://localhost:{port}/health", method="GET"
                )
                resp = urllib.request.urlopen(req, timeout=5)
                if resp.status == 200:
                    healthy.add(i)
                    print(f"  Judge GPU {gpu} port {port} ready "
                          f"({len(healthy)}/{len(JUDGE_GPUS)})")
            except Exception:
                pass

        elapsed = time.time() - start_time
        if elapsed - last_log >= 30:
            print(f"  Waiting for judges... {len(healthy)}/{len(JUDGE_GPUS)} "
                  f"ready ({elapsed:.0f}s)")
            last_log = elapsed
        if len(healthy) < len(JUDGE_GPUS):
            time.sleep(5)

    if len(healthy) < len(JUDGE_GPUS):
        _kill_judges()
        raise RuntimeError(
            f"Only {len(healthy)}/{len(JUDGE_GPUS)} judges became healthy "
            f"within {max_wait}s"
        )

    elapsed = time.time() - start_time
    print(f"  All {len(JUDGE_GPUS)} judges ready in {elapsed:.1f}s")


def _kill_judges():
    """Kill all judge server processes."""
    global _judge_processes
    for gpu, port, proc in _judge_processes:
        if proc.poll() is None:
            print(f"  Killing judge on GPU {gpu} port {port} (PID={proc.pid})...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass
    _judge_processes = []


# ============================================================================
# Data Preparation
# ============================================================================
TRAIN_TABLE = "POLICY_RL_TRAIN"
EVAL_TABLE = "POLICY_RL_EVAL"
METADATA_LOOKUP_PATH = "/tmp/metadata_lookup.json"
DATA_DATABASE = "RL_TRAINING_DB"
DATA_SCHEMA = "RL_SCHEMA"


def _get_spcs_token():
    """Read SPCS OAuth token for REST API calls."""
    with open("/snowflake/session/token") as f:
        return f.read().strip()


def _query_table(table_name, host):
    """Query a Snowflake table via REST API and return all rows.

    Handles pagination (multiple result partitions) automatically.
    Returns list of dicts with TRANSCRIPT, METADATA, PROMPT keys.
    """
    import ssl
    import gzip
    import urllib.error

    fq_table = f"{DATA_DATABASE}.{DATA_SCHEMA}.{table_name}"
    url = f"https://{host}/api/v2/statements"
    statement = f"SELECT TRANSCRIPT, METADATA, PROMPT FROM {fq_table}"
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

    def _read_response(resp):
        """Read HTTP response, decompressing gzip if needed."""
        raw = resp.read()
        if raw[:2] == b'\x1f\x8b':
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))

    token = _get_spcs_token()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")

    try:
        resp = urllib.request.urlopen(req, context=ctx, timeout=120)
        result = _read_response(resp)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"  ERROR querying {fq_table}: HTTP {e.code} — {error_body[:500]}")
        raise

    columns = [col["name"] for col in result["resultSetMetaData"]["rowType"]]
    statement_handle = result.get("statementHandle", "")

    all_row_data = result.get("data", [])

    partition_info = result.get("resultSetMetaData", {}).get("partitionInfo", [])
    if len(partition_info) > 1:
        print(f"  {fq_table}: {len(partition_info)} partitions, fetching all...")
        for i in range(1, len(partition_info)):
            part_url = f"https://{host}/api/v2/statements/{statement_handle}?partition={i}"
            token = _get_spcs_token()
            part_req = urllib.request.Request(part_url, method="GET")
            part_req.add_header("Accept", "application/json")
            part_req.add_header("Authorization", f"Bearer {token}")
            try:
                part_resp = urllib.request.urlopen(part_req, context=ctx, timeout=120)
                part_result = _read_response(part_resp)
                all_row_data.extend(part_result.get("data", []))
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")
                print(f"  ERROR fetching partition {i}: HTTP {e.code} — {error_body[:200]}")
                raise

    rows = []
    for row_data in all_row_data:
        row = dict(zip(columns, row_data))
        rows.append(row)

    print(f"  Queried {fq_table}: {len(rows)} rows ({len(partition_info)} partition(s))")
    return rows


def prepare_data():
    """Query training data from Snowflake tables, build metadata lookup."""
    host = os.environ.get("SNOWFLAKE_HOST", "")
    if not host:
        raise RuntimeError("SNOWFLAKE_HOST not set — required for table queries")

    lookup = {}
    for table_name in [TRAIN_TABLE, EVAL_TABLE]:
        rows = _query_table(table_name, host)
        for row in rows:
            transcript = row.get("TRANSCRIPT", "")
            metadata_str = row.get("METADATA", "{}")
            try:
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            except json.JSONDecodeError:
                metadata = {}
            key = hashlib.md5(transcript[:500].encode()).hexdigest()
            lookup[key] = metadata

    with open(METADATA_LOOKUP_PATH, "w") as f:
        json.dump(lookup, f)
    print(f"  Metadata lookup: {len(lookup)} entries -> {METADATA_LOOKUP_PATH}")


# ============================================================================
# Dataset Loading
# ============================================================================
def load_policy_dataset(dataset_config, tokenizer):
    """Load policy dataset from a Snowflake table.

    Queries the table specified in dataset_config.path via REST API
    and returns an HF Dataset with 'question', 'answer', 'messages' columns.
    """
    from datasets import Dataset

    table_name = dataset_config.path
    host = os.environ.get("SNOWFLAKE_HOST", "")
    if not host:
        raise RuntimeError("SNOWFLAKE_HOST not set — required for table queries")

    rows = _query_table(table_name, host)

    records = []
    for row in rows:
        prompt_text = row["PROMPT"]
        metadata_str = row.get("METADATA", "{}")
        records.append({
            "question": prompt_text,
            "answer": metadata_str,
            "messages": [{"role": "user", "content": prompt_text}],
        })

    dataset = Dataset.from_list(records)
    print(f"  Loaded {len(records)} records from {table_name}")
    return dataset


# ============================================================================
# Ray Initialization
# ============================================================================
def init_ray():
    """Start local Ray cluster with available GPUs."""
    import ray

    my_ip = socket.gethostbyname(socket.gethostname())
    num_gpus = int(os.environ.get("NUM_GPUS", NUM_AREAL_GPUS))

    judge_ports_str = ",".join(str(p) for p in JUDGE_PORTS)
    reward_pythonpath = os.environ.get("PYTHONPATH", "")
    env_vars = {
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_DISABLE": "1",
        "HF_HOME": "/tmp/hf_cache",
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        "PYTHONPATH": reward_pythonpath,
        "ENABLE_LLM_JUDGE": "1" if ENABLE_LLM_JUDGE else "0",
    }
    if ENABLE_LLM_JUDGE:
        env_vars["LOCAL_JUDGE_PORTS"] = judge_ports_str
        env_vars["LOCAL_JUDGE_MODEL"] = JUDGE_MODEL
    runtime_env = {"env_vars": env_vars}

    try:
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
        print(f"Connected to Ray cluster: {len(ray.nodes())} nodes, "
              f"{ray.cluster_resources().get('GPU', 0)} GPUs")
        return
    except ConnectionError:
        pass

    print(f"Starting local Ray on {my_ip} with {num_gpus} GPUs...")
    ray_cmd = [
        "ray", "start", "--head",
        f"--node-ip-address={my_ip}",
        "--port=6379",
        f"--num-gpus={num_gpus}",
        "--block",
    ]
    subprocess.Popen(ray_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for attempt in range(20):
        time.sleep(3)
        try:
            ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
            print(f"  Ray started: {ray.cluster_resources().get('GPU', 0)} GPUs available")
            return
        except ConnectionError:
            if attempt < 19:
                print(f"  Waiting for Ray... ({attempt + 1}/20)")
    raise RuntimeError("Failed to start Ray cluster")


# ============================================================================
# AReaL Reward API Patch
# ============================================================================
_NEW_CALL_METHOD = '''\
    _async_reward_fn_resolved = None

    async def __call__(self, *args, **kwargs) -> float:
        """Native async reward — bypasses ProcessPoolExecutor.
        Patched by run_policy_rl.py.
        """
        cls = type(self)
        if cls._async_reward_fn_resolved is None:
            try:
                from policy_rl_local_judges.policy_reward import async_policy_reward_fn
                cls._async_reward_fn_resolved = async_policy_reward_fn
                logger.info("AsyncRewardWrapper: using native async reward path")
            except ImportError:
                logger.warning("async_policy_reward_fn not found, falling back to sync path")
                cls._async_reward_fn_resolved = False

        if cls._async_reward_fn_resolved and cls._async_reward_fn_resolved is not False:
            result = await asyncio.wait_for(
                cls._async_reward_fn_resolved(*args, **kwargs),
                timeout=self.timeout_seconds,
            )
            if isinstance(result, tuple):
                reward, sub_scores = result
                from areal.utils import stats_tracker
                from areal.infra import workflow_context
                scope = workflow_context.stat_scope()
                stats_tracker.get(scope).scalar(**{
                    f"reward/{k}": v for k, v in sub_scores.items()
                })
                return reward
            return result

        # Fallback: original ProcessPoolExecutor path
        with self._lock:
            executor = self._executors.get(self._executor_key)
        if executor is None:
            raise RuntimeError("ProcessPoolExecutor has been shut down")
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            executor, partial(self.reward_fn, *args, **kwargs),
        )
        return await asyncio.wait_for(future, timeout=self.timeout_seconds)'''


def _patch_reward_api():
    """Patch AReaL's AsyncRewardWrapper for native async + longer timeout.

    Modifies the source file so Ray workers (separate processes) pick up
    the changes when they import the module.
    """
    import re as _re

    for path in ["/AReaL/src/areal/api/reward_api.py",
                 "/AReaL/areal/api/reward_api.py"]:
        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            content = f.read()
        original = content

        # Patch 1: timeout 15s -> 300s
        content = _re.sub(
            r'timeout_seconds:\s*float\s*=\s*15\b',
            'timeout_seconds: float = 300',
            content,
        )

        # Patch 2: replace __call__ with native async version
        # Match from "async def __call__" to the end of the method
        # (next unindented def or class, or end of file)
        pattern = r'(    async def __call__\(self.*?\n)(.*?)(?=\n    (?:def |async def |@)|$)'
        match = _re.search(pattern, content, _re.DOTALL)
        if match:
            content = content[:match.start()] + _NEW_CALL_METHOD + content[match.end():]

        if content != original:
            with open(path, "w") as f:
                f.write(content)
            print(f"  Patched {path}: timeout=300s, native async __call__")
        else:
            print(f"  WARNING: No patches matched in {path}")
        return

    print("  WARNING: reward_api.py not found")


# ============================================================================
# Main
# ============================================================================
def main():
    # Single-controller mode
    os.environ.pop("AREAL_SPMD_MODE", None)

    # 1. Start the local vLLM judges (phase 2 only)
    if ENABLE_LLM_JUDGE:
        start_judge_servers()
    else:
        print("\n--- Skipping judge servers (ENABLE_LLM_JUDGE=0, phase 1) ---")

    # 2. Prepare data (query tables, build metadata lookup)
    print("\n--- Data Preparation ---")
    prepare_data()

    # 3. Patch AReaL reward API for native async rewards + longer timeout
    #    Patches the SOURCE FILE so Ray workers pick up changes on import.
    print("\n--- Patching reward_api.py ---")
    _patch_reward_api()

    # 4. Copy reward module to local filesystem (avoid slow stage FUSE scan)
    REWARD_MODULE_DIR = "/tmp/reward_modules"
    stage_src = "/mnt/job_stage/app"
    local_dst = os.path.join(REWARD_MODULE_DIR, "policy_rl_local_judges")
    if os.path.exists(stage_src):
        os.makedirs(REWARD_MODULE_DIR, exist_ok=True)
        if os.path.exists(local_dst):
            shutil.rmtree(local_dst)
        shutil.copytree(stage_src, local_dst)
        print(f"  Copied reward module to {local_dst}")

    # Add to sys.path and PYTHONPATH (for Ray workers)
    sys.path.insert(0, REWARD_MODULE_DIR)
    existing_pypath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{REWARD_MODULE_DIR}:{existing_pypath}" if existing_pypath
        else REWARD_MODULE_DIR
    )
    print(f"  PYTHONPATH={os.environ['PYTHONPATH']}")

    # 5. Initialize Ray
    print("\n--- Ray Init ---")
    init_ray()

    # 6. Import AReaL (after patches)
    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from areal.utils.hf_utils import load_hf_tokenizer

    # 7. Load config
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)

    # 7b. Warm-start from a previous checkpoint on stage (if configured)
    init_stage_path = os.environ.get("INIT_MODEL_STAGE_PATH", "")
    if init_stage_path:
        print(f"\n--- Loading Init Checkpoint ---")
        print(f"  Stage path: {init_stage_path}")
        local_model = "/tmp/init_model"
        if os.path.exists(local_model):
            shutil.rmtree(local_model)
        os.makedirs(local_model, exist_ok=True)

        # Download checkpoint from stage using presigned URLs via SQL REST API.
        # This reuses the existing _get_spcs_token() and REST API pattern —
        # no extra dependencies needed (snowflake.ml.fileset requires pytz).
        import ssl
        import urllib.error
        host = os.environ.get("SNOWFLAKE_HOST", "")
        ctx = ssl.create_default_context()
        api_url = f"https://{host}/api/v2/statements"

        def _run_sql(sql_text):
            """Execute SQL via REST API (matching _query_table pattern)."""
            token = _get_spcs_token()
            payload = json.dumps({
                "statement": sql_text,
                "timeout": 120,
                "resultSetMetaData": {"format": "jsonv2"},
                "warehouse": os.environ.get("CORTEX_WAREHOUSE", "ADMIN_WH"),
                "database": DATA_DATABASE,
                "schema": DATA_SCHEMA,
            }).encode("utf-8")
            req = urllib.request.Request(api_url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")
            req.add_header("Authorization", f"Bearer {token}")
            try:
                resp = urllib.request.urlopen(req, context=ctx, timeout=120)
                return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                print(f"  SQL REST API error: HTTP {e.code} — {body[:500]}")
                raise

        # List files in the stage path
        list_result = _run_sql(f"LIST {init_stage_path}")
        files = [row[0] for row in list_result.get("data", [])]
        print(f"  Found {len(files)} files on stage")

        # Download each file via GET_PRESIGNED_URL
        # Stage path format: @db.schema.stage/prefix/file.ext
        # LIST returns: rl_payload_stage/prefix/.../file.ext (stage short name)
        stage_name = init_stage_path.split("/")[0]  # @db.schema.stage
        stage_root = stage_name.lstrip("@").split(".")[-1].lower()
        for stage_file in files:
            filename = stage_file.split("/")[-1]
            # Strip stage short name prefix to get relative path
            if stage_file.lower().startswith(stage_root):
                rel_path = stage_file[len(stage_root) + 1:]
            else:
                rel_path = stage_file

            url_result = _run_sql(
                f"SELECT GET_PRESIGNED_URL({stage_name}, '{rel_path}')"
            )
            presigned_url = url_result["data"][0][0]

            local_file = os.path.join(local_model, filename)
            urllib.request.urlretrieve(presigned_url, local_file)
            size_mb = os.path.getsize(local_file) / 1e6
            print(f"    {filename}: {size_mb:.1f} MB")

        print(f"  Contents: {os.listdir(local_model)}")

        # Override model paths so AReaL initialises from this checkpoint
        config.actor.path = local_model
        config.ref.path = local_model
        config.vllm.model = local_model
        config.tokenizer_path = local_model
        print(f"  actor.path -> {local_model}")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # 9. Load datasets directly
    print("\n--- Dataset Loading ---")
    train_dataset = load_policy_dataset(config.train_dataset, tokenizer)
    valid_dataset = load_policy_dataset(config.valid_dataset, tokenizer)

    # 10. Configure workflow with decomposed reward function
    reward_fn_path = "policy_rl_local_judges.policy_reward.policy_reward_fn"
    workflow_kwargs = dict(
        reward_fn=reward_fn_path,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=True,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    # 11. Train
    print("\n--- Training ---")
    print(f"  Reward function: {reward_fn_path}")
    print(f"  Judges: {len(JUDGE_GPUS)}x local vLLM @ ports {JUDGE_PORTS}")
    print(f"  Metadata lookup: {METADATA_LOOKUP_PATH}")

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )

    # 12. Export model checkpoint to stage
    print("\n--- Model Export ---")
    checkpoint_dir = config.cluster.fileroot
    stage_output = f"/mnt/job_stage/output/model_output_{config.trial_name}"
    if os.path.exists(checkpoint_dir):
        print(f"  Copying {checkpoint_dir} -> {stage_output}")
        if os.path.exists(stage_output):
            shutil.rmtree(stage_output)
        shutil.copytree(checkpoint_dir, stage_output)
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(stage_output):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
                file_count += 1
        print(f"  Exported {file_count} files, {total_size / 1e9:.2f} GB")
    else:
        print(f"  WARNING: Checkpoint dir not found: {checkpoint_dir}")

    print("\n" + "=" * 60)
    print("Training complete. Model exported to stage.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("=" * 60)
        print("FATAL ERROR:")
        print(traceback.format_exc())
        print("=" * 60)
        # Kill judges on fatal error
        _kill_judges()
        sys.exit(1)
