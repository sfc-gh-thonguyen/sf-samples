#!/usr/bin/env python3
"""
Policy RL Training with Decomposed Reward on SPCS

Trains a model to summarize customer support interactions in compliance with
a multi-rule policy, using a decomposed reward function (deterministic + LLM-judged).

Based on llm_as_judges/run_soap_cortex.py. Key differences:
  1. Loads customer interaction data from Snowflake tables via REST API
  2. Builds metadata lookup for the reward function
  3. Uses policy_rl.policy_reward.policy_reward_fn
  4. Increased reward timeout (Cortex API calls take longer)
"""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time

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

print("=" * 60)
print("Policy RL Training with Decomposed Reward")
print("=" * 60)
print(f"  CORTEX_JUDGE_MODEL={os.environ.get('CORTEX_JUDGE_MODEL', 'llama3.1-8b')}")
print(f"  CORTEX_WAREHOUSE={os.environ.get('CORTEX_WAREHOUSE', 'ADMIN_WH')}")

# ============================================================================
# Data Preparation
# ============================================================================
TRAIN_TABLE = "POLICY_RL_TRAIN"
EVAL_TABLE = "POLICY_RL_EVAL"
METADATA_LOOKUP_PATH = "/tmp/metadata_lookup.json"
# Fully qualified database/schema for table queries
DATA_DATABASE = "RL_TRAINING_DB"
DATA_SCHEMA = "RL_SCHEMA"


def _get_spcs_token():
    """Read SPCS OAuth token for REST API calls."""
    with open("/snowflake/session/token") as f:
        return f.read().strip()


def _query_table(table_name, host):
    """Query a Snowflake table via REST API and return all rows.

    Uses the same REST API pattern as policy_reward.py.
    Handles pagination (multiple result partitions) automatically.
    Returns list of dicts with TRANSCRIPT, METADATA, PROMPT keys.
    """
    import ssl
    import gzip
    import urllib.request
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
        # Check for gzip magic bytes (0x1f, 0x8b)
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

    # Collect rows from partition 0 (included in initial response)
    all_row_data = result.get("data", [])

    # Fetch additional partitions if present
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
    """Query training data from Snowflake tables, build metadata lookup.

    Data is loaded from POLICY_RL_TRAIN and POLICY_RL_EVAL tables via REST API.
    Metadata lookup is saved to /tmp for Ray worker access.
    """
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
# Dataset Loading (bypasses areal.dataset.get_custom_dataset entirely)
# ============================================================================
def load_policy_dataset(dataset_config, tokenizer):
    """Load policy dataset from a Snowflake table.

    Queries the table specified in dataset_config.path (which holds the table name)
    via REST API and returns an HF Dataset with 'question', 'answer', 'messages' columns.
    Data stays in memory — no intermediate files.
    """
    from datasets import Dataset

    table_name = dataset_config.path  # path field holds the table name
    host = os.environ.get("SNOWFLAKE_HOST", "")
    if not host:
        raise RuntimeError("SNOWFLAKE_HOST not set — required for table queries")

    rows = _query_table(table_name, host)

    records = []
    for row in rows:
        prompt_text = row["PROMPT"]
        metadata_str = row.get("METADATA", "{}")
        records.append({
            # Use "question" (not "prompt") to avoid collision —
            # RLVRWorkflow passes the prompt positionally AND forwards
            # all data fields as kwargs to the reward fn.
            "question": prompt_text,
            "answer": metadata_str,
            # RLVRWorkflow.default_data_extract_prompt_fn expects "messages"
            "messages": [{"role": "user", "content": prompt_text}],
        })

    dataset = Dataset.from_list(records)
    print(f"  Loaded {len(records)} records from {table_name}")
    return dataset


# ============================================================================
# SPCS Port Fix
# ============================================================================
SPCS_PORT_START = 12031
SPCS_PORT_END = 50000


def apply_port_patch():
    """Patch AReaL's port allocation to use SPCS-safe ranges."""
    try:
        import areal.utils.network as network_module
        import random

        original_find_free_ports = network_module.find_free_ports

        def patched_find_free_ports(n, port_range=None):
            if port_range is None:
                port_range = (SPCS_PORT_START, SPCS_PORT_END)
            low = max(port_range[0], SPCS_PORT_START)
            high = min(port_range[1], SPCS_PORT_END)
            if low >= high:
                low, high = SPCS_PORT_START, SPCS_PORT_END

            ports = []
            attempts = 0
            while len(ports) < n and attempts < 200:
                port = random.randint(low, high)
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("", port))
                    sock.close()
                    if port not in ports:
                        ports.append(port)
                except OSError:
                    pass
                attempts += 1

            if len(ports) < n:
                return original_find_free_ports(n, port_range=(low, high))
            return ports

        network_module.find_free_ports = patched_find_free_ports
        print(f"  Port patch applied: range {SPCS_PORT_START}-{SPCS_PORT_END}")
    except Exception as e:
        print(f"  Port patch warning: {e}")


# ============================================================================
# Ray Initialization
# ============================================================================
def init_ray():
    """Start local Ray cluster or connect to existing one."""
    import ray

    my_ip = socket.gethostbyname(socket.gethostname())
    num_gpus = int(os.environ.get("NUM_GPUS", 8))

    reward_pythonpath = os.environ.get("PYTHONPATH", "")
    runtime_env = {
        "env_vars": {
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_IB_DISABLE": "1",
            "HF_HOME": "/tmp/hf_cache",
            "TRANSFORMERS_CACHE": "/tmp/hf_cache",
            "PYTHONPATH": reward_pythonpath,
        }
    }

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
# Main
# ============================================================================
def main():
    # Single-controller mode
    os.environ.pop("AREAL_SPMD_MODE", None)

    # 1. Prepare data (query tables, build metadata lookup)
    print("\n--- Data Preparation ---")
    prepare_data()

    # 2. Patch AReaL reward timeout + error handling via Python file patch.
    #    - Increase timeout from 15s to 600s (generous; Cortex retry logic
    #      in policy_reward.py handles transient failures faster)
    #    - Change TimeoutError handler from "return 0" to "raise" so timed-out
    #      trajectories are discarded (not trained on with reward=0, which
    #      after reward_bias becomes an active penalty of -5.0)
    reward_api_path = "/AReaL/src/areal/api/reward_api.py"
    if os.path.exists(reward_api_path):
        subprocess.run(
            ["sed", "-i",
             "s/timeout_seconds: float = 15/timeout_seconds: float = 600/g",
             reward_api_path],
            check=True,
        )
        with open(reward_api_path, "r") as f:
            content = f.read()
        patched = content.replace(
            "return 0",
            "raise  # patched: discard trajectory instead of reward=0",
            1,
        )
        if patched != content:
            with open(reward_api_path, "w") as f:
                f.write(patched)
            print(f"  Patched {reward_api_path}: timeout 15s -> 600s, "
                  f"timeout handler: return 0 -> raise (discard trajectory)")
        else:
            print(f"  WARNING: Could not find 'return 0' in {reward_api_path}")

    # 3. Copy reward module to local filesystem (avoid slow stage FUSE scan)
    REWARD_MODULE_DIR = "/tmp/reward_modules"
    stage_src = "/mnt/job_stage/app"
    local_dst = os.path.join(REWARD_MODULE_DIR, "policy_rl_cortex_judges")
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

    # 4. Apply SPCS patches
    print("\n--- SPCS Patches ---")
    apply_port_patch()

    # 5. Initialize Ray
    print("\n--- Ray Init ---")
    init_ray()

    # 6. Import AReaL (after patches)
    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from areal.utils.hf_utils import load_hf_tokenizer

    # 7. Load config
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # 8. Load datasets directly (no get_custom_dataset / no monkey-patch)
    print("\n--- Dataset Loading ---")
    train_dataset = load_policy_dataset(config.train_dataset, tokenizer)
    valid_dataset = load_policy_dataset(config.valid_dataset, tokenizer)

    # 9. Configure workflow with decomposed reward function
    reward_fn_path = "policy_rl.policy_reward.policy_reward_fn"
    workflow_kwargs = dict(
        reward_fn=reward_fn_path,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    # 11. Train
    print("\n--- Training ---")
    print(f"  Reward function: {reward_fn_path}")
    print(f"  Cortex model: {os.environ.get('CORTEX_JUDGE_MODEL', 'llama3.1-8b')}")
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
    checkpoint_dir = config.cluster.fileroot  # /tmp/areal_policy_v1
    stage_output = "/mnt/job_stage/output/model_output_v11"
    if os.path.exists(checkpoint_dir):
        print(f"  Copying {checkpoint_dir} -> {stage_output}")
        if os.path.exists(stage_output):
            shutil.rmtree(stage_output)
        shutil.copytree(checkpoint_dir, stage_output)
        # Log what was saved
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
        sys.exit(1)
