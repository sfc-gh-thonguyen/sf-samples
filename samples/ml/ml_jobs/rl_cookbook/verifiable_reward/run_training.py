#!/usr/bin/env python3
"""
AReaL GRPO Training on Snowpark Container Services (SPCS)

Minimal entrypoint for running GRPO training using AReaL's single-controller mode.
"""
import os
import sys
import socket
import subprocess
import time
import json

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

# Network: SPCS uses eth0, no InfiniBand
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"

# HuggingFace: Use local disk to avoid stage mount issues
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_cache"

# Prevent Ray from hiding GPUs when num_gpus=0 on controller
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

print("SPCS Configuration:")
print(f"  NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME')}")
print(f"  HF_HOME={os.environ.get('HF_HOME')}")

# ============================================================================
# SPCS Port Fix: Patch find_free_ports for safe port range
# ============================================================================
# SPCS restricts cross-pod communication to specific port ranges.
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
                    sock.bind(('', port))
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
        print("  Port patch applied: SPCS safe range 12031-50000")
    except Exception as e:
        print(f"  Port patch warning: {e}")


# ============================================================================
# Ray Initialization
# ============================================================================
def init_ray():
    """Start local Ray cluster or connect to existing one."""
    import ray

    my_ip = socket.gethostbyname(socket.gethostname())
    num_gpus = int(os.environ.get("NUM_GPUS", 4))

    runtime_env = {
        "env_vars": {
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_IB_DISABLE": "1",
            "HF_HOME": "/tmp/hf_cache",
            "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        }
    }

    # Try connecting to existing Ray cluster
    try:
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
        print(f"Connected to Ray cluster: {len(ray.nodes())} nodes, {ray.cluster_resources().get('GPU', 0)} GPUs")
        return
    except ConnectionError:
        pass

    # Start local Ray
    print(f"Starting local Ray on {my_ip} with {num_gpus} GPUs...")
    ray_cmd = [
        "ray", "start", "--head",
        f"--node-ip-address={my_ip}",
        "--port=6379",
        f"--num-gpus={num_gpus}",
        "--block"
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
    print("=" * 60)
    print("AReaL GRPO Training on SPCS")
    print("=" * 60)

    # Single-controller mode requires AREAL_SPMD_MODE unset
    os.environ.pop("AREAL_SPMD_MODE", None)

    # Apply SPCS patches
    apply_port_patch()

    # Initialize Ray
    init_ray()

    # Import AReaL after patches
    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from areal.dataset import get_custom_dataset
    from areal.utils.hf_utils import load_hf_tokenizer

    # Load config from CLI args
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load datasets
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    # Configure workflow
    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    # Run training
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
