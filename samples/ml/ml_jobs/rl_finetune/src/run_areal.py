#!/usr/bin/env python3
"""
AReaL GRPO Training Launcher for SPCS.

Supports single-node, 2-node, and 3-node configurations.

3-node mode (set AREAL_3NODE_MODE=1):
  - Node 1 (judge):   External judge server (separate job)
  - Node 2 (rollout): SGLang policy rollout 
  - Node 3 (trainer): GRPO training

2-node mode (default):
  - Node 1 (head):   SGLang rollout
  - Node 2 (worker): Training
"""
import os
import sys
import socket
import time
import subprocess
import shutil
import ray

# ============================================================================
# CRITICAL SPCS FIX: Set NCCL environment variables BEFORE importing AReaL
# ============================================================================
# SPCS has multiple network interfaces, but only eth0 (pod network) is routable
# between nodes. The sandbox interfaces (SandboxVeth, InstSandboxVeth) are NOT
# routable and will cause NCCL connection failures.
#
# These must be set BEFORE any NCCL operations are performed.
# ============================================================================
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Force eth0 (pod network)
os.environ["NCCL_IB_DISABLE"] = "1"        # No InfiniBand in SPCS
os.environ["NCCL_DEBUG"] = "INFO"          # Enable debug logging
print("SPCS NCCL Config:")
print(f"  NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME')}")
print(f"  NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE')}")

# Node tag constants (must match ray_launcher_patched.py)
# Legacy tags (deprecated, kept for backward compatibility)
NODE_TAG_JUDGE = "judge"
NODE_TAG_ROLLOUT = "rollout"
NODE_TAG_TRAINER = "trainer"
NODE_TAG_HEAD = "head"
NODE_TAG_WORKER = "worker"

# Import the new unified node allocation module
from node_allocation import NodeAllocation, parse_node_allocation, derive_from_legacy_config


def apply_single_patch(patch_name, patch_file, module_paths):
    """Apply a single patch file to an AReaL module.
    
    Args:
        patch_name: Human-readable name for the patch
        patch_file: Path to the patch file
        module_paths: List of possible module paths to try
        
    Returns:
        True if patch was applied successfully
    """
    if not os.path.exists(patch_file):
        print(f"  [{patch_name}] No patch file found")
        return False
    
    # Find the target location
    target_path = None
    for module_path in module_paths:
        try:
            module = __import__(module_path, fromlist=[""])
            target_path = module.__file__
            break
        except ImportError:
            continue
    
    if target_path is None:
        print(f"  [{patch_name}] WARNING: Could not find target module")
        return False
    
    print(f"  [{patch_name}]")
    print(f"    Source: {patch_file}")
    print(f"    Target: {target_path}")
    
    try:
        # Backup original
        backup_path = target_path + ".original"
        if not os.path.exists(backup_path):
            shutil.copy(target_path, backup_path)
            print(f"    Backup: {backup_path}")
        
        # Apply patch
        shutil.copy(patch_file, target_path)
        print("    Applied successfully!")
        return True
    except PermissionError as e:
        print(f"    WARNING: Permission denied: {e}")
        return False
    except Exception as e:
        print(f"    WARNING: Failed to apply: {e}")
        return False


def apply_remote_inf_engine_patch():
    """Apply monkey-patch to RemoteInfEngine for proper judge resolution.
    
    This patch:
    - Increases judge resolution timeout from 1s to 5 minutes
    - FAILS with error instead of silently falling back to rollout servers
    """
    try:
        from areal_patches.remote_inf_engine_patched import apply_remote_inf_engine_patch as apply_patch
        return apply_patch()
    except ImportError as e:
        print(f"  [RemoteInfEngine] WARNING: Could not import patch: {e}")
        return False


def apply_weight_update_port_patch():
    """Patch FSDP engine to use safe port range for weight update NCCL group.
    
    SPCS blocks most ports for cross-node communication.
    Safe range is 12031-13000 for torch distributed / weight sync.
    """
    try:
        # Import the network utility module
        import areal.utils.network as network_module
        
        # Store original find_free_ports
        original_find_free_ports = network_module.find_free_ports
        
        # SPCS safe port range for weight sync
        SPCS_WEIGHT_SYNC_PORT_START = 12031
        SPCS_WEIGHT_SYNC_PORT_END = 13000
        
        def patched_find_free_ports(n, port_range=None):
            """Patched version that uses SPCS-safe port range for weight sync."""
            import socket
            import random
            
            # Use SPCS safe range
            safe_range = (SPCS_WEIGHT_SYNC_PORT_START, SPCS_WEIGHT_SYNC_PORT_END)
            
            ports = []
            attempts = 0
            max_attempts = 100
            
            while len(ports) < n and attempts < max_attempts:
                port = random.randint(safe_range[0], safe_range[1])
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
                # Fallback to original with our range
                return original_find_free_ports(n, port_range=safe_range)
            
            print(f"SPCS PATCH: Using weight sync ports: {ports} (safe range {safe_range[0]}-{safe_range[1]})")
            return ports
        
        # Apply monkey-patch
        network_module.find_free_ports = patched_find_free_ports
        print("  [WeightUpdatePort] Applied (find_free_ports → safe range 12031-13000)")
        return True
        
    except Exception as e:
        print(f"  [WeightUpdatePort] WARNING: Could not apply patch: {e}")
        return False


def apply_areal_patches():
    """Apply patches to AReaL modules to fix bugs for SPCS.
    
    Patches applied:
    1. ray_launcher_patched.py - Fixes colocated mode and node affinity
    2. sglang_server_patched.py - Fixes port range for SPCS (34000-50000)
    3. remote_inf_engine_patched.py - Fixes judge resolution timeout (5min vs 1s)
       Applied via monkey-patch since it only modifies the initialize() method
    4. Weight update port patch - Forces NCCL weight sync to use safe ports (12031-13000)
    """
    print("Applying AReaL patches for SPCS...")
    patches_dir = os.path.join(os.path.dirname(__file__), "areal_patches")
    
    patches = [
        {
            "name": "Ray Launcher",
            "file": os.path.join(patches_dir, "ray_launcher_patched.py"),
            "modules": ["areal.infra.launcher.ray", "areal.launcher.ray"],
        },
        {
            "name": "SGLang Server (SPCS port fix)",
            "file": os.path.join(patches_dir, "sglang_server_patched.py"),
            "modules": ["areal.infra.launcher.sglang_server", "areal.launcher.sglang_server"],
        },
    ]
    
    results = []
    for patch in patches:
        success = apply_single_patch(patch["name"], patch["file"], patch["modules"])
        results.append(success)
    
    success_count = sum(results)
    print(f"  File patches applied: {success_count}/{len(patches)}")
    
    # Apply monkey-patch for RemoteInfEngine (judge resolution fix)
    print("  [RemoteInfEngine (Judge resolution fix)]")
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from areal_patches.remote_inf_engine_patched import apply_remote_inf_engine_patch
        if apply_remote_inf_engine_patch():
            success_count += 1
            print("    Applied successfully (monkey-patch)!")
        else:
            print("    WARNING: Monkey-patch failed")
    except Exception as e:
        print(f"    WARNING: Could not apply: {e}")
    
    # Apply weight update port patch (force NCCL to use safe ports)
    print("  [WeightUpdatePort (NCCL safe port range)]")
    if apply_weight_update_port_patch():
        success_count += 1
    
    # Apply Ray Weight Sync patch (cross-node LoRA using Ray object store)
    # Note: The actual patch is applied in run_func on the trainer nodes
    # Here we just verify the module is importable
    print("  [Ray Weight Sync (cross-node LoRA)]")
    try:
        from areal_patches.ray_weight_sync import apply_ray_weight_sync_patch
        print("    Module importable - patch will be applied in trainer tasks")
        success_count += 1
    except Exception as e:
        print(f"    WARNING: Could not import: {e}")
    
    return success_count > 0


def wait_for_ray_cluster(timeout=300, min_gpus=4, num_nodes=2, node_allocation=None):
    """Wait for Ray cluster to be ready and register node tags.
    
    Args:
        timeout: Max seconds to wait
        min_gpus: Minimum GPUs required
        num_nodes: Expected number of nodes
        node_allocation: NodeAllocation instance for role-based tag registration.
                        If None, defaults to 1 rollout + (num_nodes-1) trainer.
    """
    if node_allocation is None:
        node_allocation = NodeAllocation(
            total=num_nodes,
            roles={"rollout": 1, "trainer": max(1, num_nodes - 1)}
        )
    print("Initializing Ray...")
    
    ray_address = os.getenv("RAY_ADDRESS")
    if ray_address:
        print(f"  Connecting to existing cluster at: {ray_address}")
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)
    
    print(f"Ray version: {ray.__version__}")
    
    start = time.time()
    while time.time() - start < timeout:
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if n.get('Alive', False)]
        
        total_gpus = sum(n.get('Resources', {}).get('GPU', 0) for n in alive_nodes)
        
        print(f"  Nodes: {len(alive_nodes)}, Total GPUs: {int(total_gpus)}")
        
        # Print node details
        for node in alive_nodes:
            node_ip = node.get('NodeManagerAddress', 'unknown')
            node_gpus = node.get('Resources', {}).get('GPU', 0)
            resources = node.get('Resources', {})
            tags = [k for k in resources.keys() if k.startswith('node_tag:')]
            tags_str = f" tags={tags}" if tags else ""
            print(f"    - {node_ip}: {int(node_gpus)} GPUs{tags_str}")
        
        if total_gpus >= min_gpus and len(alive_nodes) >= num_nodes:
            print(f"Cluster ready with {int(total_gpus)} GPUs across {len(alive_nodes)} nodes")
            
            # Register node tags for deterministic scheduling using NodeAllocation
            register_node_tags_unified(alive_nodes, node_allocation)
            
            # IMPORTANT: Do NOT call ray.shutdown() here!
            # The node tags are registered as cluster resources and must persist
            # for the AReaL launcher to use them. Shutting down would lose the tags.
            # ray.shutdown()  # <-- REMOVED
            return True
        
        time.sleep(5)
    
    print(f"WARNING: Timeout waiting for GPUs")
    # ray.shutdown()  # Don't shutdown - let cluster persist
    return False


def register_node_tags_unified(nodes: list, node_allocation: NodeAllocation):
    """Register node tags as custom Ray resources based on NodeAllocation.
    
    This is the unified version that supports N nodes with arbitrary role counts.
    Tags are assigned in order: first all rollout nodes, then all trainer nodes.
    
    For single-node roles: node_tag:rollout, node_tag:trainer
    For multi-node roles: node_tag:rollout_0, node_tag:rollout_1, etc.
    
    Args:
        nodes: List of Ray node dicts from ray.nodes()
        node_allocation: NodeAllocation defining role counts
    """
    import sys
    
    if len(nodes) < node_allocation.total:
        print(f"  WARNING: Expected {node_allocation.total} nodes but only found {len(nodes)}")
        sys.stdout.flush()
        if len(nodes) < 2:
            print("  Only 1 node - skipping node tag registration")
            sys.stdout.flush()
            return
    
    # Sort nodes by IP for deterministic ordering
    sorted_nodes = sorted(nodes, key=lambda n: n.get('NodeManagerAddress', ''))
    
    # Get tag names from NodeAllocation
    tags = node_allocation.get_node_tags()
    
    print(f"  Registering node tags (unified mode)...")
    print(f"  Roles: {node_allocation.roles}")
    print(f"  Tags to register: {tags}")
    sys.stdout.flush()
    
    for i, tag in enumerate(tags):
        if i >= len(sorted_nodes):
            print(f"    WARNING: No node available for tag '{tag}'")
            sys.stdout.flush()
            break
            
        node = sorted_nodes[i]
        node_id = node.get('NodeID')
        node_ip = node.get('NodeManagerAddress', 'unknown')
        resource_key = f"node_tag:{tag}"
        
        # Check if already registered
        existing = node.get('Resources', {}).get(resource_key, 0)
        if existing > 0:
            print(f"    {node_ip}: {resource_key} already registered ({existing})")
            sys.stdout.flush()
            continue
        
        try:
            ray.experimental.set_resource(resource_key, 1.0, node_id)
            print(f"    {node_ip}: Registered {resource_key} = 1.0 (node_id={node_id[:8]}...)")
            sys.stdout.flush()
        except Exception as e:
            print(f"    {node_ip}: WARNING: Failed to register {resource_key}: {e}")
            sys.stdout.flush()
    
    # Also register legacy tags for backward compatibility with launcher
    # This ensures old code checking for 'head'/'worker' still works
    _register_legacy_compat_tags(sorted_nodes, node_allocation)
    
    # Wait for resource propagation
    print("  Waiting 3s for resource propagation...")
    sys.stdout.flush()
    time.sleep(3)
    
    # Verify registration
    print("  Verifying node tags...")
    sys.stdout.flush()
    nodes_refreshed = ray.nodes()
    for node in nodes_refreshed:
        if node.get('Alive', False):
            node_ip = node.get('NodeManagerAddress', 'unknown')
            resources = node.get('Resources', {})
            tags = {k: v for k, v in resources.items() if k.startswith('node_tag:')}
            if tags:
                print(f"    {node_ip}: {tags}")
                sys.stdout.flush()
            else:
                print(f"    {node_ip}: NO TAGS FOUND!")
                sys.stdout.flush()


def _register_legacy_compat_tags(sorted_nodes: list, node_allocation: NodeAllocation):
    """Register legacy 'head'/'worker' tags for backward compatibility.
    
    Maps: rollout -> head, trainer -> worker (for 2-node setups)
    This ensures existing code in ray_launcher_patched.py that checks
    for node_tag:head/worker continues to work during migration.
    """
    if node_allocation.total != 2:
        return
    
    legacy_mapping = [
        ("head", 0),    # First node (rollout) also gets 'head' tag
        ("worker", 1),  # Second node (trainer) also gets 'worker' tag  
    ]
    
    print("  Registering legacy compatibility tags (head/worker)...")
    for tag, idx in legacy_mapping:
        if idx >= len(sorted_nodes):
            continue
        node = sorted_nodes[idx]
        node_id = node.get('NodeID')
        resource_key = f"node_tag:{tag}"
        
        existing = node.get('Resources', {}).get(resource_key, 0)
        if existing > 0:
            continue
            
        try:
            ray.experimental.set_resource(resource_key, 1.0, node_id)
        except Exception:
            pass


def register_node_tags(nodes, use_3node_mode=False):
    """DEPRECATED: Use register_node_tags_unified instead.
    
    Kept for backward compatibility. Converts to NodeAllocation and calls unified version.
    """
    import warnings
    warnings.warn(
        "register_node_tags() is deprecated. Use register_node_tags_unified() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    num_nodes = len([n for n in nodes if n.get('Alive', False)])
    node_allocation = NodeAllocation(
        total=num_nodes,
        roles={"rollout": 1, "trainer": max(1, num_nodes - 1)}
    )
    register_node_tags_unified(nodes, node_allocation)


def load_config_for_node_allocation(config_path: str) -> dict:
    """Load YAML config file and return as dict for node allocation parsing."""
    import yaml
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"WARNING: Could not load config for node allocation: {e}")
        return {}


def main():
    print("=" * 60)
    print("AReaL GRPO Training - SPCS (Unified Multi-Node)")
    print("=" * 60)
    
    head_ip = socket.gethostbyname(socket.gethostname())
    print(f"Host IP: {head_ip}")
    
    if len(sys.argv) < 2:
        print("Usage: python run_areal.py <config.yaml> [trainer.py]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    trainer_file = sys.argv[2] if len(sys.argv) > 2 else "soap_grpo_trainer.py"
    config_path = os.path.join("/mnt/job_stage/app", config_file)
    
    if not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(__file__), config_file)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
    
    print(f"Config: {config_path}")
    print(f"Trainer: {trainer_file}")
    
    # Parse node allocation from config (unified approach)
    config_dict = load_config_for_node_allocation(config_path)
    node_allocation = parse_node_allocation(config_dict)
    
    # Validate and display node allocation
    try:
        node_allocation.validate()
    except ValueError as e:
        print(f"ERROR: Invalid node allocation: {e}")
        sys.exit(1)
    
    print(f"\nNODE ALLOCATION (unified):")
    print(f"  Total nodes: {node_allocation.total}")
    print(f"  Roles: {node_allocation.roles}")
    for role, count in node_allocation.roles.items():
        if count == 1:
            print(f"    - {role}: 1 node")
        else:
            print(f"    - {role}: {count} nodes ({role}_0 to {role}_{count-1})")
    
    if node_allocation.external_judge.enabled:
        print(f"  External judge: enabled")
        print(f"    - experiment: {node_allocation.external_judge.experiment_name}")
        print(f"    - trial: {node_allocation.external_judge.trial_name}")
    
    # Apply patches to fix colocated mode bug BEFORE launching AReaL
    apply_areal_patches()
    
    # Pre-create necessary directories for disk-based weight sync
    fileroot = "/mnt/job_stage/checkpoints"
    exp_name = config_dict.get('experiment_name', 'math-test')
    trial_name = config_dict.get('trial_name', 'v1')
    os.makedirs(f"{fileroot}/{exp_name}/{trial_name}/checkpoints/root/{exp_name}/{trial_name}/default/weight_update", exist_ok=True)
    print(f"Pre-created weight_update directory")
    
    # Determine expected cluster size from config (fallback to env var for backward compat)
    num_nodes = node_allocation.total
    env_num_nodes = os.environ.get("AREAL_NUM_NODES")
    if env_num_nodes:
        env_num = int(env_num_nodes)
        if env_num != num_nodes:
            print(f"WARNING: AREAL_NUM_NODES={env_num} differs from config nodes.total={num_nodes}")
            print(f"         Using config value: {num_nodes}")
    
    min_gpus = num_nodes * 4  # Assume 4 GPUs per node
    
    # Wait for cluster and register node tags using unified NodeAllocation
    wait_for_ray_cluster(
        timeout=300, 
        min_gpus=min_gpus, 
        num_nodes=num_nodes,
        node_allocation=node_allocation
    )
    
    print(f"\n{'='*60}")
    print("LAUNCHING AReaL TRAINER")
    print(f"{'='*60}")
    
    # Ensure RAY_ADDRESS is set so AReaL connects to existing cluster with node tags
    head_ip = socket.gethostbyname(socket.gethostname())
    ray_address = f"{head_ip}:12001"
    os.environ["RAY_ADDRESS"] = ray_address
    print(f"RAY_ADDRESS set to: {ray_address}")
    
    # Use areal.launcher.ray to start the trainer
    cmd = [
        sys.executable, "-m", "areal.launcher.ray",
        trainer_file,
        "--config", config_path,
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    # ==========================================================================
    # SPCS LOG TRUNCATION FIX:
    # SPCS truncates logs at 16KB from the BEGINNING, so we never see errors.
    # Solution: Redirect ALL output to a temp file, then print only last N lines.
    # This guarantees we always see the error summary.
    # ==========================================================================
    import tempfile
    
    log_file_path = "/tmp/areal_training.log"
    print(f"[LOG] Redirecting all output to {log_file_path}")
    print(f"[LOG] Will print last 100 lines when complete...")
    print(f"{'='*60}")
    
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd="/mnt/job_stage/app",
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        
        # Wait for process to complete
        process.wait()
    
    elapsed = time.time() - start_time
    exit_code = process.returncode
    
    # Read last N lines from log file
    LAST_N_LINES = 100
    with open(log_file_path, "r") as f:
        all_lines = f.readlines()
    
    total_lines = len(all_lines)
    last_lines = all_lines[-LAST_N_LINES:] if total_lines > LAST_N_LINES else all_lines
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TRAINING {'COMPLETED' if exit_code == 0 else 'FAILED'}")
    print(f"{'='*60}")
    print(f"Exit code: {exit_code}")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Total log lines: {total_lines}")
    print(f"\n{'='*60}")
    print(f"LAST {len(last_lines)} LINES OF OUTPUT:")
    print(f"{'='*60}")
    for line in last_lines:
        print(line.rstrip())
    print(f"{'='*60}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
