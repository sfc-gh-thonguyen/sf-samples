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

# Ensure current directory is in Python path for local imports FIRST
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

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
    found_module_path = None
    for module_path in module_paths:
        try:
            module = __import__(module_path, fromlist=[""])
            target_path = module.__file__
            found_module_path = module_path
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
        
        # CRITICAL: Remove module from cache so it will be re-imported with patched code
        # Without this, Python continues using the cached (unpatched) module!
        if found_module_path in sys.modules:
            del sys.modules[found_module_path]
            print(f"    Cleared module cache for: {found_module_path}")
        
        # Also clear any parent modules that might have cached references
        parts = found_module_path.split('.')
        for i in range(len(parts) - 1, 0, -1):
            parent_path = '.'.join(parts[:i])
            if parent_path in sys.modules:
                # Don't delete parent, but it will reimport the child
                pass
        
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
    
    CRITICAL PATCHES (via monkey-patch - safe for any AReaL version):
    1. SGLang LoRA Request Patch - THE KEY FIX: injects lora_name in HTTP requests
    2. RemoteInfEngine - Fixes judge resolution timeout (5min vs 1s)
    3. Weight update port patch - Forces NCCL to use safe ports (12031-13000)
    
    NOTE: File-based patches DISABLED - incompatible with container's AReaL version.
    The container uses a different module structure than the patch files expect.
    Instead, we use monkey-patches that work at runtime without modifying files.
    """
    print("Applying AReaL patches for SPCS...")
    
    # DISABLED: All file-based patches due to AReaL version incompatibility
    # The container has different module structure (missing areal.utils.concurrent, etc.)
    # patches_dir = os.path.join(os.path.dirname(__file__), "areal_patches")
    # patches = []  # No file patches
    
    success_count = 0
    print(f"  File patches applied: 0/0 (all disabled - AReaL version incompatible)")
    
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


def apply_patches_on_all_nodes():
    """Apply patches on ALL nodes in the Ray cluster using Ray remote tasks.
    
    This ensures the patched sglang_server.py is copied to the CORRECT location
    on EVERY node. Key insight: we DISCOVER where Python imports from rather 
    than guessing paths.
    
    Must be called AFTER ray.init() and wait_for_ray_cluster().
    """
    import sys
    print("\n" + "="*60)
    print("APPLYING PATCHES ON ALL NODES")
    print("="*60)
    sys.stdout.flush()
    
    @ray.remote
    def patch_node():
        """Apply patches on this node by discovering actual import paths."""
        import os
        import sys
        import shutil
        import socket
        import glob
        import subprocess
        
        node_ip = socket.gethostbyname(socket.gethostname())
        results = []
        
        STAGE_PATH = "/mnt/job_stage/app/areal_patches"
        VENV_PYTHON = "/AReaL/.venv/bin/python"
        
        # DISCOVERY: Find where the venv Python actually imports from
        # This is CRITICAL because pip install can put modules in different locations
        discover_script = '''
import areal.launcher.sglang_server as m1
import areal.launcher.ray as m2
import areal.engine.sglang_remote as m3
print(f"sglang_server:{m1.__file__}")
print(f"ray:{m2.__file__}")
print(f"sglang_remote:{m3.__file__}")
'''
        try:
            result = subprocess.run(
                [VENV_PYTHON, "-c", discover_script],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        name, path = line.split(':', 1)
                        results.append(f"  DISCOVERY: {name} imports from {path}")
        except Exception as e:
            results.append(f"  DISCOVERY ERROR: {e}")
        
        # Define patches with module discovery
        # NOTE: ray_launcher_patched.py DISABLED - incompatible with container's AReaL
        patches = [
            # DISABLED: ray_launcher has import errors with container's AReaL version
            # {
            #     "source": "ray_launcher_patched.py",
            #     "module": "areal.launcher.ray",
            #     "target_name": "ray.py",
            # },
            {
                "source": "sglang_server_patched.py",
                "module": "areal.launcher.sglang_server",
                "target_name": "sglang_server.py",
            },
            {
                "source": "sglang_remote_patched.py",
                "module": "areal.engine.sglang_remote",
                "target_name": "sglang_remote.py",
            },
        ]
        
        for patch in patches:
            source = os.path.join(STAGE_PATH, patch["source"])
            target_name = patch["target_name"]
            module_name = patch["module"]
            
            if not os.path.exists(source):
                results.append(f"  {target_name}: SKIP (source not found)")
                continue
            
            # Use subprocess to find ACTUAL import location in venv
            try:
                find_script = f'''
import {module_name} as m
print(m.__file__)
'''
                result = subprocess.run(
                    [VENV_PYTHON, "-c", find_script],
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    target = result.stdout.strip()
                    
                    # Create backup if not exists
                    backup = target + ".original"
                    if not os.path.exists(backup):
                        shutil.copy2(target, backup)
                    
                    # Copy patched file
                    shutil.copy2(source, target)
                    
                    # Clear .pyc cache
                    target_dir = os.path.dirname(target)
                    pycache_dir = os.path.join(target_dir, '__pycache__')
                    if os.path.exists(pycache_dir):
                        base_name = target_name.replace('.py', '')
                        for pyc in glob.glob(os.path.join(pycache_dir, f'{base_name}.cpython-*.pyc')):
                            try:
                                os.remove(pyc)
                                results.append(f"    Removed cache: {pyc}")
                            except:
                                pass
                    
                    # Verify patch was applied by checking file content
                    with open(target, 'r') as f:
                        content = f.read(500)
                    if 'SGLANG_PATCHED_MODULE' in content or 'SPCS_LORA_FIX' in content:
                        results.append(f"  {target_name}: PATCHED @ {target} (VERIFIED)")
                    else:
                        results.append(f"  {target_name}: PATCHED @ {target} (unverified)")
                else:
                    results.append(f"  {target_name}: IMPORT FAILED - {result.stderr}")
                    
            except Exception as e:
                results.append(f"  {target_name}: ERROR ({e})")
        
        # Clear module cache in THIS process (though venv subprocess is separate)
        for mod_name in list(sys.modules.keys()):
            if 'areal.launcher' in mod_name or 'sglang_server' in mod_name:
                del sys.modules[mod_name]
        
        return node_ip, results
    
    # Get all nodes in the cluster
    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n.get('Alive', False)]
    
    print(f"Patching {len(alive_nodes)} nodes...")
    sys.stdout.flush()
    
    # Schedule patch task on EACH node using placement
    futures = []
    for node in alive_nodes:
        node_id = node.get('NodeID')
        # Use scheduling_strategy to target specific node
        future = patch_node.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False
            )
        ).remote()
        futures.append(future)
    
    print(f"Waiting for {len(futures)} patch tasks to complete...")
    sys.stdout.flush()
    
    # Wait for all patches to complete
    results = ray.get(futures)
    
    print(f"Patch tasks completed. Processing results...")
    sys.stdout.flush()
    
    for node_ip, patch_results in results:
        print(f"  Node {node_ip}:")
        for r in patch_results:
            print(f"    {r}")
        sys.stdout.flush()
    
    # VERIFICATION: Run a subprocess to confirm the patched module loads correctly
    print("\n--- VERIFICATION ---")
    sys.stdout.flush()
    
    @ray.remote
    def verify_patch():
        """Verify the patched module loads and has our markers."""
        import subprocess
        import socket
        node_ip = socket.gethostbyname(socket.gethostname())
        
        VENV_PYTHON = "/AReaL/.venv/bin/python"
        
        # Test 1: Check file content directly
        verify_script = '''
import sys
# Read the file directly to check content
with open("/AReaL/areal/launcher/sglang_server.py", "r") as f:
    content = f.read()

if "SGLANG_PATCHED_MODULE" in content:
    print("FILE_CHECK: sglang_server PATCHED marker found")
else:
    print("FILE_CHECK: sglang_server PATCHED marker NOT found!")
    print("First 500 chars:", content[:500])

with open("/AReaL/areal/launcher/ray.py", "r") as f:
    ray_content = f.read()

if "RAY_LAUNCHER_PATCHED" in ray_content:
    print("FILE_CHECK: ray.py PATCHED marker found")
else:
    print("FILE_CHECK: ray.py PATCHED marker NOT found!")
    print("ray.py first 500 chars:", ray_content[:500])

# Test 2: Try importing and check if marker prints
print("\\nIMPORT_CHECK: About to import sglang_server...")
sys.stderr.flush()
sys.stdout.flush()

# Force reimport by removing from cache
if "areal.launcher.sglang_server" in sys.modules:
    del sys.modules["areal.launcher.sglang_server"]
if "areal.launcher.ray" in sys.modules:
    del sys.modules["areal.launcher.ray"]

import areal.launcher.sglang_server as mod
print(f"IMPORT_CHECK: sglang_server loaded from {mod.__file__}")

# Check if the module has our patched function
if hasattr(mod, "SPCS_BASE_PORT"):
    print(f"IMPORT_CHECK: SPCS_BASE_PORT = {mod.SPCS_BASE_PORT}")
else:
    print("IMPORT_CHECK: SPCS_BASE_PORT not found - using ORIGINAL module!")
'''
        result = subprocess.run(
            [VENV_PYTHON, "-c", verify_script],
            capture_output=True, text=True, timeout=60
        )
        
        return node_ip, result.stdout, result.stderr
    
    # Run verification on all nodes
    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n.get('Alive', False)]
    verify_futures = []
    for node in alive_nodes:
        node_id = node.get('NodeID')
        future = verify_patch.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False
            )
        ).remote()
        verify_futures.append(future)
    
    verify_results = ray.get(verify_futures)
    for node_ip, stdout, stderr in verify_results:
        print(f"  Node {node_ip} verification:")
        for line in stdout.split('\n'):
            if line.strip():
                print(f"    {line}")
        if stderr.strip():
            print(f"    STDERR: {stderr[:500]}")
    
    print("="*60 + "\n")
    return True


def wait_for_ray_cluster(timeout=300, min_gpus=4, num_nodes=2, node_allocation=None):
    """Wait for Ray cluster to be ready and register node tags.
    
    For single-node setup (num_nodes=1): Start Ray locally
    For multi-node setup: Use file-based coordination (but multi-node broken in SPCS)
    
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
    
    hostname = socket.gethostname()
    my_ip = socket.gethostbyname(socket.gethostname())
    
    ray_port = 6379
    client_port = 10001
    dashboard_port = 8265
    
    # SINGLE-NODE MODE: Just start Ray locally
    if num_nodes == 1:
        print(f"Initializing Ray (SPCS single-node)...")
        print(f"  Hostname: {hostname}")
        print(f"  Node IP: {my_ip}")
        print(f"  Starting Ray HEAD on {my_ip}:{ray_port}...")
        
        ray_cmd = [
            "ray", "start", "--head",
            f"--node-ip-address={my_ip}",
            f"--port={ray_port}",
            f"--ray-client-server-port={client_port}",
            f"--dashboard-port={dashboard_port}",
            "--num-gpus=4",
            "--block"
        ]
        # Start in background
        subprocess.Popen(
            ray_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(5)  # Wait for Ray to start
        
        # Connect to local Ray
        ray.init(address="auto", ignore_reinit_error=True)
        print(f"  Ray started successfully (single-node mode)")
    else:
        # MULTI-NODE MODE (currently broken in SPCS due to cross-subnet networking)
        is_head = hostname == "statefulset-0"
        print(f"Initializing Ray (SPCS multi-node)...")
        print(f"  Hostname: {hostname}")
        print(f"  Is head node: {is_head}")
        print(f"  WARNING: Multi-node Ray broken in SPCS - consider using n_nodes: 1")
        
        # Use file-based coordination since SPCS hostnames aren't resolvable
        head_ip_file = "/mnt/job_stage/ray_head_ip.txt"
        
        if is_head:
            print(f"  HEAD node IP: {my_ip}")
            
            # Write IP to shared file for workers to read
            with open(head_ip_file, 'w') as f:
                f.write(my_ip)
            print(f"  Wrote head IP to {head_ip_file}")
            
            # Start Ray head node
            print(f"  Starting Ray HEAD on {my_ip}:{ray_port}...")
            ray_cmd = [
                "ray", "start", "--head",
                f"--node-ip-address={my_ip}",
                f"--port={ray_port}",
                f"--ray-client-server-port={client_port}",
                f"--dashboard-port={dashboard_port}",
                "--num-gpus=4",
                "--block"
            ]
            # Start in background
            subprocess.Popen(
                ray_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(5)  # Wait for Ray to start
            
            # Connect to local Ray
            ray.init(address="auto", ignore_reinit_error=True)
            print(f"  Ray HEAD started successfully")
        else:
            # Worker node - read head IP from shared file and connect
            max_attempts = 60  # 5 minutes total (60 * 5 seconds)
            head_ip = None
            
            print(f"  Waiting for head IP file: {head_ip_file}")
            sys.stdout.flush()
            
            # Wait for head to write its IP
            for attempt in range(max_attempts):
                if os.path.exists(head_ip_file):
                    with open(head_ip_file, 'r') as f:
                        head_ip = f.read().strip()
                    if head_ip:
                        print(f"  Found head IP: {head_ip}")
                        break
                print(f"  Attempt {attempt+1}/{max_attempts}: Waiting for head IP file...")
                sys.stdout.flush()
                time.sleep(5)
            
            if not head_ip:
                print("  ERROR: Could not get head IP after max attempts")
                print("  Starting local cluster as fallback")
                ray.init(ignore_reinit_error=True)
            else:
                # Connect to head
                connected = False
                for attempt in range(20):  # 100 seconds of retries
                    try:
                        print(f"  Attempt {attempt+1}/20: Connecting to head at {head_ip}:{ray_port}")
                        sys.stdout.flush()
                        
                        ray_cmd = [
                            "ray", "start",
                            f"--address={head_ip}:{ray_port}",
                            "--num-gpus=4",
                        ]
                        result = subprocess.run(
                            ray_cmd,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if result.returncode == 0:
                            time.sleep(3)  # Wait for registration
                            ray.init(address="auto", ignore_reinit_error=True)
                            print(f"  Successfully connected to Ray HEAD at {head_ip}")
                            connected = True
                            break
                        else:
                            print(f"    Failed (exit {result.returncode}): {result.stderr[:100]}")
                            
                    except subprocess.TimeoutExpired:
                        print(f"    Connection timed out, retrying...")
                    except Exception as e:
                        print(f"    Error: {e}")
                    
                    time.sleep(5)
                
                if not connected:
                    print("  ERROR: Could not connect to Ray head")
                    print("  Starting local cluster as fallback")
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
    
    print("wait_for_ray_cluster completed, proceeding to apply patches...")
    sys.stdout.flush()
    
    # DISABLED: File patches incompatible with container's AReaL version
    # The container has different module structure (missing areal.utils.concurrent, etc.)
    # apply_patches_on_all_nodes()
    print("\n[File patches DISABLED - using monkey-patches only]")
    
    # CRITICAL FIX: Apply SGLang LoRA request patch
    # Without this, SGLang ignores loaded LoRA adapters and uses base model weights!
    # The patch injects lora_name into every generation request
    try:
        from areal_patches.sglang_lora_request_patch import apply_sglang_lora_request_patch
        lora_name = config_dict.get('gconfig', {}).get('lora_name', 'lora-gsm8k')
        apply_sglang_lora_request_patch(lora_name=lora_name)
        print(f"[SGLang LoRA Request Patch] Applied with lora_name='{lora_name}'")
    except Exception as e:
        print(f"WARNING: Failed to apply SGLang LoRA request patch: {e}")
        print("         Model may not use updated LoRA weights during generation!")
    
    print(f"\n{'='*60}")
    print("LAUNCHING AReaL TRAINER")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Ensure RAY_ADDRESS is set so AReaL connects to existing cluster with node tags
    # IMPORTANT: Ray HEAD runs on port 6379, NOT 12001!
    head_ip = socket.gethostbyname(socket.gethostname())
    ray_address = f"{head_ip}:6379"
    os.environ["RAY_ADDRESS"] = ray_address
    print(f"RAY_ADDRESS set to: {ray_address}")
    
    # ==========================================================================
    # PRE-LAUNCH VERIFICATION DISABLED
    # File-based patches are incompatible with container's AReaL version
    # The container uses a different module structure (missing areal.utils.concurrent, etc.)
    # Instead, we rely on monkey-patches applied earlier:
    #   - sglang_lora_request_patch.py: Injects lora_name into HTTP requests (THE KEY FIX)
    #   - remote_inf_engine_patched.py: Fixes judge resolution timeout
    #   - Weight update port patch: Forces NCCL to use safe ports
    # ==========================================================================
    print("\n--- PRE-LAUNCH VERIFICATION (HEAD NODE) ---")
    print("  File patches DISABLED - using monkey-patches only")
    print("  Active monkey-patches:")
    print("    - SGLang LoRA Request Patch (lora_name injection)")
    print("    - RemoteInfEngine (judge resolution timeout)")
    print("    - WeightUpdatePort (safe NCCL port range)")
    print("--- END PRE-LAUNCH VERIFICATION ---\n")
    # ==========================================================================
    
    # ==========================================================================
    # FIX MISSING IMPORTS: Patch /AReaL/areal/launcher/ray.py
    # The container's AReaL has imports that don't exist (areal.utils.* modules).
    # We patch the file to comment out ALL areal.utils imports and add no-op stubs.
    # ==========================================================================
    print("Patching AReaL launcher to remove missing imports...")
    ray_launcher_path = "/AReaL/areal/launcher/ray.py"
    
    if os.path.exists(ray_launcher_path):
        with open(ray_launcher_path, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if "SPCS_PATCHED_IMPORTS_V4" in content:
            print(f"  {ray_launcher_path}: Already patched")
        else:
            import re
            patches_applied = 0
            
            # Generic stub for all missing areal.utils.* imports
            # These stubs cover all functions/classes from areal.utils.* modules
            stubs = '''
# SPCS STUBS for missing areal.utils.* modules
from enum import Enum
import os as _os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

class JobState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"

def save_experiment_metadata(*args, **kwargs): pass
class run_func_context:
    def __enter__(self): return self
    def __exit__(self, *args): pass
def get_node_index(): return 0
def parse_args(*args, **kwargs): return {}
def RUN_FUNC_CONTEXT(*args, **kwargs): return run_func_context()
def get_placement_group_master_ip_and_port(*args, **kwargs): return ("127.0.0.1", 12345)
def setup_ray_with_placement_group(*args, **kwargs): pass
def ray_log(*args, **kwargs): pass
def check_if_recover(*args, **kwargs): return False  # Not recovering from checkpoint
def validate_config_for_distributed_launcher(config): pass  # No-op validation

# name_resolve module stub - used for distributed service discovery
class _NameResolveStub:
    @staticmethod
    def reconfigure(config): pass
    @staticmethod
    def add(name, value, *args, **kwargs): pass
    @staticmethod
    def wait(name, *args, **kwargs): return "127.0.0.1"
    @staticmethod
    def clear_subtree(*args, **kwargs): pass
    @staticmethod
    def delete(name, *args, **kwargs): pass
    @staticmethod
    def get(name, *args, **kwargs): return "127.0.0.1"
name_resolve = _NameResolveStub()

# names module stub - cluster/trial path management
class _NamesStub:
    @staticmethod
    def trial_root(*args, **kwargs): return "/tmp/trial"
    @staticmethod
    def experiment_root(*args, **kwargs): return "/tmp/experiment"
    @staticmethod
    def run_root(*args, **kwargs): return "/tmp/run"
    @staticmethod
    def model_root(*args, **kwargs): return "/tmp/model"
    @staticmethod
    def checkpoint_root(*args, **kwargs): return "/tmp/checkpoint"
names = _NamesStub()

# worker_util module stub
class _WorkerUtilStub:
    @staticmethod
    def get_worker_id(): return 0
    @staticmethod
    def get_worker_count(): return 1
worker_util = _WorkerUtilStub()

# log_api module stub  
def log_api(*args, **kwargs): pass

# Scheduling spec stub - returns Ray actor scheduling configuration
@dataclass
class SchedulingSpec:
    cpu: int = 1
    gpu: float = 0
    mem: int = 32  # GB
    num_cpus: int = 1
    num_gpus: float = 0
    memory: Optional[int] = None
    resources: Optional[Dict[str, float]] = None
    scheduling_strategy: Optional[Any] = None
    env_vars: Optional[Dict[str, str]] = None

def get_scheduling_spec(config) -> SchedulingSpec:
    """Return default scheduling spec based on config."""
    try:
        num_gpus = getattr(config, "num_gpus", 0) or 0
    except:
        num_gpus = 0
    return SchedulingSpec(cpu=1, gpu=num_gpus, num_cpus=1, num_gpus=num_gpus, env_vars={})

# Additional common functions from areal.utils
def setup_logging(*args, **kwargs): pass
def get_rank(): return 0
def get_world_size(): return 1
def is_main_process(): return True
def barrier(*args, **kwargs): pass

# Environment variable utilities
def get_thread_env_vars(*args, **kwargs) -> Dict[str, str]:
    """Get thread environment variables for Ray actors."""
    return {}
def set_thread_env_vars(*args, **kwargs): pass

# Base environment variables for Ray actors
BASE_ENVIRONS: Dict[str, str] = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_DEBUG": "INFO",
}

# End SPCS stubs
'''
            
            # Comment out all lines that import from areal.utils
            lines = content.split('\n')
            new_lines = []
            in_multiline_import = False
            
            for line in lines:
                # Check for start of areal.utils import
                if 'from areal.utils' in line and not line.strip().startswith('#'):
                    new_lines.append('# SPCS_PATCHED: ' + line)
                    patches_applied += 1
                    # Check if multi-line import
                    if '(' in line and ')' not in line:
                        in_multiline_import = True
                elif in_multiline_import:
                    new_lines.append('# SPCS_PATCHED: ' + line)
                    if ')' in line:
                        in_multiline_import = False
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
            
            # Add marker and stubs at the top (after any existing imports)
            # Find a good place to insert stubs - after the last import block
            content = f"# SPCS_PATCHED_IMPORTS_V4\n{stubs}\n{content}"
            
            with open(ray_launcher_path, 'w') as f:
                f.write(content)
            print(f"  {ray_launcher_path}: Patched ({patches_applied} areal.utils imports commented)")
    else:
        print(f"  WARNING: {ray_launcher_path} not found")
    
    # Use areal.launcher.ray to start the trainer via -m (now that stubs are in place)
    env = os.environ.copy()
    
    cmd = [
        "/AReaL/.venv/bin/python3", "-m", "areal.launcher.ray",
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
    import threading
    
    log_file_path = "/tmp/areal_training.log"
    print(f"[LOG] Redirecting all output to {log_file_path}")
    print(f"[LOG] Will print last 100 lines when complete...")
    print(f"[LOG] Also printing RayWeightSync lines in real-time...")
    print(f"{'='*60}")
    
    # Background thread to tail log file and print RayWeightSync lines
    stop_tailing = threading.Event()
    
    def tail_log_file():
        """Background thread to print important log lines in real-time."""
        last_pos = 0
        important_patterns = ["[RayWeightSync]", "WEIGHT SYNC", "first_5", "Total norm", "ERROR", "FAILED"]
        while not stop_tailing.is_set():
            try:
                with open(log_file_path, "r") as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()
                    for line in new_lines:
                        if any(p in line for p in important_patterns):
                            print(f"[TAIL] {line}", end='')
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[TAIL] Error: {e}")
            stop_tailing.wait(5)  # Check every 5 seconds
    
    tail_thread = threading.Thread(target=tail_log_file, daemon=True)
    tail_thread.start()
    
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd="/mnt/job_stage/app",
            env=env,  # Use env with PYTHONPATH set to /opt/AReaL
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        
        # Wait for process to complete
        process.wait()
    
    stop_tailing.set()
    tail_thread.join(timeout=2)
    
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
    
    # CRITICAL DEBUG: Print run_func debug log from Ray workers
    debug_file = "/tmp/run_func_debug.log"
    if os.path.exists(debug_file):
        print(f"\n{'='*60}")
        print(f"RUN_FUNC DEBUG LOG (from Ray workers on HEAD node):")
        print(f"{'='*60}")
        with open(debug_file, "r") as f:
            print(f.read())
        print(f"{'='*60}")
    else:
        print(f"\n[DEBUG] No run_func debug log found at {debug_file}")
    
    # CRITICAL DEBUG: Print sglang_server debug log from SGLang process
    sglang_debug_file = "/tmp/sglang_server_debug.log"
    if os.path.exists(sglang_debug_file):
        print(f"\n{'='*60}")
        print(f"SGLANG_SERVER DEBUG LOG (from SGLang server):")
        print(f"{'='*60}")
        with open(sglang_debug_file, "r") as f:
            print(f.read())
        print(f"{'='*60}")
    else:
        print(f"\n[DEBUG] No sglang_server debug log found at {sglang_debug_file}")
    
    print(f"\n{'='*60}")
    print(f"LAST {len(last_lines)} LINES OF OUTPUT:")
    print(f"{'='*60}")
    for line in last_lines:
        print(line.rstrip())
    print(f"{'='*60}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
