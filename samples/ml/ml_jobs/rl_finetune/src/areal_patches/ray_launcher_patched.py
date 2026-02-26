import importlib.util
import os
import pathlib
import re
import sys
import time
from collections.abc import Callable
from functools import partial

import ray
import ray.exceptions
from ray.runtime_env import RuntimeEnv
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import areal.utils.logging as logging
from areal.api.alloc_mode import AllocationMode, AllocationType
from areal.api.cli_args import (
    ClusterSpecConfig,
    RecoverConfig,
    SchedulingSpec,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.platforms import current_platform, is_npu_available
from areal.utils import name_resolve, names
from areal.utils.exp_metadata import save_experiment_metadata
from areal.utils.launcher import (
    BASE_ENVIRONS,
    JobException,
    JobState,
    validate_config_for_distributed_launcher,
    wait_llm_server_addrs,
)
from areal.utils.offload import get_tms_env_vars
from areal.utils.ray import get_placement_group_master_ip_and_port
from areal.utils.recover import check_if_recover

logger = logging.getLogger("RayLauncher")

RAY_WAIT_CHECK_TIME_INTERVAL = 5  # seconds
DEFAULT_MAIN_FUNC_NAME = "main"
RAY_LAUNCHER = None
RECOVER_TIME_INTERVAL = 10  # seconds

# Node tag constants for multi-node scheduling
# Using unified role names: rollout, trainer (not legacy head/worker)
NODE_TAG_JUDGE = "judge"
NODE_TAG_ROLLOUT = "rollout"
NODE_TAG_TRAINER = "trainer"
NODE_TAG_HEAD = "head"      # Legacy: kept for backward compatibility
NODE_TAG_WORKER = "worker"  # Legacy: kept for backward compatibility


def get_node_allocation_from_config(config):
    """Parse NodeAllocation from config, with backward compatibility.
    
    Tries to import from node_allocation module. If not available,
    falls back to deriving from allocation_mode and environment.
    """
    try:
        import sys
        stage_path = "/mnt/job_stage/app"
        if stage_path not in sys.path:
            sys.path.insert(0, stage_path)
        from node_allocation import parse_node_allocation, NodeAllocation
        return parse_node_allocation(config)
    except ImportError:
        logger.warning("node_allocation module not found, using legacy derivation")
        return _derive_node_allocation_legacy(config)


def _derive_node_allocation_legacy(config):
    """Derive node allocation from legacy config (without nodes section)."""
    n_nodes = config.cluster.n_nodes if hasattr(config, 'cluster') else 2
    use_3node_env = os.environ.get("AREAL_3NODE_MODE") == "1"
    
    if use_3node_env:
        logger.warning("AREAL_3NODE_MODE is deprecated. Use nodes.external_judge.enabled in config.")
    
    class LegacyNodeAllocation:
        def __init__(self, total, roles, external_judge_enabled=False):
            self.total = total
            self.roles = roles
            self.external_judge_enabled = external_judge_enabled
        
        def get_rollout_tag(self, idx=0):
            count = self.roles.get('rollout', 1)
            return f"rollout_{idx}" if count > 1 else "rollout"
        
        def get_trainer_tag(self, idx=0):
            count = self.roles.get('trainer', 1)
            return f"trainer_{idx}" if count > 1 else "trainer"
    
    return LegacyNodeAllocation(
        total=n_nodes,
        roles={"rollout": 1, "trainer": max(1, n_nodes - 1)},
        external_judge_enabled=use_3node_env
    )


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


def should_use_node_tag_scheduling(node_allocation, allocation_mode, cross_nodes: bool) -> bool:
    """Determine if node_tag scheduling should be used.
    
    Returns True if:
    - total nodes >= 2
    - Not cross-node (single-node LLM servers)
    - Not colocated or decoupled_eval mode
    """
    if node_allocation.total < 2:
        return False
    if cross_nodes:
        return False
    if hasattr(allocation_mode, 'type_'):
        from areal.api.alloc_mode import AllocationType
        if allocation_mode.type_ in [AllocationType.COLOCATE, AllocationType.DECOUPLED_EVAL]:
            return False
    return True


def torch_env_hook_for_unified_node_tag(n_tasks: int, n_gpus_per_node: int, node_allocation) -> list[dict]:
    """Generate torch distributed env vars for unified node_tag scheduling.
    
    Discovers master IP from the first trainer node (trainer or trainer_0).
    """
    import socket
    
    logger.info("=" * 60)
    logger.info("torch_env_hook_for_unified_node_tag - Node discovery")
    logger.info(f"Node allocation: {node_allocation.roles}")
    logger.info("=" * 60)
    
    all_nodes = ray.nodes()
    logger.info(f"Total ray nodes: {len(all_nodes)}")
    
    # Find the first trainer node IP
    trainer_ip = None
    trainer_tag = get_trainer_node_tag(node_allocation, 0)
    trainer_resource_key = f"node_tag:{trainer_tag}"
    
    # Also check legacy tags for backward compatibility
    legacy_tags = [f"node_tag:{NODE_TAG_WORKER}", f"node_tag:{NODE_TAG_TRAINER}"]
    
    for idx, node in enumerate(all_nodes):
        node_ip = node.get("NodeManagerAddress", "unknown")
        alive = node.get("Alive", False)
        resources = node.get("Resources", {})
        
        logger.info(f"Node {idx}: IP={node_ip}, Alive={alive}")
        
        if alive:
            # Check for unified trainer tag
            if trainer_resource_key in resources:
                trainer_ip = node_ip
                logger.info(f"  -> Found trainer node ({trainer_tag})")
                break
            # Check legacy tags
            for legacy_tag in legacy_tags:
                if legacy_tag in resources:
                    trainer_ip = node_ip
                    logger.info(f"  -> Found trainer node (legacy: {legacy_tag})")
                    break
            if trainer_ip:
                break
    
    if trainer_ip is None:
        logger.warning(f"No node with {trainer_resource_key} found!")
        trainer_ip = ray.util.get_node_ip_address()
        logger.info(f"Fallback to ray.util.get_node_ip_address() = {trainer_ip}")
    
    # SPCS: Use port in safe range 12031-13000
    port = 12500
    
    logger.info(f"MASTER_ADDR={trainer_ip}, MASTER_PORT={port}")
    
    env_vars = []
    for i in range(n_tasks):
        env_vars.append({
            "RANK": str(i),
            "WORLD_SIZE": str(n_tasks),
            "LOCAL_RANK": "0",
            "MASTER_ADDR": str(trainer_ip),
            "MASTER_PORT": str(port),
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_IB_DISABLE": "1",
        })
        logger.info(f"Trainer {i}: RANK={i}, WORLD_SIZE={n_tasks}")
    
    logger.info("=" * 60)
    return env_vars


def run_func(file_path, function_name, *args, **kwargs):
    import socket
    import os
    
    # DEBUG: Log execution context
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    print(f"DEBUG run_func: Starting on host={hostname}, IP={host_ip}")
    print(f"DEBUG run_func: file_path={file_path}, function_name={function_name}")
    print(f"DEBUG run_func: Environment variables:")
    for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", 
                "CUDA_VISIBLE_DEVICES", "NCCL_DEBUG"]:
        val = os.environ.get(key, "NOT_SET")
        print(f"DEBUG run_func:   {key}={val}")
    
    # ============================================================================
    # CRITICAL: Apply Ray Weight Sync patch for cross-node LoRA weight updates
    # Uses Ray object store instead of shared filesystem (which SPCS doesn't have)
    # ============================================================================
    try:
        print("DEBUG run_func: Applying patches for SPCS...")
        
        # Import the patch from stage
        import sys
        # Stage files are mounted at /mnt/job_stage/app/ (not /mnt/job_stage/src/)
        stage_path = "/mnt/job_stage/app/areal_patches"
        stage_parent = "/mnt/job_stage/app"
        if stage_path not in sys.path:
            sys.path.insert(0, stage_path)
            sys.path.insert(0, stage_parent)
        
        from ray_weight_sync import apply_ray_weight_sync_patch
        if apply_ray_weight_sync_patch():
            print("DEBUG run_func: Ray Weight Sync patch applied successfully!")
        else:
            print("DEBUG run_func: WARNING - Ray Weight Sync patch returned False")
    except Exception as e:
        print(f"DEBUG run_func: ERROR applying Ray Weight Sync patch: {e}")
        import traceback
        traceback.print_exc()
    # ============================================================================
    
    # Convert the file path to a module name
    module_name = file_path.replace("/", "_").replace(".", "_")

    # Load the module from file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise FileNotFoundError(
            f"Cannot load module from file path '{file_path}'. "
            f"Please ensure the file exists and the path is correct."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Get the function and execute it
    try:
        function = getattr(module, function_name)
    except AttributeError as e:
        raise ValueError(
            f"Function '{function_name}' not found in module '{module_name}'. "
            f"Please ensure the name of the main function in your entry point "
            f"is '{function_name}'."
        ) from e
    return function(*args, **kwargs)


class RayLauncher:
    def __init__(self, experiment_name: str, trial_name: str, fileroot: str):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot

        # job_name to ray future
        self.jobs = {}
        self.placement_groups = {}

    @property
    def run_name(self):
        return f"{self.experiment_name}_{self.trial_name}"

    def submit(
        self,
        job_name: str,
        file_path: str,
        func_name: str,
        args: list[str],  # arguments to pass to the function
        gpus: int,
        cpus: int,
        mem: int,  # MB
        env_vars: dict | None = None,
        placement_group: PlacementGroup | None = None,
        bundle_index: int = -1,
        kwargs: (
            dict[str, str] | None
        ) = None,  # keyword arguments to pass to the function
    ):
        if kwargs is None:
            kwargs = {}
        runtime_env = RuntimeEnv(
            env_vars=env_vars or dict(),
        )
        scheduling_strategy = (
            PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_index,
                placement_group_capture_child_tasks=True,
            )
            if placement_group is not None
            else "DEFAULT"
        )
        if is_npu_available:
            future = ray.remote(
                num_cpus=cpus,
                resources={"NPU": gpus},
                memory=mem * 1024 * 1024,  # Convert MB to bytes
                runtime_env=runtime_env,
                scheduling_strategy=scheduling_strategy,
            )(run_func).remote(file_path, func_name, *args, **kwargs)
            self.jobs[job_name] = future
        else:
            future = ray.remote(
                num_cpus=cpus,
                num_gpus=gpus,
                memory=mem * 1024 * 1024,  # Convert MB to bytes
                runtime_env=runtime_env,
                scheduling_strategy=scheduling_strategy,
            )(run_func).remote(file_path, func_name, *args, **kwargs)
            self.jobs[job_name] = future
        return future

    def submit_with_node_tag(
        self,
        job_name: str,
        file_path: str,
        func_name: str,
        args: list[str],
        gpus: int,
        cpus: int,
        mem: int,  # MB
        node_tag: str,  # "judge", "rollout", "trainer", or legacy "head"/"worker"
        env_vars: dict | None = None,
        kwargs: dict[str, str] | None = None,
        node_tag_amount: float = 0.01,  # Use small fraction to allow multiple tasks on same node
    ):
        """Submit a job to a specific node using node_tag resource.
        
        Uses a small fractional resource amount (default 0.01) so multiple tasks
        can be scheduled on the same node that has node_tag:X = 1.0 resource.
        """
        if kwargs is None:
            kwargs = {}
        runtime_env = RuntimeEnv(
            env_vars=env_vars or dict(),
        )
        resource_key = f"node_tag:{node_tag}"
        logger.info(f"Submitting job {job_name} to node with {resource_key} (amount={node_tag_amount})")
        
        if is_npu_available:
            future = ray.remote(
                num_cpus=cpus,
                resources={"NPU": gpus, resource_key: node_tag_amount},
                memory=mem * 1024 * 1024,
                runtime_env=runtime_env,
            )(run_func).remote(file_path, func_name, *args, **kwargs)
        else:
            future = ray.remote(
                num_cpus=cpus,
                num_gpus=gpus,
                resources={resource_key: node_tag_amount},
                memory=mem * 1024 * 1024,
                runtime_env=runtime_env,
            )(run_func).remote(file_path, func_name, *args, **kwargs)
        self.jobs[job_name] = future
        return future

    def submit_array_with_node_tag(
        self,
        job_name: str,
        file_path: str,
        func_name: str,
        count: int,
        nodes: int,
        list_args: list[list],
        gpus_per_task: int,
        cpus_per_task: int,
        mem_per_task: int,
        node_tag: str,  # "judge", "rollout", "trainer", or legacy "head"/"worker"
        list_kwargs: list[dict] | None = None,
        env_vars: dict | None = None,
        env_hook_simple: Callable[[int], list[dict]] | None = None,
    ):
        """Submit array of jobs to specific node using node_tag resource.
        
        Unlike submit_array, this uses node_tag resources for deterministic
        node placement instead of placement groups.
        """
        if count % nodes != 0:
            raise ValueError(
                f"Count {count} is not divisible by nodes {nodes}."
            )
        if len(list_args) != count:
            raise ValueError(
                f"Length of list_args {len(list_args)} does not matc1h count {count}."
            )
        if list_kwargs is not None and len(list_kwargs) != count:
            raise ValueError(
                f"Length of list_kwargs {len(list_kwargs)} does not match count {count}."
            )

        logger.info(f"Submitting {count} tasks to node_tag:{node_tag}")
        
        if env_hook_simple:
            extra_env_vars = env_hook_simple(count)
        else:
            extra_env_vars = None

        futures = []
        for i in range(count):
            args = list_args[i]
            kwargs = list_kwargs[i] if list_kwargs is not None else {}
            _env_vars = (env_vars or {}).copy()
            
            if extra_env_vars:
                _env_vars |= extra_env_vars[i]

            future = self.submit_with_node_tag(
                job_name=f"{job_name}:{i}",
                file_path=file_path,
                func_name=func_name,
                args=args,
                gpus=gpus_per_task,
                cpus=cpus_per_task,
                mem=mem_per_task,
                node_tag=node_tag,
                env_vars=_env_vars,
                kwargs=kwargs,
            )
            futures.append(future)

        return futures

    def submit_array(
        self,
        job_name: str,
        file_path: str,
        func_name: str,
        count: int,
        nodes: int,
        list_args: list[list],
        gpus_per_task: int,
        cpus_per_task: int,
        mem_per_task: int,  # MB
        list_kwargs: list[dict] | None = None,
        env_vars: dict | None = None,
        env_hook: Callable[[PlacementGroup], list[dict]] | None = None,
    ):
        """Submit an array of jobs to Ray with ray placement groups.

        Note: Here we use `ray.remote` instead of `ray job submit` since `ray job submit`
        does not support placement groups, and can not specify which node to run the job on.
        Therefore we could not know the IP address of jobs for torch distributed initialization.
        """

        if count % nodes != 0:
            raise ValueError(
                f"Count {count} is not divisible by nodes {nodes}. "
                "Please ensure that count is a multiple of nodes."
            )
        if len(list_args) != count:
            raise ValueError(
                f"Length of list_args {len(list_args)} does not match count {count}."
            )
        if list_kwargs is not None:
            if len(list_kwargs) != count:
                raise ValueError(
                    f"Length of list_kwargs {len(list_kwargs)} does not match count {count}."
                )

        tasks_per_node = count // nodes
        gpus_per_node = gpus_per_task * tasks_per_node
        cpus_per_node = cpus_per_task * tasks_per_node
        mem_per_node = mem_per_task * tasks_per_node

        if job_name not in self.placement_groups:
            if is_npu_available:
                device_bundles = [
                    {
                        "CPU": cpus_per_node,
                        "NPU": gpus_per_node,
                        "memory": mem_per_node * 1024 * 1024,  # Convert MB to bytes
                    }
                ] * nodes
            else:
                device_bundles = [
                    {
                        "CPU": cpus_per_node,
                        "GPU": gpus_per_node,
                        "memory": mem_per_node * 1024 * 1024,  # Convert MB to bytes
                    }
                ] * nodes
            placement_group = ray.util.placement_group(
                bundles=device_bundles, strategy="PACK"
            )
            try:
                ray.get(placement_group.ready(), timeout=30)
            except ray.exceptions.GetTimeoutError as e:
                logger.error(
                    "Ray placement group timeout, please check if the resource requirement "
                    "for your experiment exceeds the available resources in the cluster. \n"
                    f"ray.nodes(): {ray.nodes()} \n"
                    f"Placement Group bundles: "
                    f"cpus_per_node={cpus_per_node}, gpus_per_node={gpus_per_node}, "
                    f"mem_per_node={mem_per_node}MB, nodes={nodes}"
                )
                raise e
            self.placement_groups[job_name] = placement_group
        else:
            # Reuse placement group in recover runs
            placement_group = self.placement_groups[job_name]

        if env_hook:
            extra_env_vars = env_hook(placement_group)

        futures = []
        for i in range(count):
            args = list_args[i]
            kwargs = list_kwargs[i] if list_kwargs is not None else {}

            # manage environment variables
            env_vars = env_vars or {}
            if current_platform.device_control_env_var in env_vars:
                logger.warning(
                    f"Setting {current_platform.device_control_env_var} before running ray jobs may result in unexpected behavior."
                )

            node_id = i // tasks_per_node

            if env_hook:
                _env_vars = env_vars.copy()
                _env_vars |= extra_env_vars[i]
            else:
                _env_vars = env_vars

            future = self.submit(
                job_name=f"{job_name}:{i}",
                file_path=file_path,
                func_name=func_name,
                args=args,
                gpus=gpus_per_task,
                cpus=cpus_per_task,
                mem=mem_per_task,
                env_vars=_env_vars,
                placement_group=placement_group,
                bundle_index=node_id,
                kwargs=kwargs,
            )
            futures.append(future)

        return futures

    def stop(self, job_name: str, force: bool = False):
        """Stop a job by name."""
        if job_name in self.jobs:
            future = self.jobs[job_name]
            try:
                ray.cancel(future, force=force)
            except Exception as e:
                logger.error(f"Failed to cancel job {job_name}: {e}")
                return
            self.jobs.pop(job_name, None)
            logger.info(f"Job {job_name} stopped.")
        else:
            logger.warning(f"Job {job_name} not found in running jobs.")

    def stop_all(self, force: bool = False, pattern: str | None = None):
        """Stop all jobs with pattern matched."""
        job_names = list(self.jobs.keys())
        if pattern:
            job_names = [
                job_name for job_name in job_names if re.search(pattern, job_name)
            ]
        for job_name in job_names:
            self.stop(job_name, force=force)
        if pattern:
            logger.info(f'Jobs matching the pattern "{pattern}" stopped')
        else:
            logger.info("All jobs stopped.")
        cur_job_names = self.jobs.keys()
        for job_name in job_names:
            if job_name in cur_job_names:
                self.jobs.pop(job_name)

    def wait(
        self, check_status=(JobState.FAILED,), remove_status=(JobState.COMPLETED,)
    ):
        """Check every RAY_WAIT_CHECK_TIME_INTERVAL seconds for the status of all jobs.
        If a ray job returns, its status changes to JobState.COMPLETED.
        If a ray job failed, its status changes to JobState.FAILED.
        If any job is in check_status, stop all jobs at once.
        If any job is in remove status, remove them from job list.
        Return if all jobs are removed from job list, or some job is in check status.
        """
        for status in list(check_status) + list(remove_status):
            assert status in [
                JobState.COMPLETED,
                JobState.FAILED,
            ], "In RayLauncher.wait, we only check completed or failed jobs."
        logger.info(f"Waiting for {len(self.jobs)} jobs.")
        while self.jobs:
            job_status = {}
            for job_name, future in list(self.jobs.items()):
                try:
                    r = ray.get(future, timeout=0.1)
                    logger.info(f"Job {job_name} completed with result: {r}")
                    job_status[job_name] = JobState.COMPLETED
                except ray.exceptions.RayTaskError as e:
                    logger.error(f"Job {job_name} failed with error: {e}.")
                    job_status[job_name] = JobState.FAILED
                except ray.exceptions.GetTimeoutError:
                    continue

            for job_name, status in job_status.items():
                if status in check_status:
                    logger.info(f"Job {job_name} is {status}, stopping all jobs.")
                    # raise exception to enter recover.
                    # should not changed to stop_all
                    raise JobException(
                        run_name=self.run_name,
                        worker_type=job_name.split(":")[0],
                        host="ray",
                        reason=status,
                    )
                if status in remove_status:
                    logger.info(f"Job {job_name} is {status}, removed.")
                    self.jobs.pop(job_name)

            time.sleep(RAY_WAIT_CHECK_TIME_INTERVAL)


def main():
    # Connect to existing Ray cluster if RAY_ADDRESS is set (SPCS multi-node)
    # This preserves node_tag resources registered by run_areal.py
    ray_address = os.environ.get("RAY_ADDRESS")
    if ray_address:
        logger.info(f"Connecting to existing Ray cluster at {ray_address}")
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        ray.init()
    config, _ = parse_cli_args(sys.argv[1:])
    ray_main(config, run_id=0)


def ray_main(config, run_id: int = 0):
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    is_recover_run = check_if_recover(config.recover, run_id)
    validate_config_for_distributed_launcher(config)

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(
            experiment_name=config.experiment_name, trial_name=config.trial_name
        )
    )

    n_nodes = config.cluster.n_nodes
    n_gpus_per_node = config.cluster.n_gpus_per_node

    # To reuse ray placement groups in recover runs.
    global RAY_LAUNCHER
    if RAY_LAUNCHER is None:
        assert run_id == 0
        launcher = RayLauncher(
            experiment_name=config.experiment_name,
            trial_name=config.trial_name,
            fileroot=config.cluster.fileroot,
        )
        RAY_LAUNCHER = launcher
    else:
        launcher = RAY_LAUNCHER

    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)

    try:
        actor_spec = to_structured_cfg(config.actor.scheduling_spec[0], SchedulingSpec)
    except AttributeError:
        actor_spec = SchedulingSpec()

    if allocation_mode.gen_backend in ("sglang", "vllm"):
        try:
            rollout_spec = to_structured_cfg(
                config.rollout.scheduling_spec[0], SchedulingSpec
            )
        except AttributeError:
            rollout_spec = SchedulingSpec()

    if not is_recover_run:
        metadata_file = save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )
        logger.info(f"Saved experiment metadata to {metadata_file}")

    sglang_addrs = []
    n_sglang_nodes = 0
    vllm_addrs = []
    n_vllm_nodes = 0
    if allocation_mode.gen_backend == "sglang":
        # Launcher should launch SGLang servers according to allocation mode.
        config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
        n_sglang_servers = allocation_mode.gen.dp_size
        n_sglang_nodes = max(1, allocation_mode.gen.world_size // n_gpus_per_node)
        node_group_size = max(1, allocation_mode.gen_instance_size // n_gpus_per_node)
        n_servers_per_node = max(n_sglang_servers // n_sglang_nodes, 1)
        cross_nodes = allocation_mode.gen_instance_size > n_gpus_per_node

        base_seed = config.sglang.random_seed
        sglang_args_list = [
            [
                sys.argv[1:]
                + [f"sglang.random_seed={base_seed + i * n_servers_per_node}"]
            ]
            for i in range(n_sglang_nodes)
        ]
        sglang_entry_point = str(
            pathlib.Path(__file__).resolve().parent.joinpath("sglang_server.py")
        )

        def sglang_env_hook(
            n_tasks: int, task_group_size: int, placement_group: PlacementGroup
        ) -> list[dict]:
            master_addrs = []
            master_ports = []
            for i in range(0, n_tasks, task_group_size):
                host_ip, port = get_placement_group_master_ip_and_port(
                    placement_group, i
                )
                master_addrs.append(host_ip)
                master_ports.append(port)

            env_vars = []
            for i in range(n_tasks):
                env_vars.append(
                    dict(
                        AREAL_SGLANG_MULTI_NODE_RANK=str(i % task_group_size),
                        AREAL_SGLANG_MULTI_NODE_MASTER_ADDR=master_addrs[
                            i // task_group_size
                        ],
                        AREAL_SGLANG_MULTI_NODE_MASTER_PORT=str(
                            master_ports[i // task_group_size]
                        ),
                    )
                )

            return env_vars

        # launch a task to start all sglang servers in one node
        # ============================================================================
        # UNIFIED NODE SCHEDULING: Replace hardcoded 2/3-node conditionals
        # ============================================================================
        node_allocation = get_node_allocation_from_config(config)
        use_node_tags = should_use_node_tag_scheduling(node_allocation, allocation_mode, cross_nodes)
        
        if use_node_tags:
            # Schedule SGLang to rollout node(s) using unified node tags
            rollout_count = node_allocation.roles.get('rollout', 1)
            logger.info(f"Using unified node_tag scheduling: {rollout_count} rollout node(s)")
            
            for server_idx in range(n_sglang_nodes):
                # Determine which rollout node this server goes to
                node_idx = server_idx % rollout_count
                node_tag = get_rollout_node_tag(node_allocation, node_idx)
                
                logger.info(f"Scheduling SGLang server {server_idx} to node_tag:{node_tag}")
                launcher.submit_with_node_tag(
                    job_name=f"llm_server:{server_idx}",
                    file_path=sglang_entry_point,
                    func_name=DEFAULT_MAIN_FUNC_NAME,
                    args=sglang_args_list[server_idx],
                    gpus=n_gpus_per_node,
                    cpus=rollout_spec.cpu * n_gpus_per_node,
                    mem=rollout_spec.mem * 1024 * n_gpus_per_node,
                    node_tag=node_tag,
                    env_vars={**BASE_ENVIRONS, **rollout_spec.env_vars},
                )
        else:
            # Fallback: use placement groups for cross-node or other modes
            logger.info("Using placement group scheduling (cross-node or special mode)")
            launcher.submit_array(
                job_name="llm_server",
                file_path=sglang_entry_point,
                func_name=DEFAULT_MAIN_FUNC_NAME,
                count=n_sglang_nodes,
                nodes=n_sglang_nodes,
                list_args=sglang_args_list,
                gpus_per_task=n_gpus_per_node,
                cpus_per_task=rollout_spec.cpu * n_gpus_per_node,
                mem_per_task=rollout_spec.mem * 1024 * n_gpus_per_node,
                env_vars={**BASE_ENVIRONS, **rollout_spec.env_vars},
                env_hook=(
                    partial(sglang_env_hook, n_sglang_nodes, node_group_size)
                    if cross_nodes
                    else None
                ),
            )
        # Get SGLang server addresses via name_resolve
        try:
            sglang_addrs = wait_llm_server_addrs(
                config.experiment_name,
                config.trial_name,
                n_sglang_servers,
            )
        except (TimeoutError, KeyboardInterrupt) as e:
            launcher.stop_all(
                force=False
            )  # force=False will send KeyboardInterrupt to sglang_server.main() to further clean all sglang-related processes
            raise e
        
        # ========== HEALTH CHECK POLLING: Wait for all servers to be truly ready ==========
        import requests
        import socket
        
        logger.info("=" * 70)
        logger.info("Waiting for SGLang servers to be fully ready (HTTP 200)...")
        logger.info(f"SGLang server addresses: {sglang_addrs}")
        logger.info("=" * 70)
        
        max_wait_time = 300  # 5 minutes max wait
        poll_interval = 5   # Check every 5 seconds
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.error(f"SGLang servers not ready after {max_wait_time}s, giving up")
                launcher.stop_all(force=False)
                raise TimeoutError(f"SGLang servers failed to become ready in {max_wait_time}s")
            
            ready_count = 0
            not_ready = []
            
            for addr in sglang_addrs:
                try:
                    url = f"http://{addr}/health"
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        ready_count += 1
                    else:
                        not_ready.append(f"{addr}(HTTP {resp.status_code})")
                except requests.exceptions.ConnectionError:
                    not_ready.append(f"{addr}(ConnErr)")
                except requests.exceptions.Timeout:
                    not_ready.append(f"{addr}(Timeout)")
                except Exception as e:
                    not_ready.append(f"{addr}({type(e).__name__})")
            
            logger.info(f"[{elapsed:.0f}s] SGLang health: {ready_count}/{len(sglang_addrs)} ready. Not ready: {not_ready}")
            
            if ready_count == len(sglang_addrs):
                logger.info(f"All {len(sglang_addrs)} SGLang servers ready after {elapsed:.1f}s")
                break
            
            time.sleep(poll_interval)
        
        # Test cross-node connectivity from TRAINER node before proceeding
        # ============================================================================
        # UNIFIED: Use trainer node tag from node_allocation
        # ============================================================================
        test_node_tag = None
        if use_node_tags:
            test_node_tag = get_trainer_node_tag(node_allocation, 0)
        
        if test_node_tag:
            logger.info("Testing cross-node HTTP connectivity from worker node...")
            
            # Dynamically create remote function with correct node_tag
            resource_key = f"node_tag:{test_node_tag}"
            
            @ray.remote(num_cpus=1)
            def test_sglang_from_node(addrs, max_wait=120, poll_interval=5):
                """Wait until all SGLang servers are reachable from worker node."""
                import requests
                import socket
                import time as worker_time
                
                local_hostname = socket.gethostname()
                local_ip = socket.gethostbyname(local_hostname)
                print(f"Test node: hostname={local_hostname}, ip={local_ip}")
                
                start = worker_time.time()
                while True:
                    elapsed = worker_time.time() - start
                    if elapsed > max_wait:
                        return f"TIMEOUT after {max_wait}s"
                    
                    ready = 0
                    not_ready = []
                    for addr in addrs:
                        try:
                            resp = requests.get(f"http://{addr}/health", timeout=10)
                            if resp.status_code == 200:
                                ready += 1
                            else:
                                not_ready.append(f"{addr}(HTTP {resp.status_code})")
                        except requests.exceptions.ConnectionError as e:
                            err = str(e)
                            if "Connection refused" in err:
                                not_ready.append(f"{addr}(ConnRefused)")
                            elif "timed out" in err.lower():
                                not_ready.append(f"{addr}(Timeout)")
                            else:
                                not_ready.append(f"{addr}(ConnErr)")
                        except requests.exceptions.Timeout:
                            not_ready.append(f"{addr}(Timeout)")
                        except Exception as e:
                            not_ready.append(f"{addr}({type(e).__name__})")
                    
                    print(f"[{elapsed:.0f}s] TestNode→SGLang: {ready}/{len(addrs)} ready. Not ready: {not_ready}")
                    
                    if ready == len(addrs):
                        return f"SUCCESS: All {len(addrs)} servers reachable after {elapsed:.1f}s"
                    
                    worker_time.sleep(poll_interval)
            
            try:
                logger.info(f"Waiting for {test_node_tag}→SGLang connectivity (timeout=120s)...")
                # Apply node_tag resource dynamically
                test_func_with_resources = test_sglang_from_node.options(resources={resource_key: 0.01})
                test_future = test_func_with_resources.remote(sglang_addrs)
                result = ray.get(test_future, timeout=130)
                logger.info(f"{test_node_tag} connectivity result: {result}")
                if result.startswith("TIMEOUT"):
                    logger.error(f"Cross-node HTTP failed from {test_node_tag}! Stopping job.")
                    launcher.stop_all(force=False)
                    raise TimeoutError(f"SGLang servers not reachable from {test_node_tag} node")
            except ray.exceptions.GetTimeoutError:
                logger.error(f"{test_node_tag} connectivity test timed out")
                launcher.stop_all(force=False)
                raise TimeoutError(f"{test_node_tag} connectivity test timed out")
            except Exception as e:
                logger.error(f"{test_node_tag} connectivity test failed: {type(e).__name__}: {e}")
                launcher.stop_all(force=False)
                raise
        
        logger.info("=" * 70)
        logger.info("SGLang servers all ready from both nodes, proceeding to launch trainers")
        logger.info("=" * 70)
    elif allocation_mode.gen_backend == "vllm":
        config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
        # Launcher should launch vLLM servers according to allocation mode.
        vllm_tp_size = allocation_mode.gen.tp_size
        n_vllm_servers = allocation_mode.gen.dp_size
        n_vllm_nodes = allocation_mode.gen.world_size // n_gpus_per_node

        base_seed = config.vllm.seed
        vllm_args_list = [
            [sys.argv[1:] + [f"vllm.seed={base_seed + i}"]]
            for i in range(n_vllm_servers)
        ]
        vllm_entry_point = str(
            pathlib.Path(__file__).resolve().parent.joinpath("vllm_server.py")
        )
        launcher.submit_array(
            job_name="llm_server",
            file_path=vllm_entry_point,
            func_name=DEFAULT_MAIN_FUNC_NAME,
            count=n_vllm_servers,
            nodes=n_vllm_nodes,
            list_args=vllm_args_list,
            gpus_per_task=vllm_tp_size,
            cpus_per_task=rollout_spec.cpu * vllm_tp_size,
            mem_per_task=rollout_spec.mem * 1024 * vllm_tp_size,
            env_vars={**BASE_ENVIRONS, **rollout_spec.env_vars},
        )
        # Get vllm server addresses via name_resolve
        try:
            vllm_addrs = wait_llm_server_addrs(
                config.experiment_name,
                config.trial_name,
                n_vllm_servers,
            )
        except (TimeoutError, KeyboardInterrupt) as e:
            launcher.stop_all(force=True)
            raise e

    if config.get("enable_offload", False):
        tms_env_vars = get_tms_env_vars()
    else:
        tms_env_vars = {}

    if allocation_mode.type_ == AllocationType.DECOUPLED_EVAL:
        trainer_n_nodes = 1
        gpus_per_task = 0
    elif allocation_mode.type_ == AllocationType.COLOCATE:
        # Colocated mode: trainer shares nodes with LLM servers
        trainer_n_nodes = n_nodes
        gpus_per_task = 1
        logger.info(f"Colocated mode: trainer will share {trainer_n_nodes} nodes with LLM servers")
    else:
        # Decoupled mode: trainer gets dedicated nodes after LLM servers
        n_llm_nodes = n_sglang_nodes if allocation_mode.gen_backend == "sglang" else n_vllm_nodes
        trainer_n_nodes = n_nodes - n_llm_nodes
        gpus_per_task = 1
        logger.info(f"Decoupled mode: {n_llm_nodes} nodes for LLM, {trainer_n_nodes} nodes for trainer")
    trainer_entry_point = sys.argv[1]
    n_trainer_processes = trainer_n_nodes * config.cluster.n_gpus_per_node
    trainer_args_list = [[sys.argv[2:]] for _ in range(n_trainer_processes)]
    if allocation_mode.type_ != AllocationType.LLM_SERVER_ONLY:
        llm_addrs = (
            sglang_addrs if allocation_mode.gen_backend == "sglang" else vllm_addrs
        )

        # In ray, we launch trainer in the granularity of processes (1 GPU per process)
        # We amend environment variable similar to torchrun to ensure correct initialization of
        # torch distributed.
        def torch_env_hook(n_tasks: int, placement_group: PlacementGroup) -> list[dict]:
            host_ip, port = get_placement_group_master_ip_and_port(placement_group)
            logger.info(
                f"Amend torch distributed env vars: MASTER_ADDR={host_ip}, PORT={port}"
            )
            env_vars = []
            for i in range(n_tasks):
                # NOTE: Here we only provide environment variables for torch distributed
                # initialization, and LOCAL_RANK for torch.device.
                # Other environment variables automatically set by torchrun are not set, and
                # they should be never accessed in trainer code.
                env_vars.append(
                    {
                        "RANK": str(i),
                        "WORLD_SIZE": str(n_tasks),
                        # Ray will automatically isolate CUDA_VISIBLE_DEVICES for each GPU
                        "LOCAL_RANK": "0",
                        "MASTER_ADDR": str(host_ip),
                        "MASTER_PORT": str(port),
                    }
                )
            return env_vars

        _env_vars = dict(
            AREAL_LLM_SERVER_ADDRS=",".join(llm_addrs),
            AREAL_RECOVER_RUN=str(int(is_recover_run)),
            # SPCS: Force eth0 (pod network) for cross-node NCCL
            NCCL_SOCKET_IFNAME="eth0",
            NCCL_IB_DISABLE="1",
        )
        if allocation_mode.gen_backend == "sglang":
            # Required by NCCL weight update group.
            _env_vars["NCCL_CUMEM_ENABLE"] = "0"
            _env_vars["NCCL_NVLS_ENABLE"] = "0"

        # ============================================================================
        # UNIFIED TRAINER SCHEDULING: Replace hardcoded 2/3-node conditionals
        # Ensure node_allocation is available (may have been parsed in SGLang section)
        # ============================================================================
        try:
            node_allocation
        except NameError:
            node_allocation = get_node_allocation_from_config(config)
        
        use_trainer_node_tags = should_use_node_tag_scheduling(
            node_allocation, allocation_mode, cross_nodes=False
        )
        
        if use_trainer_node_tags:
            # Use unified node_tag scheduling for trainers
            trainer_count = node_allocation.roles.get('trainer', 1)
            logger.info(f"Using unified node_tag scheduling: {trainer_count} trainer node(s)")
            logger.info(f"trainer_n_nodes={trainer_n_nodes}, n_gpus_per_node={n_gpus_per_node}")
            logger.info(f"Total trainer processes: {n_trainer_processes}")
            
            # Generate env vars using unified helper
            trainer_env_vars = torch_env_hook_for_unified_node_tag(
                n_trainer_processes, n_gpus_per_node, node_allocation
            )
            
            # Submit each trainer process to appropriate trainer node
            for i, args in enumerate(trainer_args_list):
                # Determine which trainer node this process goes to
                node_idx = i // n_gpus_per_node  # Which trainer node
                trainer_node_tag = get_trainer_node_tag(node_allocation, node_idx % trainer_count)
                
                full_env = {
                    **BASE_ENVIRONS,
                    **actor_spec.env_vars,
                    **_env_vars,
                    **tms_env_vars,
                    **trainer_env_vars[i],
                    "AREAL_SPMD_MODE": "1",
                }
                
                logger.info(f"Scheduling trainer:{i} to node_tag:{trainer_node_tag}")
                launcher.submit_with_node_tag(
                    job_name=f"trainer:{i}",
                    file_path=trainer_entry_point,
                    func_name=DEFAULT_MAIN_FUNC_NAME,
                    args=args,
                    gpus=gpus_per_task,
                    cpus=actor_spec.cpu,
                    mem=actor_spec.mem * 1024,
                    node_tag=trainer_node_tag,
                    env_vars=full_env,
                )
        else:
            # Fallback: use placement groups
            logger.info("Using placement group scheduling for trainers")
            launcher.submit_array(
                job_name="trainer",
                file_path=trainer_entry_point,
                func_name=DEFAULT_MAIN_FUNC_NAME,
                count=n_trainer_processes,
                nodes=trainer_n_nodes,
                list_args=trainer_args_list,
                gpus_per_task=gpus_per_task,
                cpus_per_task=actor_spec.cpu,
                mem_per_task=actor_spec.mem * 1024,
                env_vars={
                    **BASE_ENVIRONS,
                    **actor_spec.env_vars,
                    **_env_vars,
                    **tms_env_vars,
                    "AREAL_SPMD_MODE": "1",
                },
                env_hook=partial(torch_env_hook, n_trainer_processes),
            )

    try:
        launcher.wait(check_status=(JobState.COMPLETED, JobState.FAILED))
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        # The 'force' is passed to ray.cancel(future, force=force).
        # If force=False, a KeyboardInterrupt will be raised in sglang_server.main(),
        # allowing for a more thorough cleanup of all sglang-related processes.
        # This is particularly important when using sglang's dp_attention,
        # as it will leave residual processes that occupy GPU memory.
        launcher.stop_all(force=False, pattern="llm_server")
        # If force=True, the task is immediately killed, triggering the trainer to end the job.
        # Note: For trainer processes, we use force=True because the trainer doesn't
        # handle KeyboardInterrupt properly when force=False.
        launcher.stop_all(force=True, pattern="trainer")
        recover_states = [JobState.FAILED]
        if isinstance(e, JobException):
            recover_this = (
                e.reason in recover_states
                and run_id < config.recover.retries
                and config.recover.mode in ["auto", "fault"]
            )
            if recover_this:
                time.sleep(RECOVER_TIME_INTERVAL)
                ray_main(config, run_id=run_id + 1)
            else:
                raise e
        else:
            raise e


if __name__ == "__main__":
    # usage: python -m areal.launcher.ray \
    #   <entry_point> --config <config_path> [<additional_args>] \
    #   launcher.ray.main_func_name=<main_func_name_in_entry_point>
    main()
