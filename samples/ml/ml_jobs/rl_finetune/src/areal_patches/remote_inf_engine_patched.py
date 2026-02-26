"""
Patched RemoteInfEngine for SPCS.

CRITICAL FIX: When experiment_name/trial_name is configured (for external judge),
wait up to 5 minutes for name_resolve instead of 1 second, and FAIL with error
if timeout instead of silently falling back to rollout servers.

Original behavior (BROKEN):
  - timeout=1 second for name_resolve
  - Silent fallback to AREAL_LLM_SERVER_ADDRS (rollout servers)
  - Result: Judge uses policy model to judge itself!

Patched behavior (CORRECT):
  - timeout=300 seconds (5 minutes) for name_resolve  
  - Explicit error on timeout when experiment_name is configured
  - Result: Training fails fast if judge not available
"""
import os

# Try to import from areal.infra (newer versions) or areal (older versions)
try:
    from areal.infra.utils.launcher import wait_llm_server_addrs
except ImportError:
    from areal.utils.launcher import wait_llm_server_addrs

from areal.utils import logging

logger = logging.getLogger("RemoteInfEnginePatched")

# SPCS FIX: Increase timeout from 1 second to 5 minutes for external judge resolution
JUDGE_RESOLVE_TIMEOUT = 300  # 5 minutes


def patched_initialize(self, engine_id=None, addr=None, train_data_parallel_size=None):
    """
    Patched initialize method with proper judge server resolution.
    
    Key changes from original:
    1. Wait 5 minutes (not 1 second) for name_resolve
    2. FAIL with error if timeout when experiment_name is configured
    3. Only fallback to AREAL_LLM_SERVER_ADDRS if no experiment_name configured
    """
    import uuid
    import random
    import torch.distributed as dist
    
    if engine_id is None:
        if dist.is_initialized():
            engine_id = str(dist.get_rank())
        else:
            engine_id = uuid.uuid4().hex
    self.engine_id = engine_id
    self.logger = logging.getLogger(f"[RemoteInfEngine Rank {engine_id}]")

    # Priority 1: Explicit addr parameter (highest priority)
    if addr:
        self.addresses = addr if isinstance(addr, list) else [addr]
        self.logger.info("Get server addresses from the `addr` argument.")
    
    # Priority 2: Local subprocess servers
    elif len(self.local_server_processes) > 0:
        self.addresses = [f"{s.host}:{s.port}" for s in self.local_server_processes]
        self.logger.info("Get server addresses from the local subprocess.")
    
    # Priority 3: name_resolve via experiment_name/trial_name (for external judge)
    elif self.config.experiment_name is not None and self.config.trial_name is not None:
        self.logger.info(
            f"SPCS PATCH: Resolving external server via name_resolve "
            f"(experiment={self.config.experiment_name}, trial={self.config.trial_name})"
        )
        self.logger.info(f"SPCS PATCH: Waiting up to {JUDGE_RESOLVE_TIMEOUT}s for server...")
        
        try:
            self.addresses = wait_llm_server_addrs(
                experiment_name=self.config.experiment_name,
                trial_name=self.config.trial_name,
                timeout=JUDGE_RESOLVE_TIMEOUT,  # 5 minutes instead of 1 second!
            )
            self.logger.info(
                f"SPCS PATCH: Successfully resolved addresses from name_resolve: {self.addresses}"
            )
        except (TimeoutError, RuntimeError) as e:
            # CRITICAL: Do NOT silently fallback to rollout servers!
            # If experiment_name is configured, user explicitly wants an external server.
            self.logger.error(
                f"SPCS PATCH: FAILED to resolve server via name_resolve after {JUDGE_RESOLVE_TIMEOUT}s!\n"
                f"  experiment_name={self.config.experiment_name}\n"
                f"  trial_name={self.config.trial_name}\n"
                f"  error={e}\n"
                f"\n"
                f"This likely means:\n"
                f"  1. The external server (e.g., judge) has not started yet\n"
                f"  2. The name_resolve NFS path is not shared between jobs\n"
                f"  3. The experiment_name/trial_name don't match the server config\n"
                f"\n"
                f"NOT falling back to AREAL_LLM_SERVER_ADDRS (rollout servers) because\n"
                f"experiment_name was explicitly configured for external server resolution."
            )
            raise RuntimeError(
                f"External server resolution failed for experiment={self.config.experiment_name}, "
                f"trial={self.config.trial_name} after {JUDGE_RESOLVE_TIMEOUT}s. "
                f"Check that the external server is running and name_resolve paths match."
            ) from e
    
    # Priority 4: Environment variable (only if no experiment_name configured)
    elif os.getenv("AREAL_LLM_SERVER_ADDRS"):
        self.addresses = os.environ["AREAL_LLM_SERVER_ADDRS"].split(",")
        self.logger.info("Get server addresses from environment variable.")
    
    else:
        self.addresses = []

    if not self.addresses:
        raise RuntimeError(
            "No configured inference servers. "
            "Please pass in server addresses by arguments "
            "for `initialize` or environment "
            "variable `AREAL_LLM_SERVER_ADDRS`."
        )

    self.logger.info("Waiting for server ready...")
    for addr_ in self.addresses:
        self._wait_for_server(addr_)
    self.server_idx = random.randint(0, len(self.addresses) - 1)
    self.logger.info("Servers are all ready!")

    # Import here to avoid circular imports
    try:
        from areal.infra.workflow_executor import WorkflowExecutor
    except ImportError:
        from areal.core.workflow_executor import WorkflowExecutor
    
    self.workflow_executor = WorkflowExecutor(
        config=self.config,
        inference_engine=self,
    )
    self.workflow_executor.initialize(
        logger=self.logger,
        train_data_parallel_size=train_data_parallel_size
    )
    self._initialized = True


def apply_remote_inf_engine_patch():
    """
    Apply the patched initialize method to RemoteInfEngine.
    
    This must be called BEFORE any RemoteInfEngine is instantiated.
    """
    try:
        from areal.infra.remote_inf_engine import RemoteInfEngine
    except ImportError:
        from areal.core.remote_inf_engine import RemoteInfEngine
    
    # Store original for reference
    RemoteInfEngine._original_initialize = RemoteInfEngine.initialize
    
    # Replace with patched version
    RemoteInfEngine.initialize = patched_initialize
    
    logger.info("SPCS PATCH: Applied RemoteInfEngine.initialize patch")
    logger.info(f"  - Judge resolve timeout: {JUDGE_RESOLVE_TIMEOUT}s (was 1s)")
    logger.info("  - Silent fallback disabled: will ERROR if name_resolve fails")
    
    return True
