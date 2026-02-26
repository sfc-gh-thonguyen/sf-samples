"""Submit multi-node RL training job to Snowflake SPCS.

This script submits a multi-node GRPO training job using AReaL framework
on Snowflake SPCS (Snowpark Container Services).

Usage (2-node, default):
    python scripts/run_rl_train.py \
        --compute-pool GPU_POOL \
        --external-access-integrations RL_EAI \
        --database LLM_DEMO \
        --schema PUBLIC \
        --num-nodes 2

Usage (3-node with external judge):
    # First start judge server (separate job), then:
    python scripts/run_rl_train.py \
        --compute-pool GPU_POOL \
        --external-access-integrations RL_EAI \
        --database LLM_DEMO \
        --schema PUBLIC \
        --num-nodes 2 \
        --3node \
        --config grpo_3node_config.yaml
"""
import argparse
import os

from snowflake.snowpark import Session
from snowflake.ml import jobs


def submit_rl_job(
    session: Session,
    payload_dir: str,
    config: str,
    compute_pool: str,
    external_access_integrations: list,
    target_instances: int = 2,
    database: str = None,
    schema: str = None,
    stage_name: str = 'rl_payload_stage',
    runtime_environment: str = None,
    memory_limit: str = None,
    use_3node_mode: bool = False,
    trainer: str = 'soap_grpo_trainer.py',
) -> jobs.MLJob:
    """Submit multi-node RL training job to SPCS.
    
    Args:
        session: Snowpark session
        payload_dir: Directory containing training code and configs
        config: Name of the training config YAML file
        compute_pool: GPU compute pool name
        external_access_integrations: List of EAIs for network access
        target_instances: Number of nodes (default: 2 for rollout + trainer)
        database: Snowflake database
        schema: Snowflake schema
        stage_name: Stage name for job artifacts
        runtime_environment: Runtime image tag or URL (e.g., "2.1.4-py312" for Python 3.12)
        memory_limit: Memory limit per container (e.g., "150G"). If None, uses default.
        use_3node_mode: If True, enable 3-node mode (separate judge server required)
        trainer: Name of the trainer Python file
    
    Returns:
        MLJob instance
    """
    if not os.path.isfile(os.path.join(payload_dir, config)):
        raise FileNotFoundError(f"Config file not found: {os.path.join(payload_dir, config)}")
    
    kwargs = {}
    if database and schema:
        kwargs['stage_name'] = f"{database}.{schema}.{stage_name}"
    else:
        kwargs['stage_name'] = stage_name
    if runtime_environment:
        kwargs['runtime_environment'] = runtime_environment
    
    # Build spec overrides for memory and environment variables
    spec_overrides = {}
    env_vars = {
        "AREAL_NUM_NODES": str(target_instances),
    }
    if use_3node_mode:
        env_vars["AREAL_3NODE_MODE"] = "1"
    
    if memory_limit or env_vars:
        container_spec = {"name": "main"}
        if memory_limit:
            container_spec["resources"] = {
                "limits": {"memory": memory_limit},
                "requests": {"memory": memory_limit}
            }
        if env_vars:
            # SPCS spec expects env as a dict, not a list
            container_spec["env"] = env_vars
        spec_overrides = {
            "spec": {
                "containers": [container_spec]
            }
        }
        kwargs['spec_overrides'] = spec_overrides
    
    return jobs.submit_directory(
        payload_dir,
        entrypoint="log_wrapper.py",
        args=["run_areal.py", config, trainer],
        compute_pool=compute_pool,
        target_instances=target_instances,
        external_access_integrations=external_access_integrations,
        session=session,
        **kwargs,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit multi-node RL training job to SPCS')
    parser.add_argument('-p', '--compute-pool', required=True, help='GPU compute pool name')
    parser.add_argument('-n', '--num-nodes', type=int, default=1, 
                        help='Number of nodes (default: 1 for single-node multi-GPU)')
    parser.add_argument('--database', help='Snowflake database')
    parser.add_argument('--schema', help='Snowflake schema')
    parser.add_argument('--role', default='SYSADMIN', help='Snowflake role (default: SYSADMIN)')
    parser.add_argument('--stage-name', default='rl_payload_stage', help='Stage name for job artifacts')
    parser.add_argument('-e', '--external-access-integrations', nargs="+", required=True,
                        help='External access integrations for PyPI, HuggingFace, and GitHub access')
    parser.add_argument('-c', '--config', default='grpo_lora_config.yaml', help='Training config file')
    parser.add_argument('-t', '--trainer', default='soap_grpo_trainer.py', help='Trainer script file')
    parser.add_argument('--runtime', default=None, help='Runtime environment (e.g., "1.7.1-py312-gpu" for Python 3.12)')
    parser.add_argument('--memory', default=None, help='Memory limit per container (e.g., "150G")')
    parser.add_argument('--3node', dest='three_node', action='store_true', 
                        help='Enable 3-node mode (requires separate judge server)')
    parser.add_argument('--no-wait', dest='no_wait', action='store_true',
                        help='Submit job and exit immediately without waiting')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    connection_name = os.getenv('SNOWFLAKE_DEFAULT_CONNECTION_NAME')
    builder = Session.builder
    if connection_name:
        builder = builder.config('connection_name', connection_name)
    if args.database:
        builder = builder.config('database', args.database)
    if args.schema:
        builder = builder.config('schema', args.schema)
    if args.role:
        builder = builder.config('role', args.role)
    session = builder.create()
    
    if args.database:
        session.sql(f"USE DATABASE {args.database}").collect()
    if args.schema:
        session.sql(f"USE SCHEMA {args.schema}").collect()
    
    payload_dir = os.path.join(os.path.dirname(__file__), "..", "src")

    print(f"Submitting {args.num_nodes}-node RL training job...")
    print(f"  Compute pool: {args.compute_pool}")
    print(f"  Config: {args.config}")
    print(f"  Database: {args.database}")
    print(f"  Schema: {args.schema}")
    print(f"  Runtime: {args.runtime or 'default'}")
    print(f"  Memory limit: {args.memory or 'default'}")
    print(f"  Trainer: {args.trainer}")
    print(f"  3-node mode: {args.three_node}")
    
    if args.three_node:
        print("\n  NOTE: 3-node mode enabled. Make sure judge server is running first!")
        print("        Judge should have experiment_name matching config's judge.experiment_name")
    
    job = submit_rl_job(
        session=session,
        payload_dir=payload_dir,
        config=args.config,
        compute_pool=args.compute_pool,
        external_access_integrations=args.external_access_integrations,
        target_instances=args.num_nodes,
        database=args.database,
        schema=args.schema,
        stage_name=args.stage_name,
        runtime_environment=args.runtime,
        memory_limit=args.memory,
        use_3node_mode=args.three_node,
        trainer=args.trainer,
    )

    print(f"Job submitted with ID: {job.id}")
    
    if args.no_wait:
        print("--no-wait specified. Job running in background.")
        print(f"Check status: SELECT SYSTEM$GET_SERVICE_LOGS('{job.id}', 0, 'main', 1000);")
    else:
        print("Waiting for job to complete...")
        status = job.wait()
        print(f"Job finished with status: {status}")
        
        if args.verbose:
            print(f"\nJob logs:\n{job.get_logs()}")
