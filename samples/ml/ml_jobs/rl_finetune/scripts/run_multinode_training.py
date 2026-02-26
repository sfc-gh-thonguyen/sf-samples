"""Submit multi-node RL training job to Snowflake SPCS (Unified).

This script supports N-node configurations with the new unified node allocation schema.
It replaces both run_rl_train.py and run_3node_training.py.

Usage Examples:

  # 2-node training (default: 1 rollout + 1 trainer)
  python scripts/run_multinode_training.py \
      --compute-pool GPU_POOL \
      --external-access-integrations RL_EAI \
      --database LLM_DEMO \
      --schema PUBLIC \
      --config grpo_multinode_template.yaml

  # 3-node training with external judge
  python scripts/run_multinode_training.py \
      --compute-pool GPU_POOL \
      --external-access-integrations RL_EAI \
      --database LLM_DEMO \
      --schema PUBLIC \
      --config grpo_3node_config.yaml \
      --with-judge \
      --judge-config judge_server_config.yaml

  # 4-node training (2 rollout + 2 trainer)
  python scripts/run_multinode_training.py \
      --compute-pool GPU_POOL \
      --external-access-integrations RL_EAI \
      --database LLM_DEMO \
      --schema PUBLIC \
      --config grpo_4node_config.yaml
"""
import argparse
import json
import os
import sys
import time
import yaml

from snowflake.snowpark import Session
from snowflake.ml import jobs


def load_config(config_path: str) -> dict:
    """Load and parse YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_node_allocation(config: dict) -> tuple[int, dict]:
    """Extract node allocation from config.
    
    Returns:
        Tuple of (total_nodes, roles_dict)
    """
    if 'nodes' in config:
        nodes = config['nodes']
        return nodes.get('total', 2), nodes.get('roles', {'rollout': 1, 'trainer': 1})
    
    if 'cluster' in config:
        n_nodes = config['cluster'].get('n_nodes', 2)
        return n_nodes, {'rollout': 1, 'trainer': max(1, n_nodes - 1)}
    
    return 2, {'rollout': 1, 'trainer': 1}


def submit_job(
    session: Session,
    payload_dir: str,
    entrypoint: str,
    args: list,
    compute_pool: str,
    external_access_integrations: list,
    target_instances: int,
    database: str,
    schema: str,
    stage_name: str,
    runtime_environment: str = None,
    memory_limit: str = None,
    env_vars: dict = None,
) -> jobs.MLJob:
    """Submit a job to SPCS."""
    kwargs = {}
    if database and schema:
        kwargs['stage_name'] = f"{database}.{schema}.{stage_name}"
    else:
        kwargs['stage_name'] = stage_name
    if runtime_environment:
        kwargs['runtime_environment'] = runtime_environment
    
    spec_overrides = {}
    container_spec = {"name": "main"}
    
    if memory_limit:
        container_spec["resources"] = {
            "limits": {"memory": memory_limit},
            "requests": {"memory": memory_limit}
        }
    if env_vars:
        container_spec["env"] = env_vars
    
    if memory_limit or env_vars:
        spec_overrides = {"spec": {"containers": [container_spec]}}
        kwargs['spec_overrides'] = spec_overrides
    
    return jobs.submit_directory(
        payload_dir,
        entrypoint=entrypoint,
        args=args,
        compute_pool=compute_pool,
        target_instances=target_instances,
        external_access_integrations=external_access_integrations,
        session=session,
        **kwargs,
    )


def wait_for_job_running(session: Session, job_id: str, timeout: int = 300) -> bool:
    """Wait for a job to reach RUNNING status."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = session.sql(f"SELECT SYSTEM$GET_SERVICE_STATUS('{job_id}')").collect()
            if result:
                status_json = json.loads(result[0][0])
                status = status_json[0].get('status', 'UNKNOWN')
                print(f"  Job {job_id}: {status}")
                if status == 'RUNNING':
                    return True
                elif status in ('FAILED', 'DONE'):
                    return False
        except Exception as e:
            print(f"  Error checking status: {e}")
        time.sleep(10)
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Submit multi-node RL training job to SPCS (unified)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('-p', '--compute-pool', required=True, help='GPU compute pool name')
    parser.add_argument('--database', help='Snowflake database')
    parser.add_argument('--schema', help='Snowflake schema')
    parser.add_argument('--role', default='SYSADMIN', help='Snowflake role (default: SYSADMIN)')
    parser.add_argument('--stage-name', default='rl_payload_stage', help='Stage name for job artifacts')
    parser.add_argument('-e', '--external-access-integrations', nargs="+", required=True,
                        help='External access integrations for PyPI, HuggingFace access')
    
    parser.add_argument('-c', '--config', default='grpo_multinode_template.yaml',
                        help='Training config file (should contain nodes section)')
    parser.add_argument('-t', '--trainer', default='soap_grpo_trainer.py', help='Trainer script file')
    parser.add_argument('--runtime', default=None, help='Runtime image URL or tag')
    parser.add_argument('--memory', default=None, help='Memory limit per container (e.g., "150G")')
    
    parser.add_argument('--with-judge', action='store_true',
                        help='Launch external judge server before training')
    parser.add_argument('--judge-config', default='judge_server_config.yaml',
                        help='Judge server config file (used with --with-judge)')
    parser.add_argument('--skip-judge-wait', action='store_true',
                        help='Skip waiting for judge to be ready')
    
    parser.add_argument('--num-nodes', type=int, default=None,
                        help='Override number of nodes (default: from config)')
    
    parser.add_argument('--no-wait', action='store_true',
                        help='Submit job and exit immediately without waiting')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    parser.add_argument('--legacy-3node', action='store_true',
                        help='[DEPRECATED] Use legacy AREAL_3NODE_MODE env var')
    
    args = parser.parse_args()
    
    if args.legacy_3node:
        print("WARNING: --legacy-3node is deprecated. Use 'nodes' section in config instead.")
    
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
    config_path = os.path.join(payload_dir, args.config)
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    total_nodes, roles = get_node_allocation(config)
    
    if args.num_nodes is not None:
        total_nodes = args.num_nodes
        print(f"WARNING: Overriding config nodes.total with --num-nodes={total_nodes}")
    
    print("=" * 60)
    print("UNIFIED MULTI-NODE RL TRAINING")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Compute pool: {args.compute_pool}")
    print(f"Database: {args.database}")
    print(f"Schema: {args.schema}")
    print(f"Runtime: {args.runtime or 'default'}")
    print(f"Memory limit: {args.memory or 'default'}")
    print(f"Trainer: {args.trainer}")
    print()
    print(f"Node allocation: {total_nodes} nodes")
    for role, count in roles.items():
        print(f"  - {role}: {count} node(s)")
    
    judge_job = None
    
    if args.with_judge:
        print("\n" + "=" * 60)
        print("STEP 1: Launching External Judge Server (1 node)")
        print("=" * 60)
        
        judge_config_path = os.path.join(payload_dir, args.judge_config)
        if not os.path.exists(judge_config_path):
            print(f"ERROR: Judge config not found: {judge_config_path}")
            sys.exit(1)
        
        judge_job = submit_job(
            session=session,
            payload_dir=payload_dir,
            entrypoint="run_areal.py",
            args=[args.judge_config],
            compute_pool=args.compute_pool,
            external_access_integrations=args.external_access_integrations,
            target_instances=1,
            database=args.database,
            schema=args.schema,
            stage_name=args.stage_name,
            runtime_environment=args.runtime,
        )
        print(f"Judge job submitted: {judge_job.id}")
        
        if not args.skip_judge_wait:
            print("\nWaiting for judge to start (timeout: 5 min)...")
            if not wait_for_job_running(session, judge_job.id, timeout=300):
                print("ERROR: Judge server failed to start!")
                sys.exit(1)
            
            print("Judge running. Waiting 60s for name_resolve registration...")
            time.sleep(60)
    
    step = "STEP 2" if args.with_judge else "STEP 1"
    print("\n" + "=" * 60)
    print(f"{step}: Launching Training Job ({total_nodes} nodes)")
    print("=" * 60)
    
    env_vars = {"AREAL_NUM_NODES": str(total_nodes)}
    
    if args.legacy_3node:
        env_vars["AREAL_3NODE_MODE"] = "1"
    
    train_job = submit_job(
        session=session,
        payload_dir=payload_dir,
        entrypoint="log_wrapper.py",
        args=["run_areal.py", args.config, args.trainer],
        compute_pool=args.compute_pool,
        external_access_integrations=args.external_access_integrations,
        target_instances=total_nodes,
        database=args.database,
        schema=args.schema,
        stage_name=args.stage_name,
        runtime_environment=args.runtime,
        memory_limit=args.memory,
        env_vars=env_vars,
    )
    
    print(f"Training job submitted: {train_job.id}")
    
    if args.no_wait:
        print("\n--no-wait specified. Job running in background.")
        print(f"Check status: SELECT SYSTEM$GET_SERVICE_STATUS('{train_job.id}');")
        print(f"Get logs: SELECT SYSTEM$GET_SERVICE_LOGS('{train_job.id}', 0, 'main', 1000);")
    else:
        print("\nWaiting for training job to complete...")
        status = train_job.wait()
        print(f"\nTraining job finished with status: {status}")
        
        if args.verbose:
            print(f"\nJob logs:\n{train_job.get_logs()}")
    
    if judge_job and not args.no_wait:
        print("\nStopping judge server...")
        try:
            session.sql(f"DROP SERVICE IF EXISTS {judge_job.id}").collect()
            print("Judge server stopped.")
        except Exception as e:
            print(f"Warning: Could not stop judge: {e}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
