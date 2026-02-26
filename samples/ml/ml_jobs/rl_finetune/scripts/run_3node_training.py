"""Submit 3-node RL training setup to Snowflake SPCS.

This script handles the complete 3-node setup:
1. Launches judge server (1 node)
2. Waits for judge to be ready
3. Launches training job (2 nodes: rollout + trainer)

Usage:
    python scripts/run_3node_training.py \
        --compute-pool RL_GPU_POOL \
        --external-access-integrations PYPI_HF_EAI \
        --database RL_TRAINING_DB \
        --schema RL_SCHEMA \
        --runtime "<registry>/areal-runtime:v0.5.3-ray253-fix6"
"""
import argparse
import os
import time

from snowflake.snowpark import Session
from snowflake.ml import jobs


def submit_job(
    session: Session,
    payload_dir: str,
    config: str,
    compute_pool: str,
    external_access_integrations: list,
    target_instances: int,
    database: str,
    schema: str,
    stage_name: str,
    runtime_environment: str,
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
    
    if env_vars:
        kwargs['spec_overrides'] = {
            "spec": {
                "containers": [{
                    "name": "main",
                    "env": env_vars
                }]
            }
        }
    
    return jobs.submit_directory(
        payload_dir,
        entrypoint="run_areal.py",
        args=[config],
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
                import json
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit 3-node RL training to SPCS')
    parser.add_argument('-p', '--compute-pool', required=True, help='GPU compute pool name')
    parser.add_argument('--database', required=True, help='Snowflake database')
    parser.add_argument('--schema', required=True, help='Snowflake schema')
    parser.add_argument('--role', default='SYSADMIN', help='Snowflake role')
    parser.add_argument('--stage-name', default='rl_payload_stage', help='Stage name')
    parser.add_argument('-e', '--external-access-integrations', nargs="+", required=True,
                        help='External access integrations')
    parser.add_argument('--runtime', required=True, help='Runtime image URL')
    parser.add_argument('--judge-config', default='judge_server_config.yaml', 
                        help='Judge server config file')
    parser.add_argument('--train-config', default='grpo_3node_config.yaml',
                        help='Training config file')
    parser.add_argument('--skip-judge', action='store_true',
                        help='Skip launching judge (assume already running)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Connect to Snowflake
    connection_name = os.getenv('SNOWFLAKE_DEFAULT_CONNECTION_NAME')
    builder = Session.builder
    if connection_name:
        builder = builder.config('connection_name', connection_name)
    builder = builder.config('database', args.database)
    builder = builder.config('schema', args.schema)
    builder = builder.config('role', args.role)
    session = builder.create()
    
    session.sql(f"USE DATABASE {args.database}").collect()
    session.sql(f"USE SCHEMA {args.schema}").collect()
    
    payload_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    
    print("=" * 60)
    print("3-NODE RL TRAINING SETUP")
    print("=" * 60)
    print(f"Compute pool: {args.compute_pool}")
    print(f"Runtime: {args.runtime}")
    print(f"Judge config: {args.judge_config}")
    print(f"Train config: {args.train_config}")
    
    judge_job = None
    
    # Step 1: Launch judge server
    if not args.skip_judge:
        print("\n" + "=" * 60)
        print("STEP 1: Launching Judge Server (1 node)")
        print("=" * 60)
        
        judge_job = submit_job(
            session=session,
            payload_dir=payload_dir,
            config=args.judge_config,
            compute_pool=args.compute_pool,
            external_access_integrations=args.external_access_integrations,
            target_instances=1,
            database=args.database,
            schema=args.schema,
            stage_name=args.stage_name,
            runtime_environment=args.runtime,
        )
        print(f"Judge job submitted: {judge_job.id}")
        
        # Wait for judge to be running
        print("\nWaiting for judge to start (timeout: 5 min)...")
        if not wait_for_job_running(session, judge_job.id, timeout=300):
            print("ERROR: Judge server failed to start!")
            exit(1)
        
        # Give judge extra time to register with name_resolve
        print("Judge running. Waiting 60s for name_resolve registration...")
        time.sleep(60)
    else:
        print("\n[Skipping judge launch - assuming already running]")
    
    # Step 2: Launch training job
    print("\n" + "=" * 60)
    print("STEP 2: Launching Training Job (2 nodes: rollout + trainer)")
    print("=" * 60)
    
    train_job = submit_job(
        session=session,
        payload_dir=payload_dir,
        config=args.train_config,
        compute_pool=args.compute_pool,
        external_access_integrations=args.external_access_integrations,
        target_instances=2,
        database=args.database,
        schema=args.schema,
        stage_name=args.stage_name,
        runtime_environment=args.runtime,
        env_vars={
            "AREAL_3NODE_MODE": "1",
            "AREAL_NUM_NODES": "2",
        },
    )
    print(f"Training job submitted: {train_job.id}")
    
    # Wait for training to complete
    print("\nWaiting for training job to complete...")
    status = train_job.wait()
    print(f"\nTraining job finished with status: {status}")
    
    if args.verbose:
        print(f"\nTraining job logs:\n{train_job.get_logs()}")
    
    # Cleanup judge if we started it
    if judge_job:
        print("\nStopping judge server...")
        try:
            session.sql(f"DROP SERVICE IF EXISTS {judge_job.id}").collect()
            print("Judge server stopped.")
        except Exception as e:
            print(f"Warning: Could not stop judge: {e}")
    
    print("\n" + "=" * 60)
    print("3-NODE TRAINING COMPLETE")
    print("=" * 60)
