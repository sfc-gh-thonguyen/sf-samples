#!/usr/bin/env python3
"""Submit policy RL training with local vLLM judge as an ML Job.

Usage:
    SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python submit_job.py --no-wait
"""
import argparse
import os

from snowflake.snowpark import Session
from snowflake.ml import jobs


def submit_training_job(
    session: Session,
    payload_dir: str,
    config: str,
    compute_pool: str,
    external_access_integrations: list,
    database: str = None,
    schema: str = None,
    stage_name: str = "rl_payload_stage",
) -> jobs.MLJob:
    """Submit policy RL training job via ML Jobs API."""
    config_path = os.path.join(payload_dir, config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Override command AND args to bypass the Container Runtime's
    # start_headless (not present in our custom AReaL image).
    # ML Jobs mounts payload at /mnt/job_stage/app/.
    spec_overrides = {
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "image": "/rl_training_db/rl_schema/rl_images/areal-fresh:v7",
                    "command": [
                        "/AReaL/.venv/bin/python3",
                    ],
                    "args": [
                        "-u",
                        "/mnt/job_stage/app/run_policy_rl.py",
                        "--config",
                        f"/mnt/job_stage/app/{config}",
                    ],
                    "resources": {
                        "requests": {"nvidia.com/gpu": 8, "memory": "160Gi"},
                        "limits": {"nvidia.com/gpu": 8, "memory": "320Gi"},
                    },
                    "secrets": [
                        {
                            "snowflakeSecret": {
                                "objectName": "rl_training_db.rl_schema.wandb_api_key_secret",
                            },
                            "secretKeyRef": "secret_string",
                            "envVarName": "WANDB_API_KEY",
                        }
                    ],
                }
            ],
            "volumes": [
                {"name": "dev-shm", "source": "memory", "size": "96Gi"},
            ],
        }
    }

    env_vars = {
        "HF_HOME": "/tmp/hf_local",
        "TRANSFORMERS_CACHE": "/tmp/hf_local",
        "HUGGINGFACE_HUB_CACHE": "/tmp/hf_local",
        "WANDB_BASE_URL": "https://snowflake.wandb.io",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "0",
        "PYTHONUNBUFFERED": "1",
        # Local vLLM judge configuration (4 judges on GPUs 4-7)
        "LOCAL_JUDGE_MODEL": "Qwen/Qwen3-8B",
        "LOCAL_JUDGE_PORTS": "38899,38900,38901,38902",
        "LOCAL_JUDGE_BASE_PORT": "38899",
        # Phase 1: deterministic-only (0), Phase 2: with judge (1)
        "ENABLE_LLM_JUDGE": "0",
        # Warm-start from v27 epoch 9 checkpoint
        "INIT_MODEL_STAGE_PATH": "@RL_TRAINING_DB.RL_SCHEMA.RL_PAYLOAD_STAGE/run_policy_rld3d559a7/output/model_output_local_judge_v27/checkpoints/root/policy-rl-summarization/local_judge_v27/default/epoch9epochstep14globalstep149",
    }

    return jobs.submit_directory(
        payload_dir,
        entrypoint="run_policy_rl.py",
        args=["--config", config],
        compute_pool=compute_pool,
        external_access_integrations=external_access_integrations,
        env_vars=env_vars,
        spec_overrides=spec_overrides,
        stage_name=f"{database}.{schema}.{stage_name}" if database and schema else stage_name,
        database=database,
        schema=schema,
        session=session,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit policy RL training with local judge")
    parser.add_argument(
        "-p", "--compute-pool", default="RL_LOCAL_JUDGE_POOL", help="GPU compute pool"
    )
    parser.add_argument("--database", default="RL_TRAINING_DB")
    parser.add_argument("--schema", default="RL_SCHEMA")
    parser.add_argument("--stage-name", default="rl_payload_stage")
    parser.add_argument(
        "-e",
        "--external-access-integrations",
        nargs="+",
        default=["RL_TRAINING_EAI", "ALLOW_ALL_INTEGRATION"],
    )
    parser.add_argument(
        "-c", "--config", default="config_policy_rl.yaml",
    )
    parser.add_argument("--no-wait", action="store_true", help="Submit and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    connection_name = os.getenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME")
    builder = Session.builder
    if connection_name:
        builder = builder.config("connection_name", connection_name)
    if args.database:
        builder = builder.config("database", args.database)
    if args.schema:
        builder = builder.config("schema", args.schema)
    builder = builder.config("role", "SYSADMIN")
    session = builder.create()

    if args.database:
        session.sql(f"USE DATABASE {args.database}").collect()
    if args.schema:
        session.sql(f"USE SCHEMA {args.schema}").collect()

    payload_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Submitting policy RL training with local judge...")
    print(f"  Payload dir: {payload_dir}")
    print(f"  Config: {args.config}")
    print(f"  Compute pool: {args.compute_pool}")
    print(f"  Database: {args.database}.{args.schema}")

    job = submit_training_job(
        session=session,
        payload_dir=payload_dir,
        config=args.config,
        compute_pool=args.compute_pool,
        external_access_integrations=args.external_access_integrations,
        database=args.database,
        schema=args.schema,
        stage_name=args.stage_name,
    )

    print(f"Job submitted: {job.id}")

    if args.no_wait:
        print("--no-wait: job running in background.")
    else:
        print("Waiting for job to complete...")
        status = job.wait()
        print(f"Job finished with status: {status}")
        if args.verbose:
            print(f"\nLogs:\n{job.get_logs()}")
