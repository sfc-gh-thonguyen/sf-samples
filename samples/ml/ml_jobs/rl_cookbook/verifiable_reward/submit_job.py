#!/usr/bin/env python3
"""Submit verifiable_reward GRPO training as an ML Job.

Usage:
    python submit_job.py \
        --compute-pool RL_A100_POOL \
        --external-access-integrations RL_TRAINING_EAI ALLOW_ALL_INTEGRATION \
        --database RL_TRAINING_DB \
        --schema RL_SCHEMA
"""
import argparse
import os
import sys

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
    """Submit GRPO training job via ML Jobs API."""
    config_path = os.path.join(payload_dir, config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    kwargs = {}
    if database and schema:
        kwargs["stage_name"] = f"{database}.{schema}.{stage_name}"
    else:
        kwargs["stage_name"] = stage_name

    # Use the custom AReaL image via spec_overrides.
    # Override command AND args to bypass the Container Runtime's
    # _entrypoint.sh / start_headless (not present in our custom image).
    # The ML Jobs API mounts payload at /mnt/job_stage/app/, so we use
    # absolute paths for both the entrypoint and config file.
    spec_overrides = {
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "image": "/rl_training_db/rl_schema/rl_images/areal-fresh:v5",
                    "command": [
                        "/AReaL/.venv/bin/python3",
                    ],
                    "args": [
                        "-u",
                        "/mnt/job_stage/app/run_training.py",
                        "--config",
                        f"/mnt/job_stage/app/{config}",
                    ],
                    "resources": {
                        "requests": {"nvidia.com/gpu": 4, "memory": "80Gi"},
                        "limits": {"nvidia.com/gpu": 4, "memory": "160Gi"},
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
                {"name": "dev-shm", "source": "memory", "size": "48Gi"},
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
    }

    return jobs.submit_directory(
        payload_dir,
        entrypoint="run_training.py",
        args=["--config", config],
        compute_pool=compute_pool,
        external_access_integrations=external_access_integrations,
        env_vars=env_vars,
        spec_overrides=spec_overrides,
        session=session,
        **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit GRPO training as ML Job")
    parser.add_argument(
        "-p", "--compute-pool", default="RL_A100_POOL", help="GPU compute pool"
    )
    parser.add_argument("--database", default="RL_TRAINING_DB", help="Snowflake database")
    parser.add_argument("--schema", default="RL_SCHEMA", help="Snowflake schema")
    parser.add_argument(
        "--stage-name", default="rl_payload_stage", help="Stage for job artifacts"
    )
    parser.add_argument(
        "-e",
        "--external-access-integrations",
        nargs="+",
        default=["RL_TRAINING_EAI", "ALLOW_ALL_INTEGRATION"],
        help="External access integrations",
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Training config file"
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="Submit and exit without waiting"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Build Snowpark session
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

    print(f"Submitting GRPO training ML Job...")
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
