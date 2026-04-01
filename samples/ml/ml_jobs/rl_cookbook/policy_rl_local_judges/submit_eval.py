#!/usr/bin/env python3
"""Submit checkpoint evaluation as a lightweight SPCS job.

Usage:
    # Evaluate a checkpoint from stage:
    SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python submit_eval.py \
        --checkpoint-path "@RL_TRAINING_DB.RL_SCHEMA.RL_PAYLOAD_STAGE/run_policy_rl1c5fd0d8/output/model_output_local_judge_v17/checkpoints/root/policy-rl-summarization/local_judge_v17/default/epoch9epochstep14globalstep149"

    # Evaluate the base model (no checkpoint):
    SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python submit_eval.py \
        --base-model "Qwen/Qwen3-1.7B"

    # More samples:
    SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python submit_eval.py \
        --checkpoint-path "..." --num-samples 20
"""
import argparse
import os

from snowflake.snowpark import Session
from snowflake.ml import jobs


def submit_eval_job(
    session: Session,
    payload_dir: str,
    checkpoint_path: str,
    num_samples: int,
    compute_pool: str,
    external_access_integrations: list,
    base_model: str = "",
    database: str = None,
    schema: str = None,
    stage_name: str = "rl_payload_stage",
) -> jobs.MLJob:
    """Submit checkpoint evaluation job."""
    spec_overrides = {
        "spec": {
            "containers": [
                {
                    "name": "main",
                    "image": "/rl_training_db/rl_schema/rl_images/areal-fresh:v7",
                    "command": ["/AReaL/.venv/bin/python3"],
                    "args": [
                        "-u",
                        "/mnt/job_stage/app/eval_checkpoint.py",
                    ],
                    "resources": {
                        "requests": {"nvidia.com/gpu": 1, "memory": "40Gi"},
                        "limits": {"nvidia.com/gpu": 1, "memory": "80Gi"},
                    },
                }
            ],
        }
    }

    env_vars = {
        "HF_HOME": "/tmp/hf_cache",
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        "PYTHONUNBUFFERED": "1",
        "CHECKPOINT_STAGE_PATH": checkpoint_path,
        "BASE_MODEL": base_model,
        "NUM_SAMPLES": str(num_samples),
    }

    return jobs.submit_directory(
        payload_dir,
        entrypoint="eval_checkpoint.py",
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
    parser = argparse.ArgumentParser(description="Submit checkpoint evaluation job")
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Full stage path to checkpoint dir, e.g. @DB.SCHEMA.STAGE/path/to/epoch9...",
    )
    parser.add_argument(
        "--base-model",
        default="",
        help="HuggingFace model ID to evaluate instead of a checkpoint (e.g. Qwen/Qwen3-1.7B)",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("-p", "--compute-pool", default="RL_LOCAL_JUDGE_POOL")
    parser.add_argument("--database", default="RL_TRAINING_DB")
    parser.add_argument("--schema", default="RL_SCHEMA")
    parser.add_argument("--stage-name", default="rl_payload_stage")
    parser.add_argument(
        "-e", "--external-access-integrations",
        nargs="+",
        default=["RL_TRAINING_EAI", "ALLOW_ALL_INTEGRATION"],
    )
    parser.add_argument("--no-wait", action="store_true")
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

    if not args.checkpoint_path and not args.base_model:
        print("ERROR: Provide --checkpoint-path or --base-model")
        parser.print_help()
        exit(1)

    print(f"Submitting checkpoint evaluation...")
    if args.base_model:
        print(f"  Base model: {args.base_model}")
    else:
        print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Compute pool: {args.compute_pool}")

    job = submit_eval_job(
        session=session,
        payload_dir=payload_dir,
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        compute_pool=args.compute_pool,
        external_access_integrations=args.external_access_integrations,
        base_model=args.base_model,
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
