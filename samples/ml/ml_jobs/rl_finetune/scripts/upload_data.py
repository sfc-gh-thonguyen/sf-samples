"""Upload training datasets to Snowflake stage for SPCS job.

This script uploads the existing synthetic_train_data.json and synthetic_test_data.json
from rl_finetune_reference/ to a Snowflake stage for use by SPCS training jobs.

Usage:
    python scripts/upload_data.py --database LLM_DEMO --schema PUBLIC
"""
import argparse
import os

from snowflake.snowpark import Session


def upload_datasets(
    session: Session,
    data_dir: str,
    stage_name: str,
    database: str,
    schema: str,
):
    """Upload synthetic_train_data.json and synthetic_test_data.json to stage."""
    full_stage = f"{database}.{schema}.{stage_name}"
    
    session.sql(f"""
        CREATE STAGE IF NOT EXISTS {full_stage}
        ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    """).collect()
    print(f"Stage {full_stage} ready")
    
    datasets = [
        ("synthetic_train_data.json", "train"),
        ("synthetic_test_data.json", "test"),
    ]
    
    for filename, desc in datasets:
        local_path = os.path.join(data_dir, filename)
        if os.path.exists(local_path):
            session.sql(f"""
                PUT file://{local_path} @{full_stage}/data/
                AUTO_COMPRESS=FALSE OVERWRITE=TRUE
            """).collect()
            print(f"Uploaded {filename} ({desc} data) to @{stage_name}/data/")
        else:
            print(f"Warning: {local_path} not found")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload training datasets to Snowflake stage')
    parser.add_argument('--database', required=True, help='Snowflake database')
    parser.add_argument('--schema', required=True, help='Snowflake schema')
    parser.add_argument('--stage-name', default='rl_payload_stage', help='Stage name for job artifacts')
    args = parser.parse_args()

    session = Session.builder.getOrCreate()
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "rl_finetune_reference")
    
    upload_datasets(session, data_dir, args.stage_name, args.database, args.schema)
    print("Data upload complete!")
