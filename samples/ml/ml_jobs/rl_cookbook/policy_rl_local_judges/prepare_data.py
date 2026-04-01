#!/usr/bin/env python3
"""
Prepare Policy RL Data — Upload JSONL to Snowflake Tables

Reads JSONL files (one JSON object per line) with keys:
  - transcript: raw customer interaction text
  - metadata: JSON string of ground-truth labels (PII, tier, product, etc.)
  - prompt: the formatted prompt given to the model

Creates POLICY_RL_TRAIN and POLICY_RL_EVAL tables in the target schema
and inserts all rows. Run once before submitting training jobs.

Usage:
    SNOWFLAKE_DEFAULT_CONNECTION_NAME=preprod8 python prepare_data.py \
        --train-file train.jsonl --eval-file eval.jsonl
"""
import argparse
import json
import os

from snowflake.snowpark import Session


def create_table(session, table_name):
    """Create the table if it doesn't exist."""
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            TRANSCRIPT VARCHAR,
            METADATA VARCHAR,
            PROMPT VARCHAR
        )
    """).collect()
    print(f"  Table {table_name} ready")


def load_jsonl(path):
    """Read a JSONL file and return list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Read {len(records)} records from {path}")
    return records


def upload_records(session, table_name, records, batch_size=100):
    """Insert records into the table in batches."""
    # Truncate first to avoid duplicates on re-run
    session.sql(f"TRUNCATE TABLE IF EXISTS {table_name}").collect()

    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        rows = []
        for r in batch:
            transcript = r.get("transcript", "")
            metadata = r.get("metadata", "{}")
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata)
            prompt = r.get("prompt", "")
            rows.append((transcript, metadata, prompt))

        if rows:
            df = session.create_dataframe(rows, schema=["TRANSCRIPT", "METADATA", "PROMPT"])
            df.write.mode("append").save_as_table(table_name)
            total += len(rows)

    print(f"  Uploaded {total} rows to {table_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload JSONL data to Snowflake tables")
    parser.add_argument("--train-file", required=True, help="Path to training JSONL file")
    parser.add_argument("--eval-file", required=True, help="Path to evaluation JSONL file")
    parser.add_argument("--database", default="RL_TRAINING_DB")
    parser.add_argument("--schema", default="RL_SCHEMA")
    parser.add_argument("--train-table", default="POLICY_RL_TRAIN")
    parser.add_argument("--eval-table", default="POLICY_RL_EVAL")
    args = parser.parse_args()

    connection_name = os.getenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME")
    builder = Session.builder
    if connection_name:
        builder = builder.config("connection_name", connection_name)
    builder = builder.config("database", args.database)
    builder = builder.config("schema", args.schema)
    session = builder.create()

    session.sql(f"USE DATABASE {args.database}").collect()
    session.sql(f"USE SCHEMA {args.schema}").collect()

    train_table = f"{args.database}.{args.schema}.{args.train_table}"
    eval_table = f"{args.database}.{args.schema}.{args.eval_table}"

    print("Creating tables...")
    create_table(session, train_table)
    create_table(session, eval_table)

    print(f"\nUploading training data from {args.train_file}...")
    train_records = load_jsonl(args.train_file)
    upload_records(session, train_table, train_records)

    print(f"\nUploading eval data from {args.eval_file}...")
    eval_records = load_jsonl(args.eval_file)
    upload_records(session, eval_table, eval_records)

    # Verify
    train_count = session.sql(f"SELECT COUNT(*) FROM {train_table}").collect()[0][0]
    eval_count = session.sql(f"SELECT COUNT(*) FROM {eval_table}").collect()[0][0]
    print(f"\nDone. {train_table}: {train_count} rows, {eval_table}: {eval_count} rows")


if __name__ == "__main__":
    main()
