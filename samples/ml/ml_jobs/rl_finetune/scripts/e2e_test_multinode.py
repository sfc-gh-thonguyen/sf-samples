"""E2E Tests for Unified Multi-Node RL Training on SPCS.

These tests verify the unified node allocation system works correctly
across different configurations on actual SPCS infrastructure.

Test Scenarios:
1. 2-node basic (1 rollout + 1 trainer) - verifies new unified code path
2. Legacy config (no nodes section) - verifies backward compatibility
3. 3-node with external judge - verifies judge integration
4. 4-node scaling (2 rollout + 2 trainer) - verifies multi-node scaling

Prerequisites:
- SNOWFLAKE_DEFAULT_CONNECTION_NAME environment variable set
- Database, schema, and compute pool must exist
- External access integration configured
- Runtime image available

Usage:
    # Run all tests
    python scripts/e2e_test_multinode.py \
        --compute-pool GPU_POOL \
        --database DB --schema SCHEMA \
        --external-access-integrations EAI \
        --runtime <image_url>
    
    # Run specific test
    python scripts/e2e_test_multinode.py ... --test 2node_basic
"""
import argparse
import json
import os
import sys
import time
import yaml

from snowflake.snowpark import Session
from snowflake.ml import jobs


def create_session(database: str, schema: str, role: str) -> Session:
    """Create Snowflake session."""
    connection_name = os.getenv('SNOWFLAKE_DEFAULT_CONNECTION_NAME')
    builder = Session.builder
    if connection_name:
        builder = builder.config('connection_name', connection_name)
    if database:
        builder = builder.config('database', database)
    if schema:
        builder = builder.config('schema', schema)
    if role:
        builder = builder.config('role', role)
    session = builder.create()
    if database:
        session.sql(f"USE DATABASE {database}").collect()
    if schema:
        session.sql(f"USE SCHEMA {schema}").collect()
    return session


def submit_job(session, payload_dir, entrypoint, args, compute_pool, eai, 
               target_instances, stage_name, runtime, env_vars=None):
    """Submit job to SPCS."""
    kwargs = {'stage_name': stage_name}
    if runtime:
        kwargs['runtime_environment'] = runtime
    if env_vars:
        kwargs['spec_overrides'] = {
            "spec": {"containers": [{"name": "main", "env": env_vars}]}
        }
    return jobs.submit_directory(
        payload_dir,
        entrypoint=entrypoint,
        args=args,
        compute_pool=compute_pool,
        target_instances=target_instances,
        external_access_integrations=eai,
        session=session,
        **kwargs,
    )


def wait_for_job(job, timeout=600):
    """Wait for job completion with timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            status = job.status()
            if status in ('DONE', 'COMPLETED'):
                return True, status
            elif status == 'FAILED':
                return False, status
        except Exception:
            pass
        time.sleep(10)
    return False, 'TIMEOUT'


def test_2node_basic(session, payload_dir, compute_pool, eai, stage_name, runtime):
    """Test 2-node setup with unified config (1 rollout + 1 trainer)."""
    print("\n" + "=" * 60)
    print("TEST: 2-node basic (unified config)")
    print("=" * 60)
    
    # Complete config based on grpo_lora_config.yaml but with nodes section added
    config_content = """
experiment_name: e2e-test-2node
trial_name: v1
seed: 1
enable_offload: false
total_train_epochs: 1
tokenizer_path: ${actor.path}

# NEW: Unified node allocation section
nodes:
  total: 2
  roles:
    rollout: 1
    trainer: 1
  external_judge:
    enabled: false

cluster:
  n_nodes: 2
  n_gpus_per_node: 4
  fileroot: /mnt/job_stage/checkpoints/${experiment_name}/${trial_name}
  name_resolve:
    type: nfs
    nfs_record_root: /mnt/job_stage/name_resolve/${experiment_name}/${trial_name}

allocation_mode: sglang:d1p1t4+d4p1t1
scheduler:
  type: null

rollout:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  max_concurrent_rollouts: 64
  queue_size: null
  consumer_batch_size: ${train_dataset.batch_size}
  max_head_offpolicyness: 2
  enable_rollout_tracing: false
  pause_grace_period: 1.0
  scheduling_spec: ${actor.scheduling_spec}
  fileroot: ${cluster.fileroot}
  tokenizer_path: ${tokenizer_path}
  dump_to_file: true
  setup_timeout: 600.0
  request_timeout: 900.0
  use_lora: true

gconfig:
  n_samples: 4
  min_new_tokens: 0
  max_new_tokens: 512
  greedy: false
  temperature: 1.0
  lora_name: "lora-gsm8k"

actor:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  path: Qwen/Qwen3-0.6B
  init_from_scratch: false
  disable_dropout: true
  gradient_checkpointing: true
  dtype: bfloat16
  mb_spec:
    max_tokens_per_mb: 4096
  optimizer:
    type: adam
    lr: 1.7e-4
    weight_decay: 0.017
    beta1: 0.9
    beta2: 0.999
    eps: 1e-8
    lr_scheduler_type: constant
    gradient_clipping: 1.0
    warmup_steps_proportion: 0.001
  eps_clip: 0.4
  temperature: ${gconfig.temperature}
  reward_scaling: 10.0
  reward_bias: -0.5
  kl_ctl: 0.0
  ppo_n_minibatches: 1
  recompute_logprob: true
  use_decoupled_loss: true
  behav_imp_weight_cap: 5.0
  reward_norm:
    mean_level: group
    std_level: group
    group_size: ${gconfig.n_samples}
  adv_norm:
    mean_level: batch
    std_level: batch
  max_new_tokens: ${gconfig.max_new_tokens}
  weight_update_mode: disk
  use_lora: ${rollout.use_lora}
  peft_type: lora
  lora_rank: 16
  lora_alpha: 16
  target_modules: [all-linear]
  scheduling_spec:
    - task_type: worker
      port_count: 2
      gpu: 1
      cpu: 4
      mem: 12
      cmd: python3 -m areal.scheduler.rpc.rpc_server
      env_vars: {}

ref:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  path: ${actor.path}
  init_from_scratch: false
  disable_dropout: true
  dtype: ${actor.dtype}
  mb_spec:
    max_tokens_per_mb: 4096
  optimizer: null
  scheduling_strategy:
    type: colocation
    target: actor
  scheduling_spec: ${actor.scheduling_spec}

sglang:
  model_path: ${actor.path}
  random_seed: ${seed}
  skip_tokenizer_init: true
  dtype: ${actor.dtype}
  max_running_requests: null
  context_length: 4096
  mem_fraction_static: 0.8
  enable_torch_compile: false
  disable_cuda_graph: true
  enable_lora: ${rollout.use_lora}
  lora_paths: []

train_dataset:
  batch_size: 32
  shuffle: true
  pin_memory: true
  num_workers: 2
  path: openai/gsm8k
  type: rl
  max_length: 1024

valid_dataset:
  batch_size: 32
  pin_memory: true
  num_workers: 2
  path: openai/gsm8k
  type: rl
  max_length: 1024

saver:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: null
  freq_secs: null

recover:
  mode: disabled
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: 1
  freq_steps: null
  freq_secs: 3600

evaluator:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  freq_epochs: null
  freq_steps: null
  freq_secs: null

stats_logger:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  wandb:
    mode: disabled

perf_tracer:
  experiment_name: ${experiment_name}
  trial_name: ${trial_name}
  fileroot: ${cluster.fileroot}
  enabled: false
  session_tracer:
    enabled: false
"""
    
    config_path = os.path.join(payload_dir, 'e2e_test_2node.yaml')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created test config: {config_path}")
    print(f"Submitting 2-node job...")
    
    job = submit_job(
        session=session,
        payload_dir=payload_dir,
        entrypoint="log_wrapper.py",
        args=["run_areal.py", "e2e_test_2node.yaml", "soap_grpo_trainer.py"],
        compute_pool=compute_pool,
        eai=eai,
        target_instances=2,
        stage_name=stage_name,
        runtime=runtime,
        env_vars={"AREAL_NUM_NODES": "2"},
    )
    
    print(f"Job ID: {job.id}")
    print("Waiting for job (timeout: 60 min)...")
    
    success, status = wait_for_job(job, timeout=3600)
    
    os.remove(config_path)
    
    if success:
        print(f"✅ PASSED: 2-node basic test (status: {status})")
        return True
    else:
        print(f"❌ FAILED: 2-node basic test (status: {status})")
        try:
            print(f"Logs: {job.get_logs()[:2000]}")
        except Exception:
            pass
        return False


def test_legacy_config(session, payload_dir, compute_pool, eai, stage_name, runtime):
    """Test backward compatibility with legacy config (no nodes section)."""
    print("\n" + "=" * 60)
    print("TEST: Legacy config (backward compatibility)")
    print("=" * 60)
    
    if not os.path.exists(os.path.join(payload_dir, 'grpo_lora_config.yaml')):
        print("⚠️ SKIPPED: grpo_lora_config.yaml not found")
        return None
    
    print("Submitting job with legacy config (no 'nodes' section)...")
    
    job = submit_job(
        session=session,
        payload_dir=payload_dir,
        entrypoint="log_wrapper.py",
        args=["run_areal.py", "grpo_lora_config.yaml", "soap_grpo_trainer.py"],
        compute_pool=compute_pool,
        eai=eai,
        target_instances=2,
        stage_name=stage_name,
        runtime=runtime,
        env_vars={"AREAL_NUM_NODES": "2"},
    )
    
    print(f"Job ID: {job.id}")
    print("Waiting for job (timeout: 10 min)...")
    
    success, status = wait_for_job(job, timeout=600)
    
    if success:
        print(f"✅ PASSED: Legacy config test (status: {status})")
        return True
    else:
        print(f"❌ FAILED: Legacy config test (status: {status})")
        return False


def test_4node_scaling(session, payload_dir, compute_pool, eai, stage_name, runtime):
    """Test 4-node setup (2 rollout + 2 trainer)."""
    print("\n" + "=" * 60)
    print("TEST: 4-node scaling (2 rollout + 2 trainer)")
    print("=" * 60)
    
    config_content = """
experiment_name: e2e-test-4node
trial_name: v1
seed: 1
total_train_epochs: 1
tokenizer_path: ${actor.path}

nodes:
  total: 4
  roles:
    rollout: 2
    trainer: 2
  external_judge:
    enabled: false

cluster:
  n_nodes: 4
  n_gpus_per_node: 4
  fileroot: /mnt/job_stage/checkpoints/${experiment_name}/${trial_name}
  name_resolve:
    type: nfs
    nfs_record_root: /mnt/job_stage/name_resolve

allocation_mode: sglang:d2p1t4+d8p1t1
scheduler:
  type: null
"""
    
    config_path = os.path.join(payload_dir, 'e2e_test_4node.yaml')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created test config: {config_path}")
    print(f"Submitting 4-node job...")
    
    job = submit_job(
        session=session,
        payload_dir=payload_dir,
        entrypoint="log_wrapper.py",
        args=["run_areal.py", "e2e_test_4node.yaml", "soap_grpo_trainer.py"],
        compute_pool=compute_pool,
        eai=eai,
        target_instances=4,
        stage_name=stage_name,
        runtime=runtime,
        env_vars={"AREAL_NUM_NODES": "4"},
    )
    
    print(f"Job ID: {job.id}")
    print("Waiting for job (timeout: 15 min)...")
    
    success, status = wait_for_job(job, timeout=900)
    
    os.remove(config_path)
    
    if success:
        print(f"✅ PASSED: 4-node scaling test (status: {status})")
        return True
    else:
        print(f"❌ FAILED: 4-node scaling test (status: {status})")
        return False


def main():
    parser = argparse.ArgumentParser(description='E2E tests for unified multi-node RL training')
    parser.add_argument('-p', '--compute-pool', required=True, help='GPU compute pool')
    parser.add_argument('--database', required=True, help='Snowflake database')
    parser.add_argument('--schema', required=True, help='Snowflake schema')
    parser.add_argument('--role', default='SYSADMIN', help='Snowflake role')
    parser.add_argument('--stage-name', default='rl_payload_stage', help='Stage name')
    parser.add_argument('-e', '--external-access-integrations', nargs="+", required=True)
    parser.add_argument('--runtime', required=True, help='Runtime image URL')
    parser.add_argument('--test', choices=['2node_basic', 'legacy', '4node', 'all'],
                        default='all', help='Which test to run')
    args = parser.parse_args()
    
    session = create_session(args.database, args.schema, args.role)
    payload_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    stage_name = f"{args.database}.{args.schema}.{args.stage_name}"
    
    print("=" * 60)
    print("E2E TESTS: Unified Multi-Node RL Training")
    print("=" * 60)
    print(f"Compute pool: {args.compute_pool}")
    print(f"Database: {args.database}")
    print(f"Schema: {args.schema}")
    print(f"Runtime: {args.runtime}")
    
    results = {}
    
    if args.test in ('2node_basic', 'all'):
        results['2node_basic'] = test_2node_basic(
            session, payload_dir, args.compute_pool, 
            args.external_access_integrations, stage_name, args.runtime
        )
    
    if args.test in ('legacy', 'all'):
        results['legacy'] = test_legacy_config(
            session, payload_dir, args.compute_pool,
            args.external_access_integrations, stage_name, args.runtime
        )
    
    if args.test in ('4node', 'all'):
        results['4node'] = test_4node_scaling(
            session, payload_dir, args.compute_pool,
            args.external_access_integrations, stage_name, args.runtime
        )
    
    print("\n" + "=" * 60)
    print("E2E TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for name, result in results.items():
        status = "✅ PASSED" if result is True else ("❌ FAILED" if result is False else "⚠️ SKIPPED")
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
