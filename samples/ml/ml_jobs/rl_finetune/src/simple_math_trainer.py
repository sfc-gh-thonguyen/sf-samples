"""Simple math trainer with inline dataset for fast SPCS testing."""
import sys
import re
from typing import List, Dict, Any

try:
    from areal import PPOTrainer
except ImportError:
    from areal.experimental.trainer import PPOTrainer

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer


class SimpleMathDataset:
    """Inline math dataset - no external downloads needed."""
    
    def __init__(self, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.split = split
        
        # Tiny inline dataset
        self.data = [
            {"prompt": "What is 2 + 3?", "target": "5"},
            {"prompt": "What is 10 - 4?", "target": "6"},
            {"prompt": "What is 5 * 6?", "target": "30"},
            {"prompt": "What is 20 / 4?", "target": "5"},
            {"prompt": "What is 15 + 27?", "target": "42"},
            {"prompt": "What is 100 - 37?", "target": "63"},
            {"prompt": "What is 8 * 9?", "target": "72"},
            {"prompt": "What is 144 / 12?", "target": "12"},
            {"prompt": "What is 33 + 44?", "target": "77"},
            {"prompt": "What is 99 - 11?", "target": "88"},
        ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"Solve this math problem. Give only the final number.\n\nQuestion: {item['prompt']}\nAnswer:"
        
        return {
            "id": f"math_{idx}",
            "prompt": prompt,
            "target": item["target"],
        }


def simple_math_reward_fn(
    responses: List[str],
    prompts: List[str],
    extras: Dict[str, Any],
) -> List[float]:
    """Simple reward function: 1.0 if answer matches, 0.0 otherwise."""
    rewards = []
    targets = extras.get("target", [])
    
    for i, response in enumerate(responses):
        target = targets[i] if i < len(targets) else ""
        
        # Extract numbers from response
        numbers = re.findall(r'\d+', response)
        
        # Check if target appears in extracted numbers
        if target in numbers:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards


def main(args):
    print("[SimpleMathTrainer] Starting with inline dataset...")
    
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    
    print(f"[SimpleMathTrainer] Using tokenizer: {config.tokenizer_path}")
    
    # Use inline dataset
    train_dataset = SimpleMathDataset(tokenizer, split="train")
    valid_dataset = SimpleMathDataset(tokenizer, split="test")
    
    print(f"[SimpleMathTrainer] Train dataset: {len(train_dataset)} samples")
    print(f"[SimpleMathTrainer] Valid dataset: {len(valid_dataset)} samples")
    
    # Workflow config
    workflow_kwargs = dict(
        reward_fn="simple_math_trainer.simple_math_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)
    
    print("[SimpleMathTrainer] Starting PPOTrainer...")
    
    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        print("[SimpleMathTrainer] Trainer initialized, starting training...")
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )
    
    print("[SimpleMathTrainer] Training complete!")


if __name__ == "__main__":
    main(sys.argv[1:])
