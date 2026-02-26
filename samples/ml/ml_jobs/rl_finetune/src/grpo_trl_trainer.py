"""
GRPO (Group Relative Policy Optimization) Training with TRL
Multi-node RL finetuning on Snowflake SPCS
"""
import os
import json
import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer


def load_training_data(data_path: str) -> Dataset:
    """Load training data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    prompts = []
    for item in data:
        messages = item.get('messages', [])
        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
        prompts.append({"prompt": user_msg})
    
    return Dataset.from_list(prompts)


def create_reward_function(tokenizer):
    """Create a simple reward function based on response quality."""
    def reward_fn(completions: list[str], prompts: list[str] = None) -> list[float]:
        rewards = []
        for completion in completions:
            score = 0.0
            if len(completion) > 10:
                score += 0.2
            if len(completion) > 50:
                score += 0.2
            if completion.strip().endswith(('.', '!', '?')):
                score += 0.1
            if 'sorry' not in completion.lower() and 'cannot' not in completion.lower():
                score += 0.2
            if not any(word in completion.lower() for word in ['error', 'fail', 'unable']):
                score += 0.3
            rewards.append(score)
        return rewards
    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='grpo_lora_config.yaml')
    parser.add_argument('--train_data', type=str, default='synthetic_train_data.json')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--output_dir', type=str, default='./grpo_output')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_generations', type=int, default=4)
    parser.add_argument('--max_prompt_length', type=int, default=256)
    parser.add_argument('--max_completion_length', type=int, default=256)
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print(f"Loading training data from: {args.train_data}")
    train_dataset = load_training_data(args.train_data)
    print(f"Loaded {len(train_dataset)} training examples")
    
    reward_fn = create_reward_function(tokenizer)
    
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )
    
    print("Starting GRPO training...")
    trainer.train()
    
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
