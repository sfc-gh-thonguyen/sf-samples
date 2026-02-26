"""Dataset loading utilities for Medical SOAP RL training.

This module provides functions to load and process the custom medical SOAP
dataset for reinforcement learning with GRPO.
"""

from datasets import Dataset

from prompt_utils import SYSTEM_PROMPT, USER_PROMPT_PREFIX


def get_medical_soap_rl_dataset(
    path: str,
    split: str,
    tokenizer=None,
    max_length: int | None = None,
) -> Dataset:
    """Load and process the medical SOAP dataset for RL training.

    Args:
        path: Path to the JSON dataset file.
        split: Either "train" or "test" (validation).
        tokenizer: Optional tokenizer for length filtering.
        max_length: Maximum token length for filtering (requires tokenizer).

    Returns:
        Processed HuggingFace Dataset with 'messages' field and ground truth
        SOAP sections for reward computation.
    """
    dataset = Dataset.from_json(path)

    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    if split == "train":
        dataset = split_dataset["train"]
    else:
        dataset = split_dataset["test"]

    def process(sample):
        """Process a single sample into the required format."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{USER_PROMPT_PREFIX}\n\n{sample['dialogue']}",
            },
        ]

        return {
            "messages": messages,
            "dialogue": sample["dialogue"],
            "pred_S": sample["pred_S"],
            "pred_O": sample["pred_O"],
            "pred_A": sample["pred_A"],
            "pred_P": sample["pred_P"],
        }

    dataset = dataset.map(process)

    if max_length is not None and tokenizer is not None:

        def filter_length(sample):
            system_content = sample["messages"][0]["content"]
            user_content = sample["messages"][1]["content"]
            full_content = system_content + "\n" + user_content
            tokens = tokenizer.encode(full_content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
