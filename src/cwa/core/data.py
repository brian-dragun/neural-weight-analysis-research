"""Basic data handling utilities."""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Any
import logging

logger = logging.getLogger(__name__)


class SimpleTextDataset(Dataset):
    """Simple text dataset for basic testing."""

    def __init__(self, texts: List[str], tokenizer: Any, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "text": text
        }


def create_sample_data(num_samples: int = 10) -> List[str]:
    """Create sample text data for testing."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require careful analysis.",
        "Critical weights can significantly impact model performance.",
        "Security analysis is important for robust AI systems.",
        "Transformer models have revolutionized natural language processing.",
        "Adversarial attacks pose threats to deep learning models.",
        "Fault tolerance is crucial for deployment in harsh environments.",
        "Weight sensitivity analysis helps understand model behavior.",
        "Defense mechanisms can protect against various attacks.",
        "Research in AI security continues to advance rapidly."
    ]

    # Repeat and slice to get desired number of samples
    samples = (sample_texts * ((num_samples // len(sample_texts)) + 1))[:num_samples]
    logger.info(f"Created {len(samples)} sample texts for analysis")
    return samples


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    # Find the maximum length in the batch
    max_len = max(len(item["input_ids"]) for item in batch)

    # Pad all sequences to the same length
    input_ids = []
    attention_masks = []
    texts = []

    for item in batch:
        input_id = item["input_ids"]
        attention_mask = item["attention_mask"]

        # Pad sequences
        pad_length = max_len - len(input_id)
        if pad_length > 0:
            # Use the tokenizer's pad_token_id if available, otherwise use 0
            pad_token_id = getattr(item.get('tokenizer', None), 'pad_token_id', 0) or 0
            input_id = torch.cat([input_id, torch.full((pad_length,), pad_token_id, dtype=input_id.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=attention_mask.dtype)])

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        texts.append(item["text"])

    batch_tensors = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "text": texts
    }

    # Move tensors to CUDA if available
    if torch.cuda.is_available():
        for key in ["input_ids", "attention_mask"]:
            if key in batch_tensors and torch.is_tensor(batch_tensors[key]):
                batch_tensors[key] = batch_tensors[key].cuda()

    return batch_tensors


def create_data_loader(
    texts: List[str],
    tokenizer: Any,
    batch_size: int = 2,
    max_length: int = 512
) -> DataLoader:
    """Create data loader for analysis."""
    dataset = SimpleTextDataset(texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)