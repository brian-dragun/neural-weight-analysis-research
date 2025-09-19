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


def create_data_loader(
    texts: List[str],
    tokenizer: Any,
    batch_size: int = 2,
    max_length: int = 512
) -> DataLoader:
    """Create data loader for analysis."""
    dataset = SimpleTextDataset(texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)