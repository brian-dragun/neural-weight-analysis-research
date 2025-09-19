"""Logging utilities."""

import logging
from pathlib import Path
from rich.logging import RichHandler


def setup_logging(output_dir: Path, level: int = logging.INFO):
    """Setup logging with both file and console output."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(output_dir / "experiment.log")
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Rich console handler
    console_handler = RichHandler(show_path=False)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger