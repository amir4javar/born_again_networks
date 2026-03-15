"""Accuracy computation and logging helpers."""

import os
import json
import logging
from datetime import datetime

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor,
             topk=(1,)) -> list:
    """Compute top-k accuracy for the given k values."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


class AverageMeter:
    """Track a running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Create a logger that writes to console and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def save_results(results: dict, filepath: str):
    """Append results dict (with timestamp) to a JSON-lines file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results["timestamp"] = datetime.now().isoformat()
    with open(filepath, "a") as f:
        f.write(json.dumps(results) + "\n")
