"""Centralized configuration for all experiments."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────
    dataset: str = "cifar100"          # "cifar10" or "cifar100"
    data_dir: str = "./data"
    num_workers: int = 4

    # ── Model ────────────────────────────────────────────────────────────
    architecture: str = "resnet18"     # resnet18/34/50, efficientnet_b0/b1/b2
    num_classes: int = 100

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_milestones: List[int] = field(default_factory=lambda: [60, 120, 160])
    lr_gamma: float = 0.2

    # ── Knowledge Distillation ───────────────────────────────────────────
    temperature: float = 3.0
    alpha: float = 0.5                 # weight for hard-label CE loss

    # ── Born-Again Networks ──────────────────────────────────────────────
    ban_generations: int = 3           # total generations (including gen-1)

    # ── Layer-wise Self-Distillation ─────────────────────────────────────
    # Weights for auxiliary classifier KD losses (shallow → deep order)
    self_distill_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.7, 0.3]
    )

    # ── Logging / Checkpointing ──────────────────────────────────────────
    results_dir: str = "./results"
    plots_dir: str = "./plots"
    log_interval: int = 50            # print every N batches
    save_checkpoints: bool = True
    seed: int = 42
