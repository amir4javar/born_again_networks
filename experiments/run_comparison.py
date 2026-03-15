"""
Full comparison experiment: Baseline vs BAN vs Self-Distillation.

Trains all three approaches on the same dataset and architecture, then
produces a summary table and comparison plots.
"""

import sys
import os
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import Config
from models.resnet import build_resnet
from models.resnet_self_distill import build_self_distill_resnet
from utils.data import get_cifar_loaders
from utils.train import (train_standard, train_ban_generation,
                          train_self_distill)
from utils.metrics import get_logger, save_results


def plot_training_curves(histories: dict, save_dir: str):
    """Plot test accuracy curves for all methods on a single figure."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Test accuracy ────────────────────────────────────────────────────
    for label, hist in histories.items():
        key = "test_acc_main" if "test_acc_main" in hist else "test_acc"
        ax1.plot(hist[key], label=label)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Test Accuracy Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Training loss ────────────────────────────────────────────────────
    for label, hist in histories.items():
        ax2.plot(hist["train_loss"], label=label)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Training Loss Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_ban_generations(gen_accs: dict, save_dir: str):
    """Bar chart of accuracy per BAN generation."""
    os.makedirs(save_dir, exist_ok=True)

    gens = sorted(gen_accs.keys())
    accs = [gen_accs[g] for g in gens]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(gens, accs, color=["#4C72B0", "#55A868", "#C44E52",
                                      "#8172B2", "#CCB974"][:len(gens)])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("Born-Again Networks: Accuracy per Generation")
    ax.set_ylim(min(accs) - 2, max(accs) + 2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.2f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "ban_generations.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_self_distill_heads(main_acc, aux_accs, save_dir):
    """Bar chart comparing main vs auxiliary head accuracies."""
    os.makedirs(save_dir, exist_ok=True)

    labels = [f"Aux {i+1}" for i in range(len(aux_accs))] + ["Main"]
    accs = list(aux_accs) + [main_acc]
    colors = ["#8FAADC"] * len(aux_accs) + ["#4472C4"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, accs, color=colors)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Self-Distillation: Main vs Auxiliary Head Accuracy")
    ax.set_ylim(min(accs) - 5, max(accs) + 3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.2f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "self_distill_heads.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    logger = get_logger("Comparison",
                        os.path.join(cfg.results_dir, "comparison.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    train_loader, test_loader = get_cifar_loaders(
        dataset=cfg.dataset, data_dir=cfg.data_dir,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    all_histories = {}
    summary = {}

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Baseline (standard CE training)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Baseline")
    logger.info("=" * 60)
    baseline_model = build_resnet(cfg.architecture, cfg.num_classes)
    baseline_model, baseline_acc, baseline_hist = train_standard(
        baseline_model, train_loader, test_loader, cfg, device, tag="baseline"
    )
    all_histories["Baseline"] = baseline_hist
    summary["Baseline"] = baseline_acc

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Born-Again Networks (BAN)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Born-Again Networks")
    logger.info("=" * 60)

    # Gen 1 = baseline (reuse if same architecture; train fresh for fairness)
    teacher = build_resnet(cfg.architecture, cfg.num_classes)
    teacher, gen1_acc, gen1_hist = train_standard(
        teacher, train_loader, test_loader, cfg, device, tag="ban_gen1"
    )
    ban_gen_accs = {"Gen 1": gen1_acc}
    ban_last_hist = gen1_hist

    for gen in range(2, cfg.ban_generations + 1):
        student = build_resnet(cfg.architecture, cfg.num_classes)
        student, gen_acc, gen_hist = train_ban_generation(
            student, teacher, train_loader, test_loader, cfg, device,
            tag=f"ban_gen{gen}",
        )
        ban_gen_accs[f"Gen {gen}"] = gen_acc
        ban_last_hist = gen_hist
        teacher = student

    all_histories[f"BAN Gen {cfg.ban_generations}"] = ban_last_hist
    summary["BAN (best gen)"] = max(ban_gen_accs.values())

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Self-Distillation (BYOT)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Self-Distillation (BYOT)")
    logger.info("=" * 60)
    sd_model = build_self_distill_resnet(cfg.architecture, cfg.num_classes)
    sd_model, sd_acc, sd_hist = train_self_distill(
        sd_model, train_loader, test_loader, cfg, device, tag="self_distill"
    )
    all_histories["Self-Distill"] = sd_hist
    summary["Self-Distill (main)"] = sd_acc

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    for method, acc in summary.items():
        logger.info(f"  {method:25s}: {acc:.2f}%")

    # ── Plots ────────────────────────────────────────────────────────────
    p1 = plot_training_curves(all_histories, cfg.plots_dir)
    logger.info(f"Saved training curves → {p1}")

    p2 = plot_ban_generations(ban_gen_accs, cfg.plots_dir)
    logger.info(f"Saved BAN generations → {p2}")

    if sd_hist["test_acc_aux"]:
        final_aux = sd_hist["test_acc_aux"][-1]
        p3 = plot_self_distill_heads(sd_acc, final_aux, cfg.plots_dir)
        logger.info(f"Saved self-distill heads → {p3}")

    # ── Persist ──────────────────────────────────────────────────────────
    save_results(
        {"experiment": "comparison", "architecture": cfg.architecture,
         "dataset": cfg.dataset, "summary": summary,
         "ban_generations": ban_gen_accs},
        os.path.join(cfg.results_dir, "comparison_results.jsonl"),
    )

    return summary


if __name__ == "__main__":
    main()
