"""
Born-Again Networks (BAN) experiment.

1. Train Generation 1 from scratch (standard CE).
2. For each subsequent generation, use the previous generation as the teacher
   and train a fresh model of the same architecture with combined CE + KD loss.
3. Report accuracy progression across generations.
"""

import sys
import os
import json

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import Config
from models.builder import build_model
from utils.data import get_cifar_loaders
from utils.train import train_standard, train_ban_generation
from utils.metrics import get_logger, save_results


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    logger = get_logger("BAN", os.path.join(cfg.results_dir, "ban.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    train_loader, test_loader = get_cifar_loaders(
        dataset=cfg.dataset, data_dir=cfg.data_dir,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    results = {}

    # ── Generation 1: standard training ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("Generation 1 — Standard training")
    logger.info("=" * 60)
    gen1_model = build_model(cfg.architecture, cfg.num_classes)
    gen1_model, gen1_acc, gen1_hist = train_standard(
        gen1_model, train_loader, test_loader, cfg, device, tag="ban_gen1"
    )
    results["gen1"] = {"best_acc": gen1_acc, "history": gen1_hist}

    teacher = gen1_model

    # ── Subsequent generations: BAN distillation ─────────────────────────
    for gen in range(2, cfg.ban_generations + 1):
        logger.info("=" * 60)
        logger.info(f"Generation {gen} — Born-Again from Gen {gen - 1}")
        logger.info("=" * 60)

        student = build_model(cfg.architecture, cfg.num_classes)
        student, gen_acc, gen_hist = train_ban_generation(
            student, teacher, train_loader, test_loader, cfg, device,
            tag=f"ban_gen{gen}",
        )
        results[f"gen{gen}"] = {"best_acc": gen_acc, "history": gen_hist}
        teacher = student  # next generation's teacher

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BAN Results Summary")
    logger.info("=" * 60)
    for gen_key in sorted(results.keys()):
        logger.info(f"  {gen_key}: {results[gen_key]['best_acc']:.2f}%")

    # Save full results
    save_results(
        {"experiment": "BAN", "architecture": cfg.architecture,
         "dataset": cfg.dataset, "generations": cfg.ban_generations,
         "accuracies": {k: v["best_acc"] for k, v in results.items()}},
        os.path.join(cfg.results_dir, "ban_results.jsonl"),
    )

    return results


if __name__ == "__main__":
    main()
