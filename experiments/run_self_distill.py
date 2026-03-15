"""
Layer-wise Self-Distillation (BYOT) experiment.

1. Build a ResNetSelfDistill model with auxiliary classifier heads.
2. Train with the BYOT loss: the deepest head is the teacher, shallower
   heads are students.  All heads are trained jointly in a single pass.
3. Report accuracy for the main head and each auxiliary head.
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import Config
from models.resnet_self_distill import build_self_distill_resnet
from utils.data import get_cifar_loaders
from utils.train import train_self_distill
from utils.metrics import get_logger, save_results


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    logger = get_logger("BYOT",
                        os.path.join(cfg.results_dir, "self_distill.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    train_loader, test_loader = get_cifar_loaders(
        dataset=cfg.dataset, data_dir=cfg.data_dir,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    logger.info("=" * 60)
    logger.info("Layer-wise Self-Distillation (BYOT)")
    logger.info("=" * 60)

    model = build_self_distill_resnet(cfg.architecture, cfg.num_classes)
    model, best_acc, history = train_self_distill(
        model, train_loader, test_loader, cfg, device, tag="self_distill"
    )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Self-Distillation Results")
    logger.info("=" * 60)
    logger.info(f"  Main head accuracy: {best_acc:.2f}%")
    if history["test_acc_aux"]:
        final_aux = history["test_acc_aux"][-1]
        for i, acc in enumerate(final_aux):
            logger.info(f"  Aux head {i+1} accuracy: {acc:.2f}%")

    save_results(
        {"experiment": "self_distill", "architecture": cfg.architecture,
         "dataset": cfg.dataset, "main_acc": best_acc,
         "aux_accs": history["test_acc_aux"][-1] if history["test_acc_aux"] else []},
        os.path.join(cfg.results_dir, "self_distill_results.jsonl"),
    )

    return model, best_acc, history


if __name__ == "__main__":
    main()
