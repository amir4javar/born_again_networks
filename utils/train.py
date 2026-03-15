"""Training loops for standard training, BAN, and self-distillation."""

import os
import copy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from .losses import CombinedKDLoss, SelfDistillLoss
from .metrics import accuracy, AverageMeter, get_logger


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_optimizer(model, cfg):
    return torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                           momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay)


def _make_scheduler(optimizer, cfg):
    return MultiStepLR(optimizer, milestones=cfg.lr_milestones,
                       gamma=cfg.lr_gamma)


def _save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ── Standard training ────────────────────────────────────────────────────────

def train_standard(model, train_loader, test_loader, cfg, device,
                   tag="baseline"):
    """Train a model from scratch with cross-entropy only."""
    logger = get_logger(tag, os.path.join(cfg.results_dir, f"{tag}.log"))
    model = model.to(device)
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = {"train_loss": [], "test_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        # ── train ──
        model.train()
        losses = AverageMeter()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), inputs.size(0))

            if (batch_idx + 1) % cfg.log_interval == 0:
                logger.info(f"Epoch [{epoch}/{cfg.epochs}] "
                            f"Batch [{batch_idx+1}/{len(train_loader)}] "
                            f"Loss: {losses.avg:.4f}")

        scheduler.step()

        # ── evaluate ──
        test_acc = evaluate(model, test_loader, device)
        history["train_loss"].append(losses.avg)
        history["test_acc"].append(test_acc)
        logger.info(f"Epoch [{epoch}/{cfg.epochs}] "
                    f"Test Acc: {test_acc:.2f}%  (best: {best_acc:.2f}%)")

        if test_acc > best_acc:
            best_acc = test_acc
            if cfg.save_checkpoints:
                _save_checkpoint(
                    {"epoch": epoch, "state_dict": model.state_dict(),
                     "best_acc": best_acc},
                    os.path.join(cfg.results_dir, f"{tag}_best.pth"),
                )

    logger.info(f"Best test accuracy: {best_acc:.2f}%")
    return model, best_acc, history


# ── Born-Again Network (BAN) training ────────────────────────────────────────

def train_ban_generation(student, teacher, train_loader, test_loader, cfg,
                         device, tag="ban_gen"):
    """Train one BAN generation: student learns from teacher's soft labels."""
    logger = get_logger(tag, os.path.join(cfg.results_dir, f"{tag}.log"))
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    optimizer = _make_optimizer(student, cfg)
    scheduler = _make_scheduler(optimizer, cfg)
    criterion = CombinedKDLoss(temperature=cfg.temperature, alpha=cfg.alpha)

    best_acc = 0.0
    history = {"train_loss": [], "test_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        losses = AverageMeter()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)
            loss = criterion(student_logits, teacher_logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), inputs.size(0))

            if (batch_idx + 1) % cfg.log_interval == 0:
                logger.info(f"Epoch [{epoch}/{cfg.epochs}] "
                            f"Batch [{batch_idx+1}/{len(train_loader)}] "
                            f"Loss: {losses.avg:.4f}")

        scheduler.step()

        test_acc = evaluate(student, test_loader, device)
        history["train_loss"].append(losses.avg)
        history["test_acc"].append(test_acc)
        logger.info(f"Epoch [{epoch}/{cfg.epochs}] "
                    f"Test Acc: {test_acc:.2f}%  (best: {best_acc:.2f}%)")

        if test_acc > best_acc:
            best_acc = test_acc
            if cfg.save_checkpoints:
                _save_checkpoint(
                    {"epoch": epoch, "state_dict": student.state_dict(),
                     "best_acc": best_acc},
                    os.path.join(cfg.results_dir, f"{tag}_best.pth"),
                )

    logger.info(f"Best test accuracy: {best_acc:.2f}%")
    return student, best_acc, history


# ── Self-Distillation (BYOT) training ────────────────────────────────────────

def train_self_distill(model, train_loader, test_loader, cfg, device,
                       tag="self_distill"):
    """Train a ResNetSelfDistill model with auxiliary-head distillation."""
    logger = get_logger(tag, os.path.join(cfg.results_dir, f"{tag}.log"))
    model = model.to(device)
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)
    criterion = SelfDistillLoss(
        temperature=cfg.temperature,
        alpha=cfg.alpha,
        aux_weights=cfg.self_distill_weights,
    )

    best_acc = 0.0
    history = {"train_loss": [], "main_loss": [], "aux_loss": [],
               "test_acc_main": [], "test_acc_aux": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses_total = AverageMeter()
        losses_main = AverageMeter()
        losses_aux = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            main_logits, aux_logits_list = model(inputs)
            total_loss, main_loss, aux_loss = criterion(
                main_logits, aux_logits_list, targets
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses_total.update(total_loss.item(), inputs.size(0))
            losses_main.update(main_loss.item(), inputs.size(0))
            losses_aux.update(aux_loss.item(), inputs.size(0))

            if (batch_idx + 1) % cfg.log_interval == 0:
                logger.info(
                    f"Epoch [{epoch}/{cfg.epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Total: {losses_total.avg:.4f}  "
                    f"Main: {losses_main.avg:.4f}  "
                    f"Aux: {losses_aux.avg:.4f}"
                )

        scheduler.step()

        # Evaluate main and auxiliary heads
        test_acc_main, test_acc_aux = evaluate_self_distill(
            model, test_loader, device
        )
        history["train_loss"].append(losses_total.avg)
        history["main_loss"].append(losses_main.avg)
        history["aux_loss"].append(losses_aux.avg)
        history["test_acc_main"].append(test_acc_main)
        history["test_acc_aux"].append(test_acc_aux)

        logger.info(
            f"Epoch [{epoch}/{cfg.epochs}] "
            f"Main Acc: {test_acc_main:.2f}%  "
            f"Aux Accs: {[f'{a:.2f}%' for a in test_acc_aux]}  "
            f"(best: {best_acc:.2f}%)"
        )

        if test_acc_main > best_acc:
            best_acc = test_acc_main
            if cfg.save_checkpoints:
                _save_checkpoint(
                    {"epoch": epoch, "state_dict": model.state_dict(),
                     "best_acc": best_acc},
                    os.path.join(cfg.results_dir, f"{tag}_best.pth"),
                )

    logger.info(f"Best main-head test accuracy: {best_acc:.2f}%")
    return model, best_acc, history


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, test_loader, device):
    """Return top-1 test accuracy (%) for a standard model."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_self_distill(model, test_loader, device):
    """Return (main_acc, [aux1_acc, aux2_acc, aux3_acc]) for self-distill."""
    model.eval()
    total = 0
    correct_main = 0
    correct_aux = [0, 0, 0]

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        main_logits, aux_logits_list = model(inputs)

        _, pred_main = main_logits.max(1)
        correct_main += pred_main.eq(targets).sum().item()

        for i, aux_logits in enumerate(aux_logits_list):
            _, pred = aux_logits.max(1)
            correct_aux[i] += pred.eq(targets).sum().item()

        total += targets.size(0)

    main_acc = 100.0 * correct_main / total
    aux_accs = [100.0 * c / total for c in correct_aux]
    return main_acc, aux_accs
