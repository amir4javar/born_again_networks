"""Loss functions for knowledge distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """Knowledge-distillation loss (KL divergence on softened logits).

    KD_loss = T² · KL( student_soft ‖ teacher_soft )

    Parameters
    ----------
    temperature : float
        Softmax temperature.  Higher → softer distributions → more dark
        knowledge transferred.
    """

    def __init__(self, temperature: float = 3.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor) -> torch.Tensor:
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
        loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        return loss * (self.T ** 2)


class CombinedKDLoss(nn.Module):
    """Combined hard-label CE + soft-label KD loss.

    L = α · CE(student, y) + (1 − α) · KD(student, teacher)

    Parameters
    ----------
    temperature : float
    alpha : float
        Weight on the hard-label cross-entropy term.
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kd = KDLoss(temperature)

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(student_logits, targets)
        kd_loss = self.kd(student_logits, teacher_logits)
        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss


class SelfDistillLoss(nn.Module):
    """Loss for layer-wise self-distillation (BYOT).

    The main (deepest) classifier is trained with standard CE.
    Each auxiliary classifier is trained with:
        CE(aux, y) + weight_i · KD(aux, main_logits)

    Parameters
    ----------
    temperature : float
    alpha : float
        Balance between CE and KD for each auxiliary head.
    aux_weights : list[float]
        Per-auxiliary-head loss multipliers (shallow → deep order).
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5,
                 aux_weights: list = None):
        super().__init__()
        self.aux_weights = aux_weights or [1.0, 0.7, 0.3]
        self.ce = nn.CrossEntropyLoss()
        self.kd = KDLoss(temperature)
        self.alpha = alpha

    def forward(self, main_logits: torch.Tensor,
                aux_logits_list: list,
                targets: torch.Tensor) -> tuple:
        # Main classifier loss — standard CE
        main_loss = self.ce(main_logits, targets)

        # Detach teacher logits so gradients don't flow back through
        # the main head when computing auxiliary losses.
        teacher_logits = main_logits.detach()

        aux_loss = torch.tensor(0.0, device=main_logits.device)
        for i, aux_logits in enumerate(aux_logits_list):
            ce_part = self.ce(aux_logits, targets)
            kd_part = self.kd(aux_logits, teacher_logits)
            head_loss = self.alpha * ce_part + (1 - self.alpha) * kd_part
            aux_loss = aux_loss + self.aux_weights[i] * head_loss

        total_loss = main_loss + aux_loss
        return total_loss, main_loss, aux_loss
