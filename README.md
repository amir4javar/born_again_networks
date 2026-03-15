# Self-Distillation Research

Experiments comparing knowledge distillation strategies on image classification:
**Baseline** vs **Born-Again Networks (BAN)** vs **Layer-wise Self-Distillation (BYOT)**.

## Methods

- **Baseline** — standard cross-entropy training.
- **Born-Again Networks (BAN)** — sequential student generations, each trained with KD from the previous model.
- **Self-Distillation (BYOT)** — single model with auxiliary classifiers at intermediate layers; deep layers distill into shallow ones during training.

## Results

> Dataset: CIFAR-100 | Architecture: ResNet-18 | Epochs: 200 | Seed: 42

### Final Accuracy

| Method | Test Accuracy | vs Baseline |
|---|---|---|
| Baseline | 78.01% | — |
| BAN (best gen) | **79.42%** | +1.41% |
| Self-Distill (main) | 79.04% | +1.03% |

### BAN Generation Breakdown

| Generation | Test Accuracy |
|---|---|
| Gen 1 | 78.34% |
| Gen 2 | 79.34% |
| Gen 3 | **79.42%** |

### Plots

Training curves, BAN generation progression, and self-distillation head comparison are saved under [plots/](plots/).

## Project Structure

```
.
├── configs/
│   └── config.py               # All hyperparameters
├── experiments/
│   ├── run_ban.py              # BAN-only experiment
│   ├── run_self_distill.py     # Self-distillation-only experiment
│   └── run_comparison.py       # Full comparison (all three methods)
├── models/
│   ├── resnet.py / efficientnet.py
│   └── resnet_self_distill.py / efficientnet_self_distill.py
├── utils/
│   ├── data.py                 # CIFAR-10/100 dataloaders
│   ├── losses.py               # KD loss implementations
│   ├── train.py                # Training loops
│   └── metrics.py              # Logging and result saving
├── plots/                      # Generated figures
└── results/                    # Logs, checkpoints, JSONL results
```

## Usage

```bash
# Run full comparison
python experiments/run_comparison.py

# Run individual experiments
python experiments/run_ban.py
python experiments/run_self_distill.py
```

Key hyperparameters are in [configs/config.py](configs/config.py) (architecture, dataset, KD temperature, BAN generations, etc.).
