"""Data loading utilities for CIFAR-10 / CIFAR-100."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar_loaders(dataset: str = "cifar100",
                      data_dir: str = "./data",
                      batch_size: int = 128,
                      num_workers: int = 4):
    """Return (train_loader, test_loader) for the chosen CIFAR variant.

    Augmentations follow the standard protocol used in most ResNet-on-CIFAR
    papers: random crop with padding-4 + horizontal flip for training;
    no augmentation for testing.
    """
    # Normalization statistics
    if dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        Dataset = datasets.CIFAR100
    elif dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        Dataset = datasets.CIFAR10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = Dataset(root=data_dir, train=True, download=True,
                        transform=train_transform)
    test_set = Dataset(root=data_dir, train=False, download=True,
                       transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
