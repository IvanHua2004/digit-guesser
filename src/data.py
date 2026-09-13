"""MNIST loading: dataset, normalisation, batching, device selection."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Where the raw MNIST files get downloaded (first run only, ~11 MB).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),  # PIL image -> float tensor of shape (1, 28, 28), values in [0, 1]
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ]
)


def get_dataloaders(batch_size: int = 64, num_workers: int = 0):
    train_set = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=TRANSFORM
    )
    test_set = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=TRANSFORM
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=1000, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"batch of images: {tuple(images.shape)}  dtype={images.dtype}")
    print(f"batch of labels: {tuple(labels.shape)}  first 8 = {labels[:8].tolist()}")
    print(f"pixel range after normalize: [{images.min():.2f}, {images.max():.2f}]")
    print(f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")
    print(f"device: {get_device()}")
