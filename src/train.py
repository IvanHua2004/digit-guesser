import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import get_dataloaders, get_device
from model import SimpleCNN, count_parameters

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "mnist_cnn.pt"
HISTORY_PATH = ROOT / "history.json"      # per-epoch metrics, plotted by draw.py


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_every=100):
    model.train()                       # dropout on
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % log_every == 0:
            seen = batch_idx * len(data)
            total = len(loader.dataset)
            print(f"  epoch {epoch}  [{seen:>5}/{total}]  loss {loss.item():.4f}")

    return running_loss / len(loader)


@torch.no_grad()                        # no graph needed for evaluation
def evaluate(model, loader, criterion, device):
    model.eval()                        # dropout off
    total_loss = 0.0
    correct = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item() * len(data)

        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    model = SimpleCNN().to(device)
    print(f"parameters: {count_parameters(model):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start

        print(
            f"epoch {epoch}: train loss {train_loss:.4f} | "
            f"test loss {test_loss:.4f} | test acc {test_acc:.2f}% | {elapsed:.1f}s\n"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "test_loss": round(test_loss, 5),
            "test_acc": round(test_acc, 3),
        })
        # written every epoch, so an interrupted run still leaves usable curves
        HISTORY_PATH.write_text(json.dumps(history, indent=2))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"best test accuracy: {best_acc:.2f}%")
    print(f"saved weights to {CHECKPOINT_PATH}")
    print(f"saved history to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
