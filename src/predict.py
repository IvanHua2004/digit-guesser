import argparse

import matplotlib

matplotlib.use("Agg")  # write PNG files instead of opening a window
import matplotlib.pyplot as plt
import torch

from data import MNIST_MEAN, MNIST_STD, get_dataloaders, get_device
from model import SimpleCNN
from train import CHECKPOINT_PATH


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    """Undo the Normalize transform so the digit looks right on screen."""
    return img * MNIST_STD + MNIST_MEAN


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistakes", action="store_true", help="show only errors")
    parser.add_argument("--out", default="predictions.png")
    args = parser.parse_args()

    device = get_device()
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    _, test_loader = get_dataloaders()

    images, labels, preds, confs = [], [], [], []
    for data, target in test_loader:
        output = model(data.to(device))
        prob = torch.softmax(output, dim=1)  # softmax HERE, for display only
        conf, pred = prob.max(dim=1)
        pred, conf = pred.cpu(), conf.cpu()

        keep = (pred != target) if args.mistakes else torch.ones_like(target, dtype=bool)
        images.append(data[keep])
        labels.append(target[keep])
        preds.append(pred[keep])
        confs.append(conf[keep])
        if sum(len(x) for x in images) >= 16:
            break

    images = torch.cat(images)[:16]
    labels = torch.cat(labels)[:16]
    preds = torch.cat(preds)[:16]
    confs = torch.cat(confs)[:16]

    if len(images) == 0:
        print("No mistakes found in the batches scanned. Impressive.")
        return

    cols = 4
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.3 * rows))
    for ax, img, true, pred, conf in zip(
        axes.flatten(), images, labels, preds, confs
    ):
        ax.imshow(unnormalize(img).squeeze(), cmap="gray")
        ok = pred.item() == true.item()
        ax.set_title(
            f"pred {pred.item()} ({conf.item():.0%})\ntrue {true.item()}",
            fontsize=9,
            color="black" if ok else "crimson",
        )
        ax.axis("off")
    for ax in axes.flatten()[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
