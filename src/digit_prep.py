import numpy as np
import torch
from PIL import Image

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def center_of_mass(a: np.ndarray) -> tuple[float, float]:
    total = a.sum()
    if total == 0:
        return (a.shape[0] - 1) / 2, (a.shape[1] - 1) / 2
    rows = np.arange(a.shape[0])[:, None]
    cols = np.arange(a.shape[1])[None, :]
    return float((a * rows).sum() / total), float((a * cols).sum() / total)


def to_mnist_frame(img: Image.Image, center_by_mass: bool = True) -> Image.Image:
    a = np.asarray(img.convert("L"), dtype=np.float32)

    ys, xs = np.nonzero(a > 10)
    if len(ys) == 0:
        return Image.new("L", (28, 28), 0)  # blank canvas
    digit = Image.fromarray(a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1].astype(np.uint8))

    w, h = digit.size
    scale = 20.0 / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    digit = digit.resize(new_size, Image.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    if center_by_mass:
        cy, cx = center_of_mass(np.asarray(digit, dtype=np.float32))
        left = int(round(14 - cx))
        top = int(round(14 - cy))
    else:
        left = (28 - new_size[0]) // 2
        top = (28 - new_size[1]) // 2
    left = max(0, min(28 - new_size[0], left))
    top = max(0, min(28 - new_size[1], top))
    canvas.paste(digit, (left, top))
    return canvas


def to_tensor(frame28: Image.Image) -> torch.Tensor:
    """28x28 image -> normalized (1, 1, 28, 28) tensor ready for the model."""
    a = np.asarray(frame28, dtype=np.float32) / 255.0
    a = (a - MNIST_MEAN) / MNIST_STD
    return torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
