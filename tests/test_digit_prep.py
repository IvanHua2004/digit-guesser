"""Tests for the drawing-to-MNIST preprocessing. These pass out of the box.

They exist so you can change the pipeline (try a different scale, drop the
centre-of-mass step) and immediately see what you broke.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digit_prep import center_of_mass, to_mnist_frame, to_tensor  # noqa: E402


def stroke(points, width=18, size=280):
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).line(points, fill=255, width=width)
    return img


def test_output_is_28x28():
    assert to_mnist_frame(stroke([(60, 40), (70, 200)])).size == (28, 28)


def test_blank_canvas_does_not_crash():
    frame = to_mnist_frame(Image.new("L", (280, 280), 0))
    assert np.asarray(frame).sum() == 0


def test_offcentre_digit_gets_centred():
    """A stroke drawn in the corner should end up in the middle."""
    frame = to_mnist_frame(stroke([(20, 20), (30, 120)]))
    row, col = center_of_mass(np.asarray(frame, dtype=np.float32))
    assert abs(row - 13.5) < 1.5 and abs(col - 13.5) < 1.5


def test_aspect_ratio_is_preserved():
    """A tall thin stroke must not be squashed into a square."""
    frame = np.asarray(to_mnist_frame(stroke([(140, 30), (140, 250)])), dtype=np.float32)
    ys, xs = np.nonzero(frame)
    assert (ys.max() - ys.min()) > 3 * (xs.max() - xs.min())


def test_scaled_to_20px_box():
    """MNIST digits fit in 20x20 inside the 28x28 frame."""
    frame = np.asarray(to_mnist_frame(stroke([(40, 40), (240, 240)])), dtype=np.float32)
    ys, xs = np.nonzero(frame)
    assert max(ys.max() - ys.min(), xs.max() - xs.min()) <= 20


def test_tensor_shape_and_normalization():
    t = to_tensor(to_mnist_frame(stroke([(60, 40), (70, 200)])))
    assert tuple(t.shape) == (1, 1, 28, 28)
    assert t.min() < 0, "background should be negative after normalizing"
