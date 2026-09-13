"""Architecture and training-step tests. No training required.

    pytest -q
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import SimpleCNN, count_parameters  # noqa: E402


class _FakeLoader:
    """Stands in for a DataLoader: yields one batch and knows its length."""

    def __init__(self, data, target):
        self.data, self.target = data, target
        self.dataset = range(len(data))

    def __iter__(self):
        return iter([(self.data, self.target)])

    def __len__(self):
        return 1


# ---------- architecture ----------

def test_layers_exist():
    m = SimpleCNN()
    for name in ["conv1", "conv2", "pool", "fc1", "dropout", "fc2"]:
        assert hasattr(m, name), f"SimpleCNN is missing self.{name}"


def test_conv_channels():
    m = SimpleCNN()
    assert m.conv1.in_channels == 1, "MNIST is grayscale: 1 input channel"
    assert m.conv1.out_channels == 32
    assert m.conv2.in_channels == 32, "conv2 must accept conv1's output channels"
    assert m.conv2.out_channels == 64


def test_fc_shapes():
    m = SimpleCNN()
    assert m.fc1.in_features == 64 * 7 * 7, (
        "after two 2x2 pools, 28x28 -> 7x7, with 64 channels"
    )
    assert m.fc1.out_features == 128
    assert m.fc2.out_features == 10, "ten digits, ten output scores"


# ---------- forward pass ----------

def test_output_shape():
    m = SimpleCNN()
    out = m(torch.randn(8, 1, 28, 28))
    assert out.shape == (8, 10), f"expected (8, 10), got {tuple(out.shape)}"


def test_output_is_logits_not_probabilities():
    """If rows sum to 1, softmax was applied inside forward. Remove it."""
    m = SimpleCNN()
    m.eval()
    out = m(torch.randn(8, 1, 28, 28))
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones(8), atol=1e-3), (
        "forward() must return raw logits - CrossEntropyLoss applies softmax itself"
    )


def test_gradients_flow():
    """A backward pass should give every parameter a gradient."""
    m = SimpleCNN()
    loss = nn.CrossEntropyLoss()(m(torch.randn(4, 1, 28, 28)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"


def test_parameter_count_is_reasonable():
    n = count_parameters(SimpleCNN())
    assert 380_000 < n < 460_000, (
        f"got {n:,} parameters; expected ~421k. "
        "A big mismatch usually means a wrong fc1 input size."
    )


# ---------- training loop ----------

def test_can_overfit_one_batch():
    """The single best smoke test in deep learning.

    If a model cannot drive the loss to ~0 on eight examples it has seen
    200 times, something is broken in the training step - not the data,
    not the learning rate. Always run this before a long training job.
    """
    torch.manual_seed(0)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from train import train_one_epoch

    m = SimpleCNN()
    data = torch.randn(8, 1, 28, 28)
    target = torch.arange(8) % 10
    batch = _FakeLoader(data, target)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    first = None
    for i in range(200):
        loss = train_one_epoch(m, batch, crit, opt, torch.device("cpu"), 0, log_every=10**9)
        if first is None:
            first = loss
    assert loss < first, "loss did not decrease at all - check the ordering of the five training-step lines"
    assert loss < 0.05, f"could not overfit 8 examples (final loss {loss:.3f})"


def test_evaluate_counts_correctly():
    from train import evaluate

    class AlwaysSevens(nn.Module):
        def forward(self, x):
            out = torch.zeros(len(x), 10)
            out[:, 7] = 10.0
            return out

    data = torch.randn(10, 1, 28, 28)
    target = torch.tensor([7, 7, 7, 0, 1, 2, 3, 4, 5, 6])  # 3 correct out of 10

    _, acc = evaluate(
        AlwaysSevens(), _FakeLoader(data, target), nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert abs(acc - 30.0) < 1e-6, f"expected 30% accuracy, got {acc}"
