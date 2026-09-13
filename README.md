# Digit Guesser

A convolutional network that reads handwritten digits, plus a drawing pad to test it
with your own handwriting.

Train it in under a minute, then draw a digit with your mouse and watch the model
guess — with a live view of the 28×28 image it actually receives.

![Test-set predictions with confidence, errors in red](predictions.png)

```
epoch 3: train loss 0.0405 | test loss 0.0287 | test acc 99.11%
```

Built with PyTorch. No framework beyond that, no pretrained weights — the network is
about 40 lines and trains from scratch on MNIST.

---

## Quick start

Requires Python 3.11+ and, optionally, an NVIDIA GPU.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS / Linux

# GPU: get the command for your CUDA version at pytorch.org/get-started/locally
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# CPU only:
# pip install torch torchvision

pip install -r requirements.txt
```

Then:

```bash
python src/data.py                  # downloads MNIST, ~11 MB, once
python src/train.py --epochs 3      # ~30s on a laptop GPU, a few minutes on CPU
python src/draw.py                  # the drawing pad
```

## The drawing pad

`draw.py` opens a canvas. Draw a digit and it predicts live as you go.

| | |
|---|---|
| left mouse | draw |
| right mouse | erase |
| `C` | clear |
| `M` | switch how your drawing gets centred |
| `Esc` | quit |

The middle panel shows the **28×28 the model actually sees**. It is worth watching
— nearly every prediction that looks wrong makes sense the moment you look at what
got fed in. On the right are the curves from your last training run: cross-entropy
loss for train and test, and test accuracy per epoch.

You can also review the model against the real test set:

```bash
python src/predict.py --mistakes    # writes predictions.png of what it got wrong
```

Most of the surviving errors are digits a person would hesitate over too.

## Why your own handwriting is harder

The model scores 99% on MNIST and noticeably worse on the pad. That gap is the
interesting part of this project, and it isn't the network's fault.

MNIST digits were never raw scans. Every one went through a pipeline: crop to the
digit's bounding box, scale so the longest side is 20px, then place it in a 28×28
field **positioned by centre of mass** — not by the centre of its bounding box.

That last step is the one that gets skipped. A `1` carries its weight off to one
side; a `7` is top-heavy. Centre those by bounding box and they land somewhere the
model has never seen a digit, and accuracy falls off a cliff.

`src/digit_prep.py` reproduces the real pipeline, and the `M` key switches the
centre-of-mass step off so you can watch the predictions degrade in real time.

The lesson generalises well past MNIST: **when a model works on the benchmark and
fails on your data, suspect preprocessing before architecture.**

## How the network works

New to convolutional networks? I kept notes while building this —
[Notes_CNN.pdf](Notes_CNN.pdf) covers kernels, padding, pooling and how the shapes
fall out, starting from nothing.

```
input                    1 × 28 × 28
conv1 → relu → pool     32 × 14 × 14     32 kernels, each 3×3
conv2 → relu → pool     64 ×  7 ×  7     64 kernels, each 3×3×32
flatten                       3136
fc1 → relu → dropout           128
fc2                             10       one score per digit
```

421,642 parameters. Notably, the two convolutional layers are only 18,816 of them —
under 5%. The dense layer that reads out their conclusions is 95% of the model.

`padding=1` keeps each conv at the same width and height (28 + 2 − 3 + 1 = 28), and
each 2×2 max-pool halves it, which is where the `7 × 7` comes from.

## Layout

```
src/
  data.py         MNIST loaders, normalisation, device selection
  model.py        the network
  train.py        training loop and evaluation
  predict.py      renders predictions and mistakes to a PNG
  digit_prep.py   drawing → MNIST-style 28×28
  draw.py         the drawing pad
tests/
  test_model.py       architecture, gradient flow, overfit-one-batch
  test_digit_prep.py  the preprocessing pipeline
```

```bash
pytest -q     # 15 tests
```

The one worth knowing about is `test_can_overfit_one_batch`. If a network can't
drive the loss to zero on eight examples it has seen two hundred times, the bug is
in the training step — not the data, not the learning rate. It's the fastest way to
tell a broken model from a badly tuned one.

## Things to try

- **Ablate.** Remove `conv2`. Remove dropout. Replace both convs with a plain
  `nn.Linear(784, 128)`. The gap between that and the CNN is the whole argument for
  convolutions, in one number.
- **Augment.** Add `transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))` to
  the training set and see what it does to the drawing pad.
- **Look at the kernels.** `model.conv1.weight` is a `(32, 1, 3, 3)` tensor. Plot
  those 32 grids as images — they start random and end up as edge and blob
  detectors, without anyone asking them to.
- **Watch it overfit.** Run `--epochs 20` and find the point where test loss starts
  climbing while train loss keeps falling.

## Notes

Trained weights are saved to `mnist_cnn.pt`, best test accuracy only, and per-epoch
metrics to `history.json`, which is what the drawing pad plots. Both, along with
`data/`, are gitignored — clone the repo and `data.py` re-downloads in a few
seconds.
