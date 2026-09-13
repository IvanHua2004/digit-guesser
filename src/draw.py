"""Draw a digit with the mouse and watch the model guess.

    python src\\draw.py

Needs a trained checkpoint, so run train.py first.

    left mouse   draw
    right mouse  erase
    C / Backspace  clear
    M              toggle centre-of-mass alignment
    Esc            quit

The panel on the right shows the 28x28 the model actually receives. Watch it
while you draw - most surprising predictions make sense once you see it.
"""

import json
import tkinter as tk
from tkinter import font as tkfont

import torch
from PIL import Image, ImageDraw, ImageTk

from digit_prep import to_mnist_frame, to_tensor
from model import SimpleCNN
from train import CHECKPOINT_PATH, HISTORY_PATH

# ---------------------------------------------------------------- palette --
BG        = "#0e1116"   # window
PANEL     = "#161a21"   # raised surfaces
INK_FIELD = "#080a0d"   # the drawing field itself
LINE      = "#232935"   # hairlines
LINE_HI   = "#2f3745"   # hairline, hover
TEXT      = "#e8ecf2"
MUTED     = "#8a93a3"
FAINT     = "#5b6472"
ACCENT    = "#e0a33e"
ACCENT_DIM= "#6b5223"
SERIES_A  = "#c28527"   # train
SERIES_B  = "#4491cc"   # test

CANVAS   = 336          # drawing field, 12x the model's 28
PEN      = 20
PREVIEW  = 168
CHART_W  = 258
CHART_H  = 128


def pick_font(root, candidates):
    have = set(tkfont.families(root))
    for name in candidates:
        if name in have:
            return name
    return "TkDefaultFont"


class FlatButton(tk.Frame):
    def __init__(self, parent, text, command, font, toggle=False, on=False):
        super().__init__(parent, bg=PANEL, highlightthickness=1,
                         highlightbackground=LINE, highlightcolor=LINE)
        self.command, self.toggle, self.on = command, toggle, on
        self.label = tk.Label(self, text=text, bg=PANEL, fg=MUTED, font=font,
                              padx=14, pady=7, cursor="hand2")
        self.label.pack()
        for w in (self, self.label):
            w.bind("<Button-1>", self._click)
            w.bind("<Enter>", self._enter)
            w.bind("<Leave>", self._leave)
        self._paint()

    def _click(self, _e=None):
        if self.toggle:
            self.on = not self.on
            self._paint()
        self.command()

    def _enter(self, _e=None):
        if not self.on:
            self.configure(highlightbackground=LINE_HI)
            self.label.configure(fg=TEXT)

    def _leave(self, _e=None):
        if not self.on:
            self.configure(highlightbackground=LINE)
            self.label.configure(fg=MUTED)

    def _paint(self):
        if self.on:
            self.configure(highlightbackground=ACCENT_DIM)
            self.label.configure(fg=ACCENT)
        else:
            self.configure(highlightbackground=LINE)
            self.label.configure(fg=MUTED)

    def set_text(self, t):
        self.label.configure(text=t)


class DigitPad:
    def __init__(self, root):
        self.root = root
        root.title("Digit Guesser")
        root.configure(bg=BG)
        root.resizable(False, False)

        ui = pick_font(root, ["Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica"])
        mono = pick_font(root, ["Cascadia Mono", "Consolas", "SF Mono", "Courier New"])
        self.f_label = tkfont.Font(family=ui, size=9)
        self.f_body  = tkfont.Font(family=ui, size=10)
        self.f_btn   = tkfont.Font(family=ui, size=10)
        self.f_num   = tkfont.Font(family=ui, size=64, weight="normal")
        self.f_mono  = tkfont.Font(family=mono, size=9)
        self.f_pct   = tkfont.Font(family=mono, size=9)

        # ---- model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=self.device))
        self.model.eval()

        self.center_by_mass = True
        self.last = None
        self.pending = None
        self._preview_ref = None

        self.image = Image.new("L", (CANVAS, CANVAS), 0)
        self.draw = ImageDraw.Draw(self.image)

        outer = tk.Frame(root, bg=BG, padx=28, pady=26)
        outer.pack()

        # left: the canvas
        left = tk.Frame(outer, bg=BG)
        left.grid(row=0, column=0, sticky="n")

        self._eyebrow(left, "DRAW").pack(anchor="w")
        tk.Label(left, text="Digit Guesser", bg=BG, fg=TEXT,
                 font=tkfont.Font(family=ui, size=17)).pack(anchor="w", pady=(2, 14))

        field = tk.Frame(left, bg=BG, highlightthickness=1, highlightbackground=LINE)
        field.pack()
        self.canvas = tk.Canvas(field, width=CANVAS, height=CANVAS, bg=INK_FIELD,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", lambda e: self.stroke(e, 255))
        self.canvas.bind("<Button-1>", lambda e: self.stroke(e, 255))
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<B3-Motion>", lambda e: self.stroke(e, 0))
        self.canvas.bind("<ButtonRelease-3>", self.release)

        bar = tk.Frame(left, bg=BG)
        bar.pack(fill="x", pady=(14, 0))
        self.clear_btn = FlatButton(bar, "Clear", self.clear, self.f_btn)
        self.clear_btn.pack(side="left")
        self.mode_btn = FlatButton(bar, "", self.toggle_mode, self.f_btn,
                                   toggle=True, on=True)
        self.mode_btn.pack(side="right")

        self.hint = tk.Label(left, text="", bg=BG, fg=FAINT, font=self.f_label,
                             anchor="w", justify="left")
        self.hint.pack(fill="x", pady=(12, 0))

        # right: the readout
        right = tk.Frame(outer, bg=BG, padx=30)
        right.grid(row=0, column=1, sticky="n")

        self._eyebrow(right, "PREDICTION").pack(anchor="w")

        row = tk.Frame(right, bg=BG)
        row.pack(anchor="w")
        self.guess = tk.Label(row, text="–", bg=BG, fg=TEXT, font=self.f_num)
        self.guess.pack(side="left")
        conf_col = tk.Frame(row, bg=BG)
        conf_col.pack(side="left", padx=(16, 0), pady=(22, 0))
        self.conf = tk.Label(conf_col, text="—", bg=BG, fg=ACCENT, font=self.f_body)
        self.conf.pack(anchor="w")
        self.runner = tk.Label(conf_col, text="draw something", bg=BG, fg=FAINT,
                               font=self.f_label)
        self.runner.pack(anchor="w")

        tk.Frame(right, bg=LINE, height=1).pack(fill="x", pady=16)

        # ten rows: digit, bar, percentage
        rows = tk.Frame(right, bg=BG)
        rows.pack(anchor="w")
        self.row_widgets = []
        for d in range(10):
            r = tk.Frame(rows, bg=BG)
            r.pack(anchor="w", pady=1)
            lab = tk.Label(r, text=str(d), bg=BG, fg=FAINT, font=self.f_mono, width=2)
            lab.pack(side="left")
            track = tk.Canvas(r, width=170, height=8, bg=PANEL, highlightthickness=0)
            track.pack(side="left", padx=(4, 8))
            fill = track.create_rectangle(0, 0, 0, 8, fill=ACCENT_DIM, width=0)
            pct = tk.Label(r, text="", bg=BG, fg=FAINT, font=self.f_pct, width=4,
                           anchor="e")
            pct.pack(side="left")
            self.row_widgets.append((lab, track, fill, pct))

        tk.Frame(right, bg=LINE, height=1).pack(fill="x", pady=16)

        self._eyebrow(right, "MODEL INPUT  28 × 28").pack(anchor="w", pady=(0, 8))
        pv = tk.Frame(right, bg=BG, highlightthickness=1, highlightbackground=LINE)
        pv.pack(anchor="w")
        self.preview = tk.Label(pv, bg=INK_FIELD, width=PREVIEW, height=PREVIEW)
        self.preview.pack()

        self.status = tk.Label(right, text="", bg=BG, fg=FAINT, font=self.f_label)
        self.status.pack(anchor="w", pady=(12, 0))

        # far right: curves
        far = tk.Frame(outer, bg=BG)
        far.grid(row=0, column=2, sticky="n")
        self.build_charts(far)

        # keys
        for k in ("c", "C", "<BackSpace>"):
            root.bind(k if k.startswith("<") else f"<Key-{k}>", lambda e: self.clear())
        for k in ("m", "M"):
            root.bind(f"<Key-{k}>", lambda e: self.mode_btn._click())
        root.bind("<Escape>", lambda e: root.destroy())

        self.refresh_mode()
        self.clear()
        self.status.configure(
            text=f"{'GPU · ' + torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'}"
        )
        self.center(root)

    # charts 
    def build_charts(self, parent):
        """Per-epoch curves from the last training run, if there was one."""
        self._eyebrow(parent, "TRAINING").pack(anchor="w")
        tk.Label(parent, text="Last run", bg=BG, fg=TEXT,
                 font=tkfont.Font(family=self.f_body.cget("family"), size=17)
                 ).pack(anchor="w", pady=(2, 14))

        history = []
        if HISTORY_PATH.exists():
            try:
                history = json.loads(HISTORY_PATH.read_text())
            except (ValueError, OSError):
                history = []

        if not history:
            box = tk.Frame(parent, bg=BG, highlightthickness=1,
                           highlightbackground=LINE, width=CHART_W, height=200)
            box.pack_propagate(False)
            box.pack()
            tk.Label(box, text="No training history yet.\n\nRun train.py and the\n"
                              "loss and accuracy curves\nappear here.",
                     bg=BG, fg=FAINT, font=self.f_label, justify="left").pack(
                         expand=True, padx=16)
            return

        epochs = [h["epoch"] for h in history]

        # cross-entropy loss: two series, so it carries a legend
        self._eyebrow(parent, "CROSS-ENTROPY LOSS  ·  PER EPOCH").pack(anchor="w", pady=(0, 4))
        # final values live in the legend, not on the curve: train and test
        # converge, so endpoint labels would sit on top of each other
        legend = tk.Frame(parent, bg=BG)
        legend.pack(anchor="w", pady=(0, 6))
        finals = (("train", SERIES_A, history[-1]["train_loss"]),
                  ("test", SERIES_B, history[-1]["test_loss"]))
        for name, colour, value in finals:
            swatch = tk.Canvas(legend, width=14, height=10, bg=BG, highlightthickness=0)
            swatch.create_line(0, 5, 14, 5, fill=colour, width=2)
            swatch.pack(side="left")
            tk.Label(legend, text=name, bg=BG, fg=MUTED, font=self.f_label).pack(
                side="left", padx=(5, 3))
            tk.Label(legend, text=f"{value:.3f}", bg=BG, fg=TEXT,
                     font=self.f_pct).pack(side="left", padx=(0, 14))

        loss_cv = tk.Canvas(parent, width=CHART_W, height=CHART_H, bg=BG,
                            highlightthickness=0)
        loss_cv.pack(anchor="w")
        self.plot(loss_cv, epochs, [
            {"colour": SERIES_A, "values": [h["train_loss"] for h in history]},
            {"colour": SERIES_B, "values": [h["test_loss"] for h in history]},
        ], lambda v: f"{v:.3f}", label_endpoint=False)

        # accuracy: one series, so the heading names it and no legend is needed
        self._eyebrow(parent, "TEST ACCURACY %  ·  PER EPOCH").pack(anchor="w", pady=(18, 4))
        acc_cv = tk.Canvas(parent, width=CHART_W, height=CHART_H, bg=BG,
                           highlightthickness=0)
        acc_cv.pack(anchor="w")
        self.plot(acc_cv, epochs, [
            {"colour": SERIES_B, "values": [h["test_acc"] for h in history]},
        ], lambda v: f"{v:.2f}")

        best = max(h["test_acc"] for h in history)
        tk.Label(parent, text=f"best {best:.2f}%  ·  {len(epochs)} epochs",
                 bg=BG, fg=FAINT, font=self.f_label).pack(anchor="w", pady=(10, 0))

    def plot(self, cv, epochs, series, fmt, label_endpoint=True):
        """A small line chart: one y-scale, recessive grid, nothing clipped.

        Insets are measured from the rendered text rather than guessed, which
        is what stops the y-axis labels running off the left edge.
        """
        values = [v for s in series for v in s["values"]]
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        margin = (hi - lo) * 0.2
        lo, hi = lo - margin, hi + margin

        ticks = [lo, (lo + hi) / 2, hi]
        left = max(self.f_pct.measure(fmt(t)) for t in ticks) + 12
        right = (self.f_pct.measure(fmt(values[-1])) + 16) if label_endpoint else 14
        top, bottom = 14, 24
        pw = CHART_W - left - right
        ph = CHART_H - top - bottom

        def X(i):
            return left + (pw * i / max(1, len(epochs) - 1))

        def Y(v):
            return top + ph * (1 - (v - lo) / (hi - lo))

        for frac, tick in zip((1.0, 0.5, 0.0), ticks[::-1]):   # recessive grid
            y = top + ph * (1 - frac)
            cv.create_line(left, y, left + pw, y, fill=LINE)
            cv.create_text(left - 8, y, text=fmt(tick), anchor="e",
                           fill=FAINT, font=self.f_pct)

        step = max(1, (len(epochs) + 5) // 6)
        for i, e in enumerate(epochs):
            if i % step == 0 or i == len(epochs) - 1:
                cv.create_text(X(i), top + ph + 12, text=str(e),
                               fill=FAINT, font=self.f_pct)

        for s in series:
            pts = [(X(i), Y(v)) for i, v in enumerate(s["values"])]
            if len(pts) > 1:
                cv.create_line([c for p in pts for c in p],
                               fill=s["colour"], width=2)
            for x, y in pts:
                cv.create_oval(x - 4, y - 4, x + 4, y + 4,
                               fill=s["colour"], outline=BG, width=2)
            if label_endpoint:
                x, y = pts[-1]
                cv.create_text(x + 10, y, text=fmt(s["values"][-1]), anchor="w",
                               fill=TEXT, font=self.f_pct)

    # helpers 
    def _eyebrow(self, parent, text):
        return tk.Label(parent, text=text, bg=BG, fg=FAINT, font=self.f_label)

    @staticmethod
    def center(root):
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        x = (root.winfo_screenwidth() - w) // 2
        y = max(0, (root.winfo_screenheight() - h) // 2 - 30)
        root.geometry(f"+{x}+{y}")

    # input 
    def stroke(self, event, ink):
        x, y = event.x, event.y
        color = "#ffffff" if ink else INK_FIELD
        if self.last is not None:
            self.canvas.create_line(*self.last, x, y, width=PEN, fill=color,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last, (x, y)], fill=ink, width=PEN, joint="curve")
        r = PEN / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=color)
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=ink)
        self.last = (x, y)
        self.schedule()

    def release(self, _e):
        self.last = None
        self.predict()

    def schedule(self):
        """Coalesce rapid mouse events into one prediction every ~50ms."""
        if self.pending is None:
            self.pending = self.root.after(50, self._run)

    def _run(self):
        self.pending = None
        self.predict()

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS, CANVAS], fill=0)
        self.last = None
        self.guess.configure(text="–", fg=TEXT)
        self.conf.configure(text="—")
        self.runner.configure(text="draw something")
        self.set_bars([0.0] * 10)
        self.show_preview(Image.new("L", (28, 28), 0))

    def toggle_mode(self):
        self.center_by_mass = self.mode_btn.on
        self.refresh_mode()
        self.predict()

    def refresh_mode(self):
        on = self.mode_btn.on
        self.mode_btn.set_text("Centre of mass" if on else "Bounding box")
        self.hint.configure(text="Right click to erase  ·  C clear  ·  M align  ·  Esc quit")

    # inference 
    @torch.no_grad()
    def predict(self):
        frame = to_mnist_frame(self.image, center_by_mass=self.center_by_mass)
        self.show_preview(frame)

        x = to_tensor(frame).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().tolist()

        order = sorted(range(10), key=lambda i: -probs[i])
        best, second = order[0], order[1]
        if probs[best] < 0.001:
            return
        self.guess.configure(text=str(best), fg=TEXT)
        self.conf.configure(text=f"{probs[best]:.0%} confident")
        self.runner.configure(text=f"next best · {second} at {probs[second]:.0%}")
        self.set_bars(probs, best)

    def set_bars(self, probs, best=None):
        for d, (lab, track, fill, pct) in enumerate(self.row_widgets):
            top = d == best
            track.coords(fill, 0, 0, max(0, probs[d]) * 170, 8)
            track.itemconfig(fill, fill=ACCENT if top else ACCENT_DIM)
            lab.configure(fg=ACCENT if top else FAINT)
            pct.configure(text=f"{probs[d]:.0%}" if probs[d] >= 0.005 else "",
                          fg=ACCENT if top else FAINT)

    def show_preview(self, frame28):
        img = frame28.resize((PREVIEW, PREVIEW), Image.NEAREST)
        self._preview_ref = ImageTk.PhotoImage(img)
        self.preview.configure(image=self._preview_ref, width=PREVIEW, height=PREVIEW)


def main():
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(
            f"No trained model at {CHECKPOINT_PATH}\n"
            "Run  python src\\train.py --epochs 3  first."
        )
    root = tk.Tk()
    DigitPad(root)
    root.mainloop()


if __name__ == "__main__":
    main()
