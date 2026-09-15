"""Draw a digit with the mouse and watch the model guess.

    python src\\draw.py

Needs a trained checkpoint, so run train.py first.

    left mouse   draw
    right mouse  erase
    C / Backspace  clear
    M              toggle centre of mass alignment
    Tab            switch between Prediction and Training
    Esc            quit

Layout is two columns. The drawing field is on the left and a tabbed panel on
the right. Prediction is the default tab because that is what you use while
drawing. The training curves are something you read once after a run.
"""

import json
import tkinter as tk
from tkinter import font as tkfont

import torch
from PIL import Image, ImageDraw, ImageTk

from digit_prep import to_mnist_frame, to_tensor
from model import SimpleCNN
from train import CHECKPOINT_PATH, HISTORY_PATH

# palette
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
TAB_ON_BG = "#1b1a14"
# two series chart palette, checked for colour blind separation on this surface
SERIES_A  = "#c28527"   # train
SERIES_B  = "#4491cc"   # test

CANVAS   = 336          # drawing field, 12x the model's 28
PEN      = 20
PREVIEW  = 120
PANEL_W  = 272          # right column
PANEL_H  = 470          # fallback only, the real height gets measured at build
CHART_W  = 258
CHART_H  = 128
BAR_W    = 150


def pick_font(root, candidates):
    have = set(tkfont.families(root))
    for name in candidates:
        if name in have:
            return name
    return "TkDefaultFont"


class FlatButton(tk.Frame):
    """A borderless button that doesn't look like 1998."""

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


class Tab(tk.Frame):
    """One entry in the right column's tab strip."""

    def __init__(self, parent, text, command, font):
        super().__init__(parent, bg=BG)
        self.command = command
        self.label = tk.Label(self, text=text, bg=BG, fg=FAINT, font=font,
                              padx=12, pady=5, cursor="hand2")
        self.label.pack()
        self.underline = tk.Frame(self, bg=BG, height=2)
        self.underline.pack(fill="x")
        for w in (self, self.label):
            w.bind("<Button-1>", lambda _e: self.command())
            w.bind("<Enter>", lambda _e: self._hover(True))
            w.bind("<Leave>", lambda _e: self._hover(False))
        self.active = False

    def _hover(self, on):
        if not self.active:
            self.label.configure(fg=MUTED if on else FAINT)

    def set_active(self, active):
        self.active = active
        self.label.configure(fg=ACCENT if active else FAINT,
                             bg=TAB_ON_BG if active else BG)
        self.configure(bg=TAB_ON_BG if active else BG)
        self.underline.configure(bg=ACCENT if active else BG)


class DigitPad:
    def __init__(self, root):
        self.root = root
        root.title("Digit Guesser")
        root.configure(bg=BG)
        root.resizable(False, False)

        ui = pick_font(root, ["Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica"])
        mono = pick_font(root, ["Cascadia Mono", "Consolas", "SF Mono", "Courier New"])
        self.ui = ui
        self.f_label = tkfont.Font(family=ui, size=9)
        self.f_body  = tkfont.Font(family=ui, size=10)
        self.f_btn   = tkfont.Font(family=ui, size=10)
        self.f_tab   = tkfont.Font(family=ui, size=10)
        self.f_num   = tkfont.Font(family=ui, size=58)
        self.f_mono  = tkfont.Font(family=mono, size=9)
        self.f_pct   = tkfont.Font(family=mono, size=9)

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

        outer = tk.Frame(root, bg=BG, padx=26, pady=24)
        outer.pack()

        self.build_left(outer)
        self.build_right(outer)

        for key in ("<Key-c>", "<Key-C>", "<BackSpace>"):
            root.bind(key, lambda e: self.clear())
        for key in ("<Key-m>", "<Key-M>"):
            root.bind(key, lambda e: self.mode_btn._click())
        root.bind("<Tab>", lambda e: (self.cycle_tab(), "break")[1])
        root.bind("<Escape>", lambda e: root.destroy())

        self.refresh_mode()
        self.clear()
        self.center(root)

    # left column
    def build_left(self, parent):
        left = tk.Frame(parent, bg=BG)
        left.grid(row=0, column=0, sticky="n")

        self._eyebrow(left, "DRAW").pack(anchor="w")
        tk.Label(left, text="Digit Guesser", bg=BG, fg=TEXT,
                 font=tkfont.Font(family=self.ui, size=17)).pack(anchor="w", pady=(2, 14))

        field = tk.Frame(left, bg=BG, highlightthickness=1, highlightbackground=LINE)
        field.pack()
        self.canvas = tk.Canvas(field, width=CANVAS, height=CANVAS, bg=INK_FIELD,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", lambda e: self.stroke(e, 255))
        self.canvas.bind("<B1-Motion>", lambda e: self.stroke(e, 255))
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<B3-Motion>", lambda e: self.stroke(e, 0))
        self.canvas.bind("<ButtonRelease-3>", self.release)

        bar = tk.Frame(left, bg=BG)
        bar.pack(fill="x", pady=(14, 0))
        FlatButton(bar, "Clear", self.clear, self.f_btn).pack(side="left")
        self.mode_btn = FlatButton(bar, "", self.toggle_mode, self.f_btn,
                                   toggle=True, on=True)
        self.mode_btn.pack(side="right")

        self.hint = tk.Label(left, text="", bg=BG, fg=FAINT, font=self.f_label)
        self.hint.pack(anchor="w", pady=(12, 0))

        device_name = (f"GPU · {torch.cuda.get_device_name(0)}"
                       if self.device.type == "cuda" else "CPU")
        tk.Label(left, text=device_name, bg=BG, fg=FAINT,
                 font=self.f_label).pack(anchor="w", pady=(6, 0))

    # right column
    def build_right(self, parent):
        right = tk.Frame(parent, bg=BG, padx=26)
        right.grid(row=0, column=1, sticky="n")

        strip = tk.Frame(right, bg=BG)
        strip.pack(anchor="w", pady=(0, 16))
        self.tabs = {}
        for name in ("Prediction", "Training"):
            t = Tab(strip, name, lambda n=name: self.show_tab(n), self.f_tab)
            t.pack(side="left")
            self.tabs[name] = t

        # fixed size so switching tabs never resizes the window. the real
        # height gets set below, once both panels can be measured
        self.stack = tk.Frame(right, bg=BG, width=PANEL_W, height=PANEL_H)
        self.stack.pack(anchor="w")
        self.stack.pack_propagate(False)

        self.panels = {
            "Prediction": self.build_prediction(self.stack),
            "Training": self.build_training(self.stack),
        }

        # size the stack to whichever panel is tallest, measured instead of
        # assumed. font sizes are in points, so at 125% display scaling the
        # content ends up taller than any pixel constant would predict
        heights, widths = [], []
        for panel in self.panels.values():
            panel.pack(anchor="nw")
            self.stack.update_idletasks()
            heights.append(panel.winfo_reqheight())
            widths.append(panel.winfo_reqwidth())
            panel.pack_forget()
        self.stack.configure(width=max(PANEL_W, *widths), height=max(heights) + 2)

        self.show_tab("Prediction")

    def build_prediction(self, parent):
        p = tk.Frame(parent, bg=BG)

        row = tk.Frame(p, bg=BG)
        row.pack(anchor="w")
        self.guess = tk.Label(row, text="–", bg=BG, fg=TEXT, font=self.f_num)
        self.guess.pack(side="left")
        col = tk.Frame(row, bg=BG)
        col.pack(side="left", padx=(14, 0), pady=(20, 0))
        self.conf = tk.Label(col, text="—", bg=BG, fg=ACCENT, font=self.f_body)
        self.conf.pack(anchor="w")
        self.runner = tk.Label(col, text="draw something", bg=BG, fg=FAINT,
                               font=self.f_label)
        self.runner.pack(anchor="w")

        tk.Frame(p, bg=LINE, height=1).pack(fill="x", pady=14)

        rows = tk.Frame(p, bg=BG)
        rows.pack(anchor="w")
        self.row_widgets = []
        for d in range(10):
            r = tk.Frame(rows, bg=BG)
            r.pack(anchor="w", pady=1)
            lab = tk.Label(r, text=str(d), bg=BG, fg=FAINT, font=self.f_mono, width=2)
            lab.pack(side="left")
            track = tk.Canvas(r, width=BAR_W, height=8, bg=PANEL, highlightthickness=0)
            track.pack(side="left", padx=(2, 8))
            fill = track.create_rectangle(0, 0, 0, 8, fill=ACCENT_DIM, width=0)
            pct = tk.Label(r, text="", bg=BG, fg=FAINT, font=self.f_pct, width=4,
                           anchor="e")
            pct.pack(side="left")
            self.row_widgets.append((lab, track, fill, pct))

        tk.Frame(p, bg=LINE, height=1).pack(fill="x", pady=14)

        foot = tk.Frame(p, bg=BG)
        foot.pack(anchor="w")
        box = tk.Frame(foot, bg=BG, highlightthickness=1, highlightbackground=LINE)
        box.pack(side="left")
        # no width or height here. without an image tkinter reads them as
        # characters and lines instead of pixels, and the widget blows up
        self.preview = tk.Label(box, bg=INK_FIELD, borderwidth=0)
        self.preview.pack()
        # give it a blank frame straight away so the panel gets measured at
        # its real height instead of with an empty placeholder in it
        self.show_preview(Image.new("L", (28, 28), 0))
        tk.Label(foot, text="what the model\nreceives, 28×28", bg=BG, fg=FAINT,
                 font=self.f_label, justify="left").pack(side="left", padx=(12, 0),
                                                         anchor="s", pady=(0, 4))
        return p

    def build_training(self, parent):
        p = tk.Frame(parent, bg=BG)

        history = []
        if HISTORY_PATH.exists():
            try:
                history = json.loads(HISTORY_PATH.read_text())
            except (ValueError, OSError):
                history = []

        if not history:
            tk.Label(p, text="No training run yet.\n\nRun train.py and the loss\n"
                             "and accuracy curves show\nup here.",
                     bg=BG, fg=FAINT, font=self.f_label, justify="left").pack(
                         anchor="w", pady=40)
            return p

        epochs = [h["epoch"] for h in history]

        # final values go in the legend. train and test converge, so endpoint
        # labels on the curve would land on top of each other
        self._eyebrow(p, "CROSS-ENTROPY LOSS").pack(anchor="w", pady=(0, 4))
        legend = tk.Frame(p, bg=BG)
        legend.pack(anchor="w", pady=(0, 6))
        for name, colour, value in (("train", SERIES_A, history[-1]["train_loss"]),
                                    ("test", SERIES_B, history[-1]["test_loss"])):
            sw = tk.Canvas(legend, width=14, height=10, bg=BG, highlightthickness=0)
            sw.create_line(0, 5, 14, 5, fill=colour, width=2)
            sw.pack(side="left")
            tk.Label(legend, text=name, bg=BG, fg=MUTED, font=self.f_label).pack(
                side="left", padx=(5, 3))
            tk.Label(legend, text=f"{value:.3f}", bg=BG, fg=TEXT,
                     font=self.f_pct).pack(side="left", padx=(0, 14))

        cv = tk.Canvas(p, width=CHART_W, height=CHART_H, bg=BG, highlightthickness=0)
        cv.pack(anchor="w")
        self.plot(cv, epochs, [
            {"colour": SERIES_A, "values": [h["train_loss"] for h in history]},
            {"colour": SERIES_B, "values": [h["test_loss"] for h in history]},
        ], lambda v: f"{v:.3f}", label_endpoint=False)

        self._eyebrow(p, "TEST ACCURACY %").pack(anchor="w", pady=(20, 6))
        cv2 = tk.Canvas(p, width=CHART_W, height=CHART_H, bg=BG, highlightthickness=0)
        cv2.pack(anchor="w")
        self.plot(cv2, epochs, [
            {"colour": SERIES_B, "values": [h["test_acc"] for h in history]},
        ], lambda v: f"{v:.2f}")

        tk.Frame(p, bg=LINE, height=1).pack(fill="x", pady=14)
        best = max(h["test_acc"] for h in history)
        tk.Label(p, text=f"best {best:.2f}%   ·   {len(epochs)} epochs",
                 bg=BG, fg=FAINT, font=self.f_label).pack(anchor="w")
        return p

    def show_tab(self, name):
        for panel in self.panels.values():
            panel.pack_forget()
        self.panels[name].pack(anchor="nw")
        for key, tab in self.tabs.items():
            tab.set_active(key == name)
        self.active_tab = name

    def cycle_tab(self):
        order = list(self.panels)
        self.show_tab(order[(order.index(self.active_tab) + 1) % len(order)])

    # charts
    def plot(self, cv, epochs, series, fmt, label_endpoint=True):
        """Small line chart: one y-scale, recessive grid, nothing clipped.

        Insets are measured from the rendered text rather than guessed, which
        is what keeps the y-axis labels from running off the left edge.
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

        for frac, tick in zip((1.0, 0.5, 0.0), ticks[::-1]):
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
        colour = "#ffffff" if ink else INK_FIELD
        if self.last is not None:
            self.canvas.create_line(*self.last, x, y, width=PEN, fill=colour,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last, (x, y)], fill=ink, width=PEN, joint="curve")
        r = PEN / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=colour, outline=colour)
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
        self.guess.configure(text="–")
        self.conf.configure(text="—")
        self.runner.configure(text="draw something")
        self.set_bars([0.0] * 10)
        self.show_preview(Image.new("L", (28, 28), 0))

    def toggle_mode(self):
        self.center_by_mass = self.mode_btn.on
        self.refresh_mode()
        self.predict()

    def refresh_mode(self):
        self.mode_btn.set_text("Centre of mass" if self.mode_btn.on else "Bounding box")
        self.hint.configure(text="right click erase  ·  C clear  ·  M align  ·  Esc quit")

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
        self.guess.configure(text=str(best))
        self.conf.configure(text=f"{probs[best]:.0%} confident")
        self.runner.configure(text=f"next best · {second} at {probs[second]:.0%}")
        self.set_bars(probs, best)

    def set_bars(self, probs, best=None):
        for d, (lab, track, fill, pct) in enumerate(self.row_widgets):
            top = d == best
            track.coords(fill, 0, 0, max(0.0, probs[d]) * BAR_W, 8)
            track.itemconfig(fill, fill=ACCENT if top else ACCENT_DIM)
            lab.configure(fg=ACCENT if top else FAINT)
            pct.configure(text=f"{probs[d]:.0%}" if probs[d] >= 0.005 else "",
                          fg=ACCENT if top else FAINT)

    def show_preview(self, frame28):
        img = frame28.resize((PREVIEW, PREVIEW), Image.NEAREST)
        self._preview_ref = ImageTk.PhotoImage(img)
        # with an image set, width and height count as pixels
        self.preview.configure(image=self._preview_ref,
                               width=PREVIEW, height=PREVIEW)


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
