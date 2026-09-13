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

import tkinter as tk
from tkinter import font as tkfont

import torch
from PIL import Image, ImageDraw, ImageTk

from digit_prep import to_mnist_frame, to_tensor
from model import SimpleCNN
from train import CHECKPOINT_PATH

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

CANVAS   = 336          # drawing field, 12x the model's 28
PEN      = 20
PREVIEW  = 168


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


class DigitPad:
    def __init__(self, root):
        self.root = root
        root.title("Digit Pad")
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

        # ================================================= left: the canvas
        left = tk.Frame(outer, bg=BG)
        left.grid(row=0, column=0, sticky="n")

        self._eyebrow(left, "DRAW").pack(anchor="w")
        tk.Label(left, text="Digit Pad", bg=BG, fg=TEXT,
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

        # ================================================ right: the readout
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

        # ---- keys
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

    # ---------------------------------------------------------- helpers --
    def _eyebrow(self, parent, text):
        return tk.Label(parent, text=text, bg=BG, fg=FAINT, font=self.f_label)

    @staticmethod
    def center(root):
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        x = (root.winfo_screenwidth() - w) // 2
        y = max(0, (root.winfo_screenheight() - h) // 2 - 30)
        root.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------ input --
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

    # -------------------------------------------------------- inference --
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
