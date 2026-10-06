# draw_window.py
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math


class DrawingApp:
    def __init__(self, network, canvas_size=280, brush_size=20):
        self.network = network
        self.canvas_size = canvas_size
        self.brush_size = brush_size
        self.grid_size = 28
        self.scale = canvas_size // self.grid_size

        self.image = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.erase_mode = False

        # Debounce handle for auto-update
        self._update_job = None

        # ----- Build UI -----
        self.root = tk.Tk()
        self.root.title("Digit Recognizer — Draw a Number")

        # Left column: drawing canvas + controls
        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="n")

        self.canvas = tk.Canvas(
            left,
            width=canvas_size,
            height=canvas_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="gray",
        )
        self.canvas.grid(row=0, column=0, columnspan=3, pady=(0, 8))
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Button-1>", self.on_drag)

        ttk.Button(left, text="Clear", command=self.clear).grid(
            row=1, column=0, sticky="ew", padx=2
        )
        self.erase_btn = ttk.Button(
            left, text="Erase: OFF", command=self.toggle_erase
        )
        self.erase_btn.grid(row=1, column=1, sticky="ew", padx=2)
        ttk.Button(left, text="Predict", command=self.predict).grid(
            row=1, column=2, sticky="ew", padx=2
        )

        # Big prediction label
        self.pred_label = ttk.Label(
            left, text="Prediction: —",
            font=("Helvetica", 16, "bold")
        )
        self.pred_label.grid(row=2, column=0, columnspan=3, pady=(10, 2))

        # Probability bars
        prob_frame = ttk.LabelFrame(left, text="Probabilities")
        prob_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        self.prob_bars = []
        self.prob_labels = []
        for i in range(10):
            ttk.Label(prob_frame, text=str(i), width=3).grid(
                row=i, column=0, sticky="w"
            )
            bar = ttk.Progressbar(
                prob_frame, orient="horizontal", length=180, maximum=100
            )
            bar.grid(row=i, column=1, sticky="ew", padx=4, pady=1)
            lbl = ttk.Label(prob_frame, text="0.0%", width=7)
            lbl.grid(row=i, column=2, sticky="e")
            self.prob_bars.append(bar)
            self.prob_labels.append(lbl)

        prob_frame.columnconfigure(1, weight=1)

        # Right column: layer visualization
        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky="nsew")

        layers_frame = ttk.LabelFrame(right, text="Layer Activations")
        layers_frame.pack(fill="both", expand=True)

        n_layers = len(network.weights) + 1  # input + each layer output

        # Figure size scales with number of layers
        fig_w = max(6, 2.8 * n_layers)
        self.fig = Figure(figsize=(fig_w, 3.4), dpi=90)
        self.fig.subplots_adjust(
            left=0.02, right=0.98, top=0.85, bottom=0.05, wspace=0.25
        )

        self.axes = []
        for i in range(n_layers):
            ax = self.fig.add_subplot(1, n_layers, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Layer {i}", fontsize=9)
            self.axes.append(ax)

        self.layer_canvas = FigureCanvasTkAgg(self.fig, master=layers_frame)
        self.layer_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Key bindings
        self.root.bind("<space>", lambda e: self.clear())
        self.root.bind("<Return>", lambda e: self.predict())
        self.root.bind("<e>", lambda e: self.toggle_erase())

        # Initial render
        self._render_layers(self._current_activations())

    # ---------- Helpers ----------
    @staticmethod
    def _square_side(n):
        return max(1, math.ceil(math.sqrt(n)))

    def _activations_to_grid(self, vec):
        vec = np.asarray(vec, dtype=float)
        n = len(vec)
        side = self._square_side(n)
        pad = side * side - n
        if pad > 0:
            vec = np.concatenate([vec, np.zeros(pad)])
        return vec.reshape(side, side)

    def _current_activations(self):
        """Run forward on current drawing, return list of activations."""
        flat = self.image.flatten().astype(float).tolist()
        return self.network.forward(flat)

    # ---------- Drawing ----------
    def on_drag(self, event):
        x, y = event.x, event.y
        val = 0 if self.erase_mode else 255

        gx = x // self.scale
        gy = y // self.scale
        r = self.brush_size // self.scale

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist <= r:
                        if dist > r - 1:
                            blend = 1 - (dist - (r - 1))
                            self.image[ny, nx] = max(
                                self.image[ny, nx], int(val * blend)
                            )
                        else:
                            self.image[ny, nx] = val

        self.redraw_canvas()
        self.schedule_update()

    def redraw_canvas(self):
        self.canvas.delete("all")
        s = self.scale
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                v = int(self.image[y, x])
                if v > 0:
                    color = f"#{v:02x}{v:02x}{v:02x}"
                    self.canvas.create_rectangle(
                        x * s, y * s, (x + 1) * s, (y + 1) * s,
                        fill=color, outline=color,
                        )

    # ---------- Auto update with debounce ----------
    def schedule_update(self):
        """Coalesce rapid events: run one update ~40 ms after the last drag."""
        if self._update_job is not None:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(40, self._do_update)

    def _do_update(self):
        self._update_job = None
        self.update_prediction()
        self._render_layers(self._current_activations())

    # ---------- Controls ----------
    def clear(self):
        self.image[:] = 0
        self.redraw_canvas()
        self.schedule_update()

    def toggle_erase(self):
        self.erase_mode = not self.erase_mode
        self.erase_btn.config(
            text=f"Erase: {'ON' if self.erase_mode else 'OFF'}"
        )

    # ---------- Prediction ----------
    def update_prediction(self, activations=None):
        if activations is None:
            activations = self._current_activations()
        output = list(activations[-1])


        vecLen = len(output)
        softmax = [0]*vecLen

        sm = sum(math.exp(i) for i in output)
        for i in range(vecLen):
            softmax[i] = math.exp(output[i])/sm

        best = int(np.argmax(softmax))
        self.pred_label.config(text=f"Prediction: {best}")

        for i, (bar, lbl) in enumerate(zip(self.prob_bars, self.prob_labels)):
            p = float(softmax[i]) * 100
            bar["value"] = p
            lbl.config(text=f"{p:.1f}%")

    def predict(self):
        acts = self._current_activations()
        self.update_prediction(acts)
        self._render_layers(acts)

    # ---------- Layer visualization (embedded on main screen) ----------
    def _render_layers(self, activations):
        for i, act in enumerate(activations):
            ax = self.axes[i]
            ax.clear()

            grid = self._activations_to_grid(act)

            # Optional: show softmax for the final layer
            if i == len(activations) - 1:
                out = np.array(act, dtype=float)
                exps = np.exp(out - out.max())
                probs = exps / exps.sum()
                grid = self._activations_to_grid(probs.tolist())

            gmax = grid.max() if grid.max() > 0 else 1.0
            ax.imshow(
                grid, cmap="viridis", vmin=0, vmax=gmax,
                interpolation="nearest",
            )

            side = grid.shape[0]
            dim = len(act)
            ax.set_title(
                f"Layer {i}\n{dim} → {side}×{side}\nmax={gmax:.2f}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        self.layer_canvas.draw_idle()

    def run(self):
        self.root.mainloop()


def launch(network):
    app = DrawingApp(network)
    app.run()