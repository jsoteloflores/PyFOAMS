# postprocessing_gui.py
# Step 3 GUI: Binary mask editor (brush, eraser, Otsu-brush, fill) with zoom/pan, undo/redo.

from __future__ import annotations
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Dict, Tuple

from PIL import Image, ImageTk

# Pillow resampling shim
if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)

ToolType = str  # "brush_fg" | "brush_bg" | "otsu_brush" | "fill_fg" | "fill_bg"

def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return g

def colorize_overlay(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay cyan on mask==255 over grayscale."""
    if gray.ndim == 2:
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        base = gray.copy()
    m = (mask > 0).astype(np.uint8)
    overlay = base.copy()
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], (m * 255))  # G
    overlay[:, :, 2] = np.maximum(overlay[:, :, 2], (m * 255))  # R
    out = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
    return out

class PostprocessWindow(tk.Toplevel):
    """
    Edit binary masks for a list of images.
    Tools:
      - Brush FG (paint pores=255)
      - Eraser BG (paint matrix=0)
      - Otsu Brush (local Otsu threshold inside a circular brush window)
      - Fill FG / Fill BG (flood fill)
    UI:
      - Size slider, overlay alpha, zoom/pan, undo/redo
      - Prev/Next navigation
      - Apply (save masks back via callback)
    """
    def __init__(
        self,
        parent: tk.Tk,
        images: List[np.ndarray],
        masks: Optional[List[Optional[np.ndarray]]] = None,
        paths: Optional[List[str]] = None,
        startIndex: int = 0,
        onMasksUpdated=None,   # callback(List[Optional[np.ndarray]])
    ):
        super().__init__(parent)
        self.title("PyFOAMS – Post-processing (Mask Editor)")
        self.transient(parent)
        self.grab_set()

        self.images = images
        self.paths = paths or [f"Image {i+1}" for i in range(len(images))]
        self.gray = [to_gray_u8(im) for im in images]
        self.masks = [None] * len(images) if masks is None else [
            (m.copy() if m is not None else None) for m in masks
        ]
        self.index = max(0, min(startIndex, len(images)-1))
        self.onMasksUpdated = onMasksUpdated

        # State
        self.tool: ToolType = "brush_fg"
        self.brushSize = tk.IntVar(value=12)
        self.overlayAlpha = tk.DoubleVar(value=0.45)
        self.viewMaskOnly = tk.BooleanVar(value=False)
        self._scale = 1.0
        self._offset = np.array([0.0, 0.0], dtype=float)
        self._photo = None
        self._pil = None

        # Undo/redo stacks per image
        self._undo: List[List[np.ndarray]] = [[] for _ in images]
        self._redo: List[List[np.ndarray]] = [[] for _ in images]

        # Build UI
        self._build_ui()
        self._load_current()

        self.geometry("1180x800")
        self.minsize(900, 600)

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self); top.pack(side="top", fill="x")

        ttk.Button(top, text="Prev", command=self._prev).pack(side="left", padx=4, pady=6)
        ttk.Button(top, text="Next", command=self._next).pack(side="left", padx=4)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)

        # Tools
        self.toolVar = tk.StringVar(value=self.tool)
        for text, val in [
            ("Brush FG", "brush_fg"),
            ("Eraser BG", "brush_bg"),
            ("Otsu-Brush", "otsu_brush"),
            ("Fill FG", "fill_fg"),
            ("Fill BG", "fill_bg"),
        ]:
            ttk.Radiobutton(top, text=text, variable=self.toolVar, value=val,
                            command=lambda v=val: self._set_tool(v)).pack(side="left", padx=4)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Label(top, text="Size").pack(side="left")
        ttk.Scale(top, from_=1, to=128, orient="horizontal", variable=self.brushSize, length=160)\
            .pack(side="left", padx=4)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Checkbutton(top, text="Mask only", variable=self.viewMaskOnly, command=self._render).pack(side="left", padx=4)
        ttk.Label(top, text="Overlay α").pack(side="left", padx=(12,2))
        ttk.Scale(top, from_=0.0, to=1.0, orient="horizontal", variable=self.overlayAlpha, length=120,
                  command=lambda v: self._render()).pack(side="left", padx=2)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(top, text="Undo", command=self._undo_action).pack(side="left", padx=4)
        ttk.Button(top, text="Redo", command=self._redo_action).pack(side="left", padx=4)
        ttk.Button(top, text="Apply (save masks)", command=self._apply).pack(side="left", padx=12)

        # Canvas
        self.canvas = tk.Canvas(self, bg="#111", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)

        # Bindings
        self.canvas.bind("<Configure>", lambda e: self._render())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Pan with middle mouse or Space-drag
        self.canvas.bind("<ButtonPress-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._pan_end)
        self.bind("<space>", lambda e: self.canvas.config(cursor="fleur"))
        self.bind("<KeyRelease-space>", lambda e: self.canvas.config(cursor=""))

        # Keyboard shortcuts
        self.bind("<Control-z>", lambda e: self._undo_action())
        self.bind("<Control-y>", lambda e: self._redo_action())

    def _set_tool(self, v: ToolType):
        self.tool = v

    # ---------- Image/mask IO ----------
    def _ensure_mask(self, i: int):
        if self.masks[i] is None:
            h, w = self.gray[i].shape[:2]
            self.masks[i] = np.zeros((h, w), np.uint8)

    def _load_current(self):
        self._scale = 1.0
        self._offset[:] = 0.0
        self._pil = Image.fromarray(self.gray[self.index], mode="L")
        self._render()
        name = os.path.basename(self.paths[self.index]) if self.paths else f"Image {self.index+1}"
        self.title(f"Mask Editor – {name} ({self.index+1}/{len(self.images)})")

    # ---------- Render ----------
    def _render(self):
        if self._pil is None:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        iw, ih = self._pil.size
        base = min(cw / float(iw), ch / float(ih))
        base = min(base, 1.0)
        s = base * self._scale

        disp_w = max(1, int(round(iw * s)))
        disp_h = max(1, int(round(ih * s)))
        disp = self._pil.resize((disp_w, disp_h), resample=RESAMPLE_LANCZOS)
        gx = np.array(disp)

        self.canvas.delete("all")
        ox, oy = int((cw - disp_w) / 2 + self._offset[0]), int((ch - disp_h) / 2 + self._offset[1])

        # Compose view
        m = self.masks[self.index]
        if m is None:
            view = gx
        else:
            m_disp = cv2.resize(m, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            if self.viewMaskOnly.get():
                view = m_disp
            else:
                # overlay on gray
                base_rgb = cv2.cvtColor(gx, cv2.COLOR_GRAY2BGR)
                view = colorize_overlay(base_rgb, m_disp, alpha=float(self.overlayAlpha.get()))
        if view.ndim == 2:
            pil = Image.fromarray(view, mode="L")
        else:
            pil = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB), mode="RGB")

        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo, tags="img")
        self._dispOrigin = (ox, oy)
        self._dispScale = s
        self._dispSize = (disp_w, disp_h)

    # ---------- Navigation ----------
    def _prev(self):
        if self.index > 0:
            self.index -= 1
            self._load_current()

    def _next(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self._load_current()

    # ---------- Zoom/Pan ----------
    def _on_wheel(self, event):
        delta = 1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else -1
        factor = 1.1 if delta > 0 else 0.9
        self._scale = max(0.1, min(self._scale * factor, 20.0))
        self._render()

    def _pan_start(self, event):
        self._lastPan = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _pan_drag(self, event):
        if getattr(self, "_lastPan", None) is None:
            return
        dx = event.x - self._lastPan[0]
        dy = event.y - self._lastPan[1]
        self._offset += np.array([dx, dy], dtype=float)
        self._lastPan = (event.x, event.y)
        self._render()

    def _pan_end(self, event):
        self._lastPan = None
        self.canvas.config(cursor="")

    # ---------- Painting helpers ----------
    def _canvas_to_img(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        s = getattr(self, "_dispScale", 1.0)
        ox, oy = getattr(self, "_dispOrigin", (0, 0))
        iw, ih = self._pil.size
        ix = int((x - ox) / max(s, 1e-6))
        iy = int((y - oy) / max(s, 1e-6))
        if 0 <= ix < iw and 0 <= iy < ih:
            return ix, iy
        return None

    def _push_undo(self):
        self._ensure_mask(self.index)
        m = self.masks[self.index]
        if m is None:
            return
        # limit history
        if len(self._undo[self.index]) > 30:
            self._undo[self.index].pop(0)
        self._undo[self.index].append(m.copy())
        self._redo[self.index].clear()

    def _undo_action(self):
        if not self._undo[self.index]:
            return
        self._ensure_mask(self.index)
        cur = self.masks[self.index]
        prev = self._undo[self.index].pop()
        self._redo[self.index].append(cur.copy())
        self.masks[self.index] = prev
        self._render()

    def _redo_action(self):
        if not self._redo[self.index]:
            return
        self._ensure_mask(self.index)
        cur = self.masks[self.index]
        nxt = self._redo[self.index].pop()
        self._undo[self.index].append(cur.copy())
        self.masks[self.index] = nxt
        self._render()

    # ---------- Tools ----------
    def _on_press(self, event):
        pt = self._canvas_to_img(event.x, event.y)
        if pt is None:
            return
        if self.tool in ("brush_fg", "brush_bg", "otsu_brush"):
            self._push_undo()
            self._paint_at(*pt)
            self._render()
        elif self.tool in ("fill_fg", "fill_bg"):
            self._push_undo()
            self._fill_at(*pt)
            self._render()

    def _on_drag(self, event):
        if self.tool in ("brush_fg", "brush_bg", "otsu_brush"):
            pt = self._canvas_to_img(event.x, event.y)
            if pt is None:
                return
            self._paint_at(*pt)
            self._render()

    def _on_release(self, event):
        pass

    def _paint_at(self, ix: int, iy: int):
        self._ensure_mask(self.index)
        m = self.masks[self.index]
        h, w = m.shape[:2]
        r = int(max(1, self.brushSize.get()))
        x0 = max(0, ix - r); x1 = min(w, ix + r + 1)
        y0 = max(0, iy - r); y1 = min(h, iy + r + 1)

        if self.tool == "brush_fg":
            cv2.circle(m, (ix, iy), r, 255, thickness=-1, lineType=cv2.LINE_8)
        elif self.tool == "brush_bg":
            cv2.circle(m, (ix, iy), r, 0, thickness=-1, lineType=cv2.LINE_8)
        elif self.tool == "otsu_brush":
            # local Otsu threshold within the patch; pores expected to be bright in mask (255)
            g = self.gray[self.index][y0:y1, x0:x1]
            if g.size < 16:
                return
            # blur slightly to stabilize Otsu
            k = max(3, int(r // 3) * 2 + 1)
            g_blur = cv2.GaussianBlur(g, (k, k), 0)
            _, g_bin = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Write into mask only within brush circle
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask_circle = (xx - ix) ** 2 + (yy - iy) ** 2 <= r * r
            patch = m[y0:y1, x0:x1]
            patch[mask_circle] = g_bin[mask_circle]

    def _fill_at(self, ix: int, iy: int):
        self._ensure_mask(self.index)
        m = self.masks[self.index]
        h, w = m.shape[:2]
        seed_val = int(m[iy, ix])
        target_val = 255 if self.tool == "fill_fg" else 0
        if seed_val == target_val:
            return
        # Use OpenCV floodFill with mask (requires 2px border)
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        _, _, _, _ = cv2.floodFill(m, ff_mask, seedPoint=(ix, iy), newVal=target_val,
                                   loDiff=0, upDiff=0, flags=4)  # 4-connectivity

    # ---------- Apply ----------
    def _apply(self):
        # Return masks through callback
        if self.onMasksUpdated:
            self.onMasksUpdated(self.masks)
        messagebox.showinfo("Post-processing", "Masks saved back to the main window.")
