# preprocessing_gui.py
# Preprocessing GUI:
# - Load up to 20 images
# - Dynamic grid (5 columns x up to 4 rows), fills row-by-row
# - Thumbs auto-resize with window
# - Double-click a thumb -> Enlarged viewer with zoom/pan
# - Batch Crop: draw ROI in enlarged view -> confirm current/all (with relative margins)
# - Pixel Scale Calibration (Part 2):
#     * Calibrate mode: click first point, second point constrained to same vertical pixel
#     * Enter real-world length + units, apply to current or all
#     * Manual scale entry (units/px or px/unit)
#     * Per-image scale is displayed and stored

from __future__ import annotations
import os
import math
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import List, Optional, Tuple, Dict
import preprocessing
from PIL import Image, ImageTk

# Pillow resampling shim
if hasattr(Image, "Resampling"):  # Pillow >= 9.1 (incl. 10+)
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)

from preprocessing import (
    loadImage,
    cropWithRect, cropWithMargins, applyCropBatch,
    clampRectToImage, rectToMargins, marginsToRect
)

MAX_IMAGES = 20
GRID_COLS = 5
GRID_ROWS = 4  # capacity 20

Rect = Tuple[int, int, int, int]  # (x0,y0,x1,y1)


class PreprocessApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("PyFOAMS – Preprocessing")
        self.images: List[np.ndarray] = []
        self.paths: List[str] = []
        self.selectedIndex: Optional[int] = None

        # Per-image scales: dict or None, e.g. {"unitsPerPx": 0.005, "unitName": "mm"}
        self.scales: List[Optional[Dict[str, float | str]]] = []

        self._thumbPhotos: Dict[int, ImageTk.PhotoImage] = {}
        self._thumbSizes: Dict[int, Tuple[int, int]] = {}  # cached sizes per cell

        self._buildUi()

    # --------------------------- UI scaffolding ---------------------------

    def _buildUi(self):
        # Toolbar
        self.toolbar = ttk.Frame(self.master)
        self.toolbar.pack(side="top", fill="x")

        ttk.Button(self.toolbar, text="Load Images", command=self.onLoad).pack(side="left", padx=6, pady=6)
        self.cropAllVar = tk.BooleanVar(value=True)  # default apply to all
        ttk.Button(self.toolbar, text="Batch Crop", command=self.onBatchCropClick).pack(side="left", padx=6)
        ttk.Checkbutton(self.toolbar, text="Use relative margins", variable=self.cropAllVar).pack(side="left", padx=6)

        self.statusVar = tk.StringVar(value="No images loaded")
        ttk.Label(self.toolbar, textvariable=self.statusVar).pack(side="left", padx=12)

        # Grid frame
        self.gridFrame = ttk.Frame(self.master)
        self.gridFrame.pack(side="top", fill="both", expand=True)

        # Create a 5x4 grid of canvases (placeholders)
        self.cells: List[tk.Canvas] = []
        for r in range(GRID_ROWS):
            self.gridFrame.rowconfigure(r, weight=1)
        for c in range(GRID_COLS):
            self.gridFrame.columnconfigure(c, weight=1)

        for i in range(GRID_ROWS * GRID_COLS):
            canvas = tk.Canvas(self.gridFrame, bg="#222", highlightthickness=1, highlightbackground="#444")
            r = i // GRID_COLS
            c = i % GRID_COLS
            canvas.grid(row=r, column=c, sticky="nsew")
            canvas.bind("<Configure>", lambda e, idx=i: self._redrawThumb(idx))
            canvas.bind("<Button-1>", lambda e, idx=i: self._onThumbClick(idx))
            canvas.bind("<Double-Button-1>", lambda e, idx=i: self._openEnlarged(idx))
            self.cells.append(canvas)

        # Resize handler (update thumbs)
        self.master.bind("<Configure>", lambda e: self._redrawAllThumbs())

    def onBatchCropClick(self):
        """Open enlarged viewer on the selected image and start crop mode."""
        if not self.images:
            messagebox.showwarning("Batch Crop", "Load images first.")
            return

        idx = self.selectedIndex if self.selectedIndex is not None else 0
        viewer = self._openEnlarged(idx, startCrop=True)  # start crop right away

        # If the main toolbar has a 'Use relative margins' checkbox, sync it into the viewer
        try:
            if hasattr(self, "cropAllVar") and hasattr(viewer, "useMarginsVar"):
                viewer.useMarginsVar.set(bool(self.cropAllVar.get()))
        except Exception:
            pass

    # --------------------------- Image loading ---------------------------

    def onLoad(self):
        paths = filedialog.askopenfilenames(
            title="Select up to 20 images",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")]
        )
        if not paths:
            return
        if len(paths) > MAX_IMAGES:
            messagebox.showwarning("Load", f"Selected {len(paths)} images; only first {MAX_IMAGES} will be used.")
            paths = paths[:MAX_IMAGES]

        imgs: List[np.ndarray] = []
        ok: List[str] = []
        for p in paths:
            try:
                img = loadImage(p, asGray=False)  # show color if present; grayscale later if needed
                imgs.append(img)
                ok.append(p)
            except Exception as e:
                print(f"Load failed for {p}: {e}")

        if not imgs:
            messagebox.showerror("Load", "No images could be loaded.")
            return

        self.images = imgs
        self.paths = ok
        self.scales = [None] * len(self.images)
        self.selectedIndex = 0
        self.statusVar.set(f"Loaded {len(self.images)} images. Double-click to enlarge; click to select.")
        self._redrawAllThumbs()

    # --------------------------- Grid rendering ---------------------------

    def _redrawAllThumbs(self):
        for i in range(GRID_ROWS * GRID_COLS):
            self._redrawThumb(i)

    def _redrawThumb(self, idx: int):
        canvas = self.cells[idx]
        canvas.delete("all")

        if idx >= len(self.images):
            return

        img = self.images[idx]
        pil = self._npToPil(img)
        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())

        # Fit image to cell, no upscale beyond 1:1
        iw, ih = pil.size
        s = min(cw / float(iw), ch / float(ih))
        s = min(s, 1.0)
        new_w = max(1, int(round(iw * s)))
        new_h = max(1, int(round(ih * s)))
        if (new_w, new_h) != (iw, ih):
            pil = pil.resize((new_w, new_h), resample=RESAMPLE_LANCZOS)

        photo = ImageTk.PhotoImage(pil)
        self._thumbPhotos[idx] = photo
        canvas.create_image((cw - new_w) // 2, (ch - new_h) // 2, anchor="nw", image=photo)

        # Selection outline
        if self.selectedIndex == idx:
            canvas.create_rectangle(2, 2, cw - 2, ch - 2, outline="#66b3ff", width=2)

    def _onThumbClick(self, idx: int):
        if idx < len(self.images):
            self.selectedIndex = idx
            self._redrawAllThumbs()

    # --------------------------- Enlarged viewer ---------------------------

    def _openEnlarged(self, idx: int, startCrop: bool = False):
        if idx >= len(self.images):
            return None
        self.selectedIndex = idx
        self._redrawAllThumbs()

        viewer = EnlargedViewer(
            parent=self.master,
            images=self.images,
            paths=self.paths,
            scales=getattr(self, "scales", [None] * len(self.images)) if hasattr(self, "scales") else [None] * len(self.images),
            startIndex=idx,
            onImagesUpdated=self._onImagesUpdated,
            onScalesUpdated=getattr(self, "_onScalesUpdated", lambda s: None)  # no-op if not present
        )

        if startCrop:
            # wait for window to render, then enter crop mode
            self.master.after(100, viewer._startCrop)

        return viewer


    def _onImagesUpdated(self, newImages: List[np.ndarray]):
        """Callback when batch operations (like crop) modify images."""
        self.images = newImages
        self._redrawAllThumbs()

    def _onScalesUpdated(self, newScales: List[Optional[Dict[str, float | str]]]):
        self.scales = newScales
        # Optional: reflect scale somewhere in main window if desired

    # --------------------------- Utilities ---------------------------

    def _npToPil(self, arr: np.ndarray) -> Image.Image:
        a = arr
        if a.ndim == 2:
            return Image.fromarray(a, mode="L")
        if a.ndim == 3 and a.shape[2] == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            return Image.fromarray(a, mode="RGB")
        if a.ndim == 3 and a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(a, mode="RGBA")
        # Fallback to grayscale
        g = a.mean(axis=2).astype(np.uint8) if a.ndim == 3 else a.astype(np.uint8)
        return Image.fromarray(g, mode="L")


# ============================== Enlarged Viewer ==============================

class EnlargedViewer(tk.Toplevel):
    """Modal enlarged viewer with zoom/pan, crop overlay, and pixel-scale calibration.

    Gestures:
      - Mouse wheel: zoom (centered at cursor)
      - Middle mouse drag or Pan mode: pan
      - Left+Right held together: pan (alternative gesture)
      - Buttons on toolbar for zoom/pan/reset (mobile-friendly)
      - Crop: draw rectangle, confirm "current / all" and margins mode
      - Calibrate: click start point, then click second point → constrained to same vertical pixel (same y)
    """

    def __init__(
        self,
        parent: tk.Tk,
        images: List[np.ndarray],
        paths: List[str],
        scales: List[Optional[Dict[str, float | str]]],
        startIndex: int,
        onImagesUpdated,
        onScalesUpdated
    ):
        super().__init__(parent)
        self.title("Viewer")
        self.transient(parent)
        self.grab_set()

        self.images = images
        self.paths = paths
        self.scales = scales
        self.index = startIndex
        self.onImagesUpdated = onImagesUpdated
        self.onScalesUpdated = onScalesUpdated

        self._photo = None
        self._pil = None
        self._scale = 1.0
        self._offset = np.array([0.0, 0.0], dtype=float)  # pan offset
        self._panActive = False
        self._lastDrag = None

        # Crop overlay state
        self._cropMode = False
        self._cropRect: Optional[Rect] = None

        # Calibration state
        self._calibMode = False
        self._calibStart: Optional[Tuple[int, int]] = None  # image coords (x,y)
        self._calibEnd: Optional[Tuple[int, int]] = None    # image coords (x,y), y constrained to start.y
        self._calibPxDist: Optional[float] = None

        # Drawn ids
        self._calibLineId = None
        self._calibTextId = None

        # Build UI
        self._buildUi()
        self._loadCurrent()

        self.geometry("1150x780")
        self.minsize(820, 560)

    # -------- UI --------

    def _buildUi(self):
        # Toolbar
        tb = ttk.Frame(self)
        tb.pack(side="top", fill="x")

        ttk.Button(tb, text="Prev", command=self._prev).pack(side="left", padx=4, pady=6)
        ttk.Button(tb, text="Next", command=self._next).pack(side="left", padx=4)

        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(tb, text="Zoom In", command=lambda: self._zoomAtCenter(1.25)).pack(side="left", padx=2)
        ttk.Button(tb, text="Zoom Out", command=lambda: self._zoomAtCenter(0.8)).pack(side="left", padx=2)
        ttk.Button(tb, text="Reset View", command=self._resetView).pack(side="left", padx=8)
        self.panVar = tk.BooleanVar(value=False)
        ttk.Checkbutton(tb, text="Pan Mode", variable=self.panVar, command=self._updatePanMode).pack(side="left", padx=6)

        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(tb, text="Crop", command=self._startCrop).pack(side="left", padx=4)
        self.useMarginsVar = tk.BooleanVar(value=True)
        ttk.Checkbutton(tb, text="Use relative margins", variable=self.useMarginsVar).pack(side="left", padx=6)
        ttk.Button(tb, text="Apply Crop…", command=self._applyCropDialog).pack(side="left", padx=4)

        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=6)

        # Calibration controls
        self.calibStatusVar = tk.StringVar(value="Scale: not set")
        ttk.Button(tb, text="Calibrate", command=self._startCalibration).pack(side="left", padx=4)
        ttk.Button(tb, text="Set Scale…", command=self._manualScaleDialog).pack(side="left", padx=4)
        ttk.Label(tb, textvariable=self.calibStatusVar).pack(side="left", padx=12)

        # Canvas
        self.canvas = tk.Canvas(self, bg="#111", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)

        # Bindings
        self.canvas.bind("<Configure>", lambda e: self._render())
        self.canvas.bind("<MouseWheel>", self._onWheel)  # Windows/macOS
        self.canvas.bind("<Button-4>", self._onWheel)    # some X11
        self.canvas.bind("<Button-5>", self._onWheel)
        self.canvas.bind("<ButtonPress-2>", self._onPanStart)
        self.canvas.bind("<B2-Motion>", self._onPanDrag)
        self.canvas.bind("<ButtonRelease-2>", self._onPanEnd)

        # Left+Right simult pan
        self.canvas.bind("<ButtonPress-1>", self._onButtonComboDown)
        self.canvas.bind("<ButtonPress-3>", self._onButtonComboDown)
        self.canvas.bind("<ButtonRelease-1>", self._onButtonComboUp)
        self.canvas.bind("<ButtonRelease-3>", self._onButtonComboUp)
        self.canvas.bind("<B1-Motion>", self._onComboDrag)
        self.canvas.bind("<B3-Motion>", self._onComboDrag)

        # Crop interactions (added with "+", so they stack with other bindings)
        self.canvas.bind("<ButtonPress-1>", self._onCropPress, add="+")
        self.canvas.bind("<B1-Motion>", self._onCropDrag, add="+")
        self.canvas.bind("<ButtonRelease-1>", self._onCropRelease, add="+")
        self.canvas.bind("<Double-Button-1>", self._onCropConfirm, add="+")

        # Calibration interactions (Ctrl-click required)
        self.canvas.bind("<Control-Button-1>", self._onCalibPress, add="+")
        self.canvas.bind("<Motion>", self._onCalibMotion, add="+")
        self.canvas.bind("<Control-ButtonRelease-1>", self._onCalibRelease, add="+")


    # -------- Image IO / render --------

    def _loadCurrent(self):
        img = self.images[self.index]
        self._pil = self._npToPil(img)
        self._resetView()
        self._updateTitle()
        self._updateScaleStatus()
        self._render()

    def _updateTitle(self):
        name = os.path.basename(self.paths[self.index]) if self.paths and self.index < len(self.paths) else f"Image {self.index+1}"
        self.title(f"Viewer – {name} ({self.index+1}/{len(self.images)})")

    def _resetView(self):
        self._scale = 1.0
        self._offset[:] = 0.0
        self._render()

    def _render(self):
        if self._pil is None:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        iw, ih = self._pil.size

        # Fit to view initially (respect current scale)
        baseScale = min(cw / float(iw), ch / float(ih))
        baseScale = min(baseScale, 1.0)
        s = baseScale * self._scale

        w = max(1, int(round(iw * s)))
        h = max(1, int(round(ih * s)))
        disp = self._pil.resize((w, h), resample=RESAMPLE_LANCZOS)
        self.canvas.delete("all")
        self._photo = ImageTk.PhotoImage(disp)
        ox, oy = int((cw - w) / 2 + self._offset[0]), int((ch - h) / 2 + self._offset[1])
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo, tags="img")

        # Draw overlays
        self._drawCropOverlay()
        self._drawCalibrationOverlay()

    def _npToPil(self, arr: np.ndarray) -> Image.Image:
        a = arr
        if a.ndim == 2:
            return Image.fromarray(a, mode="L")
        if a.ndim == 3 and a.shape[2] == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            return Image.fromarray(a, mode="RGB")
        if a.ndim == 3 and a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(a, mode="RGBA")
        g = a.mean(axis=2).astype(np.uint8) if a.ndim == 3 else a.astype(np.uint8)
        return Image.fromarray(g, mode="L")

    # -------- Zoom & Pan --------

    def _zoomAtCenter(self, factor: float):
        self._scale *= factor
        self._scale = max(0.1, min(self._scale, 20.0))
        self._render()

    def _onWheel(self, event):
        # Normalize delta
        delta = 1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else -1
        factor = 1.1 if delta > 0 else 0.9
        self._zoomAtCenter(factor)

    def _updatePanMode(self):
        self._panActive = bool(self.panVar.get())

    def _onPanStart(self, event):
        self._panActive = True
        self._lastDrag = (event.x, event.y)

    def _onPanDrag(self, event):
        if not self._panActive or self._lastDrag is None:
            return
        dx = event.x - self._lastDrag[0]
        dy = event.y - self._lastDrag[1]
        self._offset += np.array([dx, dy], dtype=float)
        self._lastDrag = (event.x, event.y)
        self._render()

    def _onPanEnd(self, event):
        self._lastDrag = None
        if not self.panVar.get():
            self._panActive = False

    # Left+Right combined as pan
    def _onButtonComboDown(self, event):
        state = event.state
        left_down = (state & 0x100) != 0 or event.num == 1
        right_down = (state & 0x400) != 0 or event.num == 3
        if left_down and right_down:
            self._panActive = True
            self._lastDrag = (event.x, event.y)

    def _onComboDrag(self, event):
        if self._panActive and self._lastDrag is not None:
            dx = event.x - self._lastDrag[0]
            dy = event.y - self._lastDrag[1]
            self._offset += np.array([dx, dy], dtype=float)
            self._lastDrag = (event.x, event.y)
            self._render()

    def _onButtonComboUp(self, event):
        self._lastDrag = None
        if not self.panVar.get():
            self._panActive = False

    # -------- Crop overlay --------

    def _startCrop(self):
        self._cropMode = True
        self._calibMode = False
        self._cropRect = None
        self._calibStart = None
        self._calibEnd = None
        self._render()

    def _applyCropDialog(self):
        if not self._cropRect:
            messagebox.showwarning("Crop", "Draw a crop box first (Crop button, then drag).")
            return

        # Confirm dialog
        apply_all = messagebox.askyesno(
            "Apply Crop",
            "Apply this crop to ALL images?\n\nYes = all images\nNo = current image only"
        )

        useMargins = self.useMarginsVar.get()

        # Compute rect in image coordinates
        rect_img = self._canvasRectToImageRect(self._cropRect)
        if rect_img is None:
            messagebox.showwarning("Crop", "Crop is out of bounds or too small.")
            return

        # Apply
        if apply_all:
            newImages = applyCropBatch(self.images, rect=rect_img, margins=None, useMargins=useMargins)
            # Replace only successful crops
            for i, cropped in enumerate(newImages):
                if cropped is not None:
                    self.images[i] = cropped
        else:
            img = self.images[self.index]
            rect_img = clampRectToImage(rect_img, img.shape)
            if rect_img is None:
                messagebox.showwarning("Crop", "Crop is invalid for this image.")
                return
            self.images[self.index] = cropWithRect(img, rect_img)

        self.onImagesUpdated(self.images)
        self._cropMode = False
        self._cropRect = None
        self._render()
        messagebox.showinfo("Crop", "Crop applied.")

    def _drawCropOverlay(self):
        if not self._cropMode or self._cropRect is None:
            return
        x0, y0, x1, y1 = self._cropRect
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00ff88", width=2, dash=(4, 2))
        r = 5
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1),
                       ((x0+x1)//2, y0), ((x0+x1)//2, y1), (x0, (y0+y1)//2), (x1, (y0+y1)//2)):
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#00ff88", fill="", width=2)

    def _onCropPress(self, event):
        if not self._cropMode:
            return
        self._cropRect = (event.x, event.y, event.x, event.y)
        self._render()

    def _onCropDrag(self, event):
        if not self._cropMode or self._cropRect is None:
            return
        x0, y0, _, _ = self._cropRect
        self._cropRect = (min(x0, event.x), min(y0, event.y), max(x0, event.x), max(y0, event.y))
        self._render()

    def _onCropRelease(self, event):
        if not self._cropMode or self._cropRect is None:
            return
        # clamp to canvas bounds
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        x0, y0, x1, y1 = self._cropRect
        x0 = max(0, min(x0, cw - 1))
        y0 = max(0, min(y0, ch - 1))
        x1 = max(1, min(x1, cw))
        y1 = max(1, min(y1, ch))
        if x1 - x0 < 3 or y1 - y0 < 3:
            self._cropRect = None
        else:
            self._cropRect = (x0, y0, x1, y1)
        self._render()

    def _onCropConfirm(self, event):
        if self._cropMode and self._cropRect is not None:
            # convenience: double-click confirms then opens apply dialog
            self._applyCropDialog()

    def _canvasRectToImageRect(self, rect: Rect) -> Optional[Rect]:
        """Convert a canvas rect to image pixel rect, accounting for scale/offset/fit."""
        if self._pil is None or rect is None:
            return None
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        iw, ih = self._pil.size

        baseScale = min(cw / float(iw), ch / float(ih))
        baseScale = min(baseScale, 1.0)
        s = baseScale * self._scale
        w = int(round(iw * s))
        h = int(round(ih * s))
        ox = int((cw - w) / 2 + self._offset[0])
        oy = int((ch - h) / 2 + self._offset[1])

        x0c, y0c, x1c, y1c = rect
        # Convert canvas coords to image coords
        x0i = int((x0c - ox) / max(s, 1e-6))
        y0i = int((y0c - oy) / max(s, 1e-6))
        x1i = int((x1c - ox) / max(s, 1e-6))
        y1i = int((y1c - oy) / max(s, 1e-6))

        # Clamp to image bounds
        x0i, x1i = min(x0i, x1i), max(x0i, x1i)
        y0i, y1i = min(y0i, y1i), max(y0i, y1i)
        r = clampRectToImage((x0i, y0i, x1i, y1i), (ih, iw))
        return r

    # -------- Calibration --------

    def _startCalibration(self):
        self._calibMode = True
        self._cropMode = False
        self._calibStart = None
        self._calibEnd = None
        self._calibPxDist = None
        self._render()
        messagebox.showinfo(
            "Calibration",
            "Ctrl-click the start point, then Ctrl-click the second point.\n"
            "The second point will be constrained to the same vertical pixel (same y).\n"
            "After the second Ctrl-click, you'll enter the real length and units."
        )


    def _onCalibPress(self, event):
        if not self._calibMode:
            return

        # Convert canvas->image coords
        pt = self._canvasToImagePoint(event.x, event.y)
        if pt is None:
            return
        x, y = pt

        if self._calibStart is None:
            self._calibStart = (x, y)
            self._calibEnd = None
        else:
            # finalize with constrained y
            x1, y1 = self._calibStart
            x2 = x
            y2 = y1  # same vertical pixel
            self._calibEnd = (x2, y2)
            self._calibPxDist = abs(x2 - x1)
            self._render()
            self._calibrationDialog()  # prompt for real length & units

    def _onCalibMotion(self, event):
        if not self._calibMode:
            return
        if self._calibStart is None:
            return
        pt = self._canvasToImagePoint(event.x, event.y)
        if pt is None:
            return
        x, y = pt
        x1, y1 = self._calibStart
        # live endpoint constrained to same y
        self._calibEnd = (x, y1)
        self._calibPxDist = abs(x - x1)
        self._render()

    def _onCalibRelease(self, event):
        # nothing extra; finalize handled on press of second point
        pass

    def _drawCalibrationOverlay(self):
        if not self._calibMode:
            return
        if self._calibStart is None:
            return

        # Convert image points back to canvas space for drawing
        a = self._imageToCanvasPoint(*self._calibStart)
        b = self._imageToCanvasPoint(*self._calibEnd) if self._calibEnd else None
        if a is None:
            return
        x0c, y0c = a

        # draw start point
        r = 4
        self.canvas.create_oval(x0c-r, y0c-r, x0c+r, y0c+r, outline="#ffaa00", width=2)

        if b is not None:
            x1c, y1c = b
            self.canvas.create_line(x0c, y0c, x1c, y1c, fill="#ffaa00", width=2)
            # label
            if self._calibPxDist is not None:
                txt = f"{int(round(self._calibPxDist))} px"
                self.canvas.create_text((x0c + x1c)//2, y0c - 12, text=txt, fill="#ffaa00", anchor="s")

    def _calibrationDialog(self):
        if self._calibPxDist is None or self._calibStart is None or self._calibEnd is None:
            return
        px = float(self._calibPxDist)

        # Simple dialog: real length + units + apply to all?
        dlg = tk.Toplevel(self)
        dlg.title("Calibration")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text=f"Measured distance: {px:.3f} px").grid(row=0, column=0, columnspan=4, padx=8, pady=(10, 6), sticky="w")

        ttk.Label(dlg, text="Enter real length:").grid(row=1, column=0, padx=8, pady=4, sticky="e")
        lengthVar = tk.DoubleVar(value=1.0)
        ttk.Entry(dlg, textvariable=lengthVar, width=10).grid(row=1, column=1, padx=4, pady=4, sticky="w")

        ttk.Label(dlg, text="Units:").grid(row=1, column=2, padx=8, pady=4, sticky="e")
        unitVar = tk.StringVar(value="mm")
        ttk.Entry(dlg, textvariable=unitVar, width=8).grid(row=1, column=3, padx=4, pady=4, sticky="w")

        applyAllVar = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg, text="Apply to all images", variable=applyAllVar).grid(row=2, column=0, columnspan=4, padx=8, pady=6, sticky="w")

        # Buttons
        btns = ttk.Frame(dlg); btns.grid(row=3, column=0, columnspan=4, pady=(6, 10))
        def on_ok():
            try:
                realLen = float(lengthVar.get())
                unitName = unitVar.get().strip() or "mm"
                if realLen <= 0:
                    raise ValueError
            except Exception:
                messagebox.showerror("Calibration", "Please enter a positive real length.")
                return
            unitsPerPx = realLen / max(px, 1e-9)  # units per pixel
            self._setScale(unitsPerPx, unitName, applyAllVar.get())
            dlg.destroy()
        def on_cancel():
            dlg.destroy()

        ttk.Button(btns, text="OK", command=on_ok).pack(side="left", padx=8)
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="left", padx=8)

        dlg.wait_window()

        # Exit calibration mode after dialog
        self._calibMode = False
        self._calibStart = None
        self._calibEnd = None
        self._calibPxDist = None
        self._render()

    def _manualScaleDialog(self):
        # Dialog to set scale either as units/px or px/unit, with units; apply to current or all
        dlg = tk.Toplevel(self)
        dlg.title("Set Scale")
        dlg.transient(self)
        dlg.grab_set()

        modeVar = tk.StringVar(value="unitsPerPx")  # or "pxPerUnit"
        ttk.Radiobutton(dlg, text="Units per pixel", value="unitsPerPx", variable=modeVar).grid(row=0, column=0, padx=8, pady=(10,4), sticky="w")
        unitsPerPxVar = tk.DoubleVar(value=0.005)
        ttk.Entry(dlg, textvariable=unitsPerPxVar, width=10).grid(row=0, column=1, padx=4, pady=(10,4), sticky="w")

        ttk.Radiobutton(dlg, text="Pixels per unit", value="pxPerUnit", variable=modeVar).grid(row=1, column=0, padx=8, pady=4, sticky="w")
        pxPerUnitVar = tk.DoubleVar(value=200.0)
        ttk.Entry(dlg, textvariable=pxPerUnitVar, width=10).grid(row=1, column=1, padx=4, pady=4, sticky="w")

        ttk.Label(dlg, text="Units:").grid(row=2, column=0, padx=8, pady=4, sticky="e")
        unitVar = tk.StringVar(value="mm")
        ttk.Entry(dlg, textvariable=unitVar, width=8).grid(row=2, column=1, padx=4, pady=4, sticky="w")

        applyAllVar = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg, text="Apply to all images", variable=applyAllVar).grid(row=3, column=0, columnspan=2, padx=8, pady=6, sticky="w")

        btns = ttk.Frame(dlg); btns.grid(row=4, column=0, columnspan=2, pady=(6, 10))
        def on_ok():
            unitName = unitVar.get().strip() or "mm"
            mode = modeVar.get()
            try:
                if mode == "unitsPerPx":
                    val = float(unitsPerPxVar.get())
                    if val <= 0:
                        raise ValueError
                    unitsPerPx = val
                else:
                    val = float(pxPerUnitVar.get())
                    if val <= 0:
                        raise ValueError
                    unitsPerPx = 1.0 / val
            except Exception:
                messagebox.showerror("Set Scale", "Please enter a positive value.")
                return
            self._setScale(unitsPerPx, unitName, applyAllVar.get())
            dlg.destroy()
        def on_cancel():
            dlg.destroy()

        ttk.Button(btns, text="OK", command=on_ok).pack(side="left", padx=8)
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="left", padx=8)

        dlg.wait_window()

    def _setScale(self, unitsPerPx: float, unitName: str, applyAll: bool):
        # Save into scales list
        if applyAll:
            for i in range(len(self.scales)):
                self.scales[i] = {"unitsPerPx": float(unitsPerPx), "unitName": unitName}
        else:
            self.scales[self.index] = {"unitsPerPx": float(unitsPerPx), "unitName": unitName}

        # Callback to parent
        self.onScalesUpdated(self.scales)
        self._updateScaleStatus()
        self._render()
        messagebox.showinfo("Scale", f"Scale set to {unitsPerPx:.6g} {unitName}/px" + (" (applied to all)" if applyAll else ""))

    def _updateScaleStatus(self):
        s = self.scales[self.index] if (0 <= self.index < len(self.scales)) else None
        if s and "unitsPerPx" in s and "unitName" in s:
            self.calibStatusVar.set(f"Scale: {s['unitsPerPx']:.6g} {s['unitName']}/px")
        else:
            self.calibStatusVar.set("Scale: not set")

    # -------- Navigation --------

    def _prev(self):
        if self.index > 0:
            self.index -= 1
            self._cropMode = False
            self._calibMode = False
            self._calibStart = None
            self._calibEnd = None
            self._loadCurrent()

    def _next(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self._cropMode = False
            self._calibMode = False
            self._calibStart = None
            self._calibEnd = None
            self._loadCurrent()

    # -------- Helpers: coordinate transforms --------

    def _currentViewParams(self):
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        iw, ih = self._pil.size if self._pil is not None else (1, 1)
        baseScale = min(cw / float(iw), ch / float(ih))
        baseScale = min(baseScale, 1.0)
        s = baseScale * self._scale
        w = int(round(iw * s))
        h = int(round(ih * s))
        ox = int((cw - w) / 2 + self._offset[0])
        oy = int((ch - h) / 2 + self._offset[1])
        return s, ox, oy

    def _canvasToImagePoint(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        if self._pil is None:
            return None
        s, ox, oy = self._currentViewParams()
        iw, ih = self._pil.size
        ix = int((x - ox) / max(s, 1e-6))
        iy = int((y - oy) / max(s, 1e-6))
        if 0 <= ix < iw and 0 <= iy < ih:
            return ix, iy
        return None

    def _imageToCanvasPoint(self, ix: int, iy: int) -> Optional[Tuple[int, int]]:
        if self._pil is None:
            return None
        s, ox, oy = self._currentViewParams()
        x = int(ix * s + ox)
        y = int(iy * s + oy)
        return x, y


def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        style.configure("TButton", padding=4)
        style.configure("TLabel", padding=2)
    except Exception:
        pass
    app = PreprocessApp(root)
    root.geometry("1400x900")
    root.minsize(900, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
