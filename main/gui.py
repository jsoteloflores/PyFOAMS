"""
Graphical user interface for PyFOAMS segmentation and preprocessing.

This module defines the ``PyFOAMS_GUI`` class which provides an
interactive GUI for loading vesicle images, cropping unwanted regions,
automatically binarising them using robust thresholding, cleaning and
separating touching vesicles, and manually editing the binary masks
before final labelling.  The workflow is designed to mirror the
functionality of the original FOAMS package while adding conveniences
such as batch processing and automatic preprocessing.

Key features
------------

* **Multi‑image support:** load one or more images and step through
  them using "Prev" and "Next" buttons.  Each image maintains its
  own binary mask and edits.
* **Interactive cropping:** a traditional ROI cropping tool with
  draggable handles on the corners and edges, allowing the user to
  remove scale bars or annotations before binarisation.
* **Automatic thresholding:** on first use of an editing tool the
  current grayscale image is converted to a binary mask using Otsu's
  method, with preprocessing and automatic polarity selection.
* **Morphological cleaning:** options to remove small islands and
  erode the binary mask are provided.  These operations rely on
  morphological opening and erosion as described in the OpenCV
  morphological transformations tutorial【523771896140784†screenshot】.
* **Manual editing:** a paintbrush tool for drawing black pixels
  (adding to vesicle regions) and a bucket fill tool for flood‑filling
  connected regions with the opposite colour.  These allow the user
  to correct segmentation errors such as holes or noise.
* **Labelled output:** after editing, images can be saved as colour
  labelled masks where each vesicle has a unique random colour.  The
  labelling uses a distance transform and watershed to separate
  touching vesicles followed by connected component analysis.

The GUI makes use of the functions defined in ``main.imageProcessing``
for thresholding, cleaning, erosion and watershed separation.

Note: this file replaces the minimal GUI skeleton present in the
original PyFOAMS repository.  It represents a substantial
functional upgrade intended to support both automated and user‑assisted
segmentation workflows.
"""

import os
import random
import tkinter as tk
from tkinter import filedialog, ttk
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

import imageProcessing

from imageProcessing import thresholdImageAdvanced
from PIL import Image, ImageTk

# --- Pillow (ANTIALIAS removal) compatibility ---
if hasattr(Image, "Resampling"):  # Pillow >= 9.1 (incl. 10+)
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:  # Older Pillow
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)
# ------------------------------------------------


class PyFOAMS_GUI:
    """Tkinter GUI for PyFOAMS segmentation and preprocessing."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("PyFOAMS – Segmentation and Preprocessing")
        # Lists to store original grayscale images and current binary masks
        self.images: List[np.ndarray] = []
        self.binaries: List[Optional[np.ndarray]] = []
        # Index of currently displayed image
        self.current_index: int = 0
        # Current drawing state for paintbrush
        self.drawing: bool = False
        self.last_x: int = 0
        self.last_y: int = 0
        self.brush_size: int = 5
        self.paint_color: int = 0  # black pixels when painting
        # Crop rectangle and handles
        self.crop_rect = None
        self.crop_handles = {}
        self.active_handle: Optional[str] = None
        # Set up UI: toolbar and canvas
        self._build_toolbar()
        self._build_canvas()
        # Default tool state
        self.current_tool: Optional[str] = None

    def _build_toolbar(self) -> None:
        """Create the toolbar with buttons for the various tools."""
        self.toolbar = ttk.Frame(self.master)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        # Load button for multiple images
        self.btnLoad = ttk.Button(self.toolbar, text="📂 Load Image(s)", command=self.loadImages)
        self.btnLoad.pack(side=tk.LEFT, padx=5, pady=5)
        # Crop tool
        self.btnCrop = ttk.Button(self.toolbar, text="Crop", command=self.activateCrop)
        self.btnCrop.pack(side=tk.LEFT, padx=5)
        # Paintbrush tool
        self.btnPaint = ttk.Button(self.toolbar, text="Paint", command=self.activatePaintbrush)
        self.btnPaint.pack(side=tk.LEFT, padx=5)
        # Bucket fill tool
        self.btnBucket = ttk.Button(self.toolbar, text="Bucket", command=self.activateBucket)
        self.btnBucket.pack(side=tk.LEFT, padx=5)
        # Process islands (remove small noise)
        self.btnIslands = ttk.Button(self.toolbar, text="Process Islands", command=self.processIslands)
        self.btnIslands.pack(side=tk.LEFT, padx=5)
        # Erode tool
        self.btnErode = ttk.Button(self.toolbar, text="Erode", command=self.activateErode)
        self.btnErode.pack(side=tk.LEFT, padx=5)
        # Navigation buttons
        self.btnPrev = ttk.Button(self.toolbar, text="◀ Prev", command=self.prevImage)
        self.btnPrev.pack(side=tk.RIGHT, padx=5)
        self.btnNext = ttk.Button(self.toolbar, text="Next ▶", command=self.nextImage)
        self.btnNext.pack(side=tk.RIGHT, padx=5)
        # Save labelled output
        self.btnSave = ttk.Button(self.toolbar, text="💾 Save Labeled", command=self.saveLabeledImages)
        self.btnSave.pack(side=tk.RIGHT, padx=5)
        # Initially disable navigation and editing buttons until images loaded
        for btn in (self.btnCrop, self.btnPaint, self.btnBucket, self.btnIslands, self.btnErode, self.btnPrev, self.btnNext, self.btnSave):
            btn.config(state=tk.DISABLED)

    def _build_canvas(self) -> None:
        """Create the image display canvas."""
        self.canvas = tk.Canvas(self.master, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Image loading and navigation
    # ------------------------------------------------------------------
    def loadImages(self) -> None:
        """Open a file dialog to select one or more image files."""
        filepaths = filedialog.askopenfilenames(
            title="Select Image(s)",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")],
        )
        if not filepaths:
            return
        # Load each image as grayscale using our helper
        self.images = []
        for path in filepaths:
            try:
                img = imageProcessing.loadGreyscaleImage(path)
            except FileNotFoundError:
                continue
            self.images.append(img)
        if not self.images:
            return
        # Initialise binary placeholders for each image
        self.binaries = [None] * len(self.images)
        self.current_index = 0
        # Update UI state
        # Enable tools now that images are loaded
        for btn in (self.btnCrop, self.btnPaint, self.btnBucket, self.btnIslands, self.btnErode, self.btnSave):
            btn.config(state=tk.NORMAL)
        # Enable navigation if more than one image
        self.btnPrev.config(state=tk.DISABLED)
        if len(self.images) > 1:
            self.btnNext.config(state=tk.NORMAL)
        else:
            self.btnNext.config(state=tk.DISABLED)
        # Display first image
        self.displayImage(self.images[0])

    def prevImage(self) -> None:
        """Display the previous image in the list."""
        if not self.images or self.current_index == 0:
            return
        self.current_index -= 1
        # Show binary if available, otherwise grayscale
        img = self.binaries[self.current_index] if self.binaries[self.current_index] is not None else self.images[self.current_index]
        self.displayImage(img)
        # Update navigation button states
        self.btnNext.config(state=tk.NORMAL)
        if self.current_index == 0:
            self.btnPrev.config(state=tk.DISABLED)

    def nextImage(self) -> None:
        """Display the next image in the list."""
        if not self.images or self.current_index >= len(self.images) - 1:
            return
        self.current_index += 1
        img = self.binaries[self.current_index] if self.binaries[self.current_index] is not None else self.images[self.current_index]
        self.displayImage(img)
        # Update navigation button states
        self.btnPrev.config(state=tk.NORMAL)
        if self.current_index == len(self.images) - 1:
            self.btnNext.config(state=tk.DISABLED)

    def _npToPil(self, arr):
        """
        Convert a NumPy image (grayscale/binary or BGR color) to a PIL Image.
        - Accepts uint8/uint16/float arrays; scales to 8-bit for display if needed.
        - Treats 2D arrays as grayscale; 3-channel arrays as BGR (OpenCV) -> RGB.
        """
        import numpy as np
        import cv2

        a = arr
        if a is None:
            raise ValueError("displayImage: received None array")

        # Ensure ndarray
        a = np.asarray(a)

        # If boolean, map to 0/255
        if a.dtype == np.bool_:
            a = a.astype("uint8") * 255

        # If not uint8, scale to 0..255 for visualization
        if a.dtype != np.uint8:
            a = a.astype("float32")
            vmin = float(a.min())
            vmax = float(a.max())
            if vmax > vmin:
                a = (255.0 * (a - vmin) / (vmax - vmin)).clip(0, 255).astype("uint8")
            else:
                a = np.zeros_like(a, dtype="uint8")

        # Grayscale vs color
        if a.ndim == 2:  # grayscale/binary
            pil_img = Image.fromarray(a, mode="L")
        elif a.ndim == 3:
            if a.shape[2] == 3:
                # Assume BGR (OpenCV) -> RGB
                a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(a_rgb, mode="RGB")
            elif a.shape[2] == 4:
                # BGRA -> RGBA
                a_rgba = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
                pil_img = Image.fromarray(a_rgba, mode="RGBA")
            else:
                # Fallback: collapse to grayscale
                import numpy as np
                gray = a.mean(axis=2).astype("uint8")
                pil_img = Image.fromarray(gray, mode="L")
        else:
            raise ValueError(f"Unsupported array shape for display: {a.shape}")

        return pil_img


    def displayImage(self, img_array, fit_to_canvas=True, allow_upscale=False, pad=0):
        """
        Render a NumPy image on self.canvas with safe Pillow resampling (no ANTIALIAS).
        - fit_to_canvas: if True, maintains aspect ratio to fit within canvas bounds.
        - allow_upscale: if False, never scales above 1:1.
        - pad: optional padding (pixels) when fitting.
        Keeps a reference to the PhotoImage to avoid GC.
        """
        # Convert to PIL
        pil = self._npToPil(img_array)

        # Make sure canvas geometry is up-to-date
        self.canvas.update_idletasks()
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        iw, ih = pil.size

        if fit_to_canvas:
            # Target area inside padding
            tw = max(1, cw - 2 * pad)
            th = max(1, ch - 2 * pad)
            # Scale ratio
            sx = tw / float(iw)
            sy = th / float(ih)
            s = min(sx, sy)
            if not allow_upscale:
                s = min(s, 1.0)
            new_w = max(1, int(round(iw * s)))
            new_h = max(1, int(round(ih * s)))
            if (new_w, new_h) != (iw, ih):
                pil = pil.resize((new_w, new_h), resample=RESAMPLE_LANCZOS)
        else:
            # 1:1 unless explicit upscaling requested elsewhere
            if not allow_upscale:
                # no resize
                pass
            # else: caller can resize first, then call display

        # Draw
        self.canvas.delete("all")
        photo = ImageTk.PhotoImage(pil)
        # Store reference to avoid garbage collection
        self._current_photo = photo
        self.canvas.create_image(pad, pad, anchor="nw", image=photo)
        # Optionally set scrollregion if you use scrollbars later
        self.canvas.configure(scrollregion=(0, 0, pil.size[0] + pad, pil.size[1] + pad))

    # ------------------------------------------------------------------
    # Cropping implementation
    # ------------------------------------------------------------------
    def activateCrop(self) -> None:
        """Activate interactive cropping mode."""
        if not self.images:
            return
        self.current_tool = "crop"
        # Remove any previous crop overlay
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        for handle_id in self.crop_handles.values():
            self.canvas.delete(handle_id)
        self.crop_handles.clear()
        # Bind events for ROI drawing and resizing
        self.canvas.bind("<Button-1>", self._onCropStart)
        self.canvas.bind("<B1-Motion>", self._onCropDrag)
        self.canvas.bind("<ButtonRelease-1>", self._onCropRelease)
        self.canvas.bind("<Double-Button-1>", self._onCropApply)

    def _onCropStart(self, event: tk.Event) -> None:
        # Start drawing a new ROI rectangle
        self.x0, self.y0 = event.x, event.y
        # Remove existing rect if any
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        for handle_id in self.crop_handles.values():
            self.canvas.delete(handle_id)
        self.crop_handles.clear()
        # Draw provisional rectangle
        self.crop_rect = self.canvas.create_rectangle(self.x0, self.y0, self.x0, self.y0, outline="red", dash=(2, 2), width=2)

    def _onCropDrag(self, event: tk.Event) -> None:
        # Update rectangle as mouse drags
        if self.crop_rect:
            self.canvas.coords(self.crop_rect, self.x0, self.y0, event.x, event.y)

    def _onCropRelease(self, event: tk.Event) -> None:
        # Finalise rectangle and draw handles
        if not self.crop_rect:
            return
        self.x1, self.y1 = event.x, event.y
        # Ensure coordinates are ordered
        if self.x0 > self.x1:
            self.x0, self.x1 = self.x1, self.x0
        if self.y0 > self.y1:
            self.y0, self.y1 = self.y1, self.y0
        # Draw eight handles (corners and midpoints)
        r = 4  # radius of handle circles
        positions = {
            "tl": (self.x0, self.y0),
            "tr": (self.x1, self.y0),
            "bl": (self.x0, self.y1),
            "br": (self.x1, self.y1),
            "top": ((self.x0 + self.x1) // 2, self.y0),
            "bottom": ((self.x0 + self.x1) // 2, self.y1),
            "left": (self.x0, (self.y0 + self.y1) // 2),
            "right": (self.x1, (self.y0 + self.y1) // 2),
        }
        for key, (cx, cy) in positions.items():
            handle = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="red", outline="black", tags=("handle", key))
            self.crop_handles[key] = handle
        # Rebind motion events to handle dragging
        self.canvas.bind("<Button-1>", self._onHandleClick)
        self.canvas.bind("<B1-Motion>", self._onHandleDrag)
        self.canvas.bind("<Double-Button-1>", self._onCropApply)

    def _onHandleClick(self, event: tk.Event) -> None:
        # Determine which handle is clicked
        tags = self.canvas.gettags("current")
        for t in tags:
            if t in ("tl", "tr", "bl", "br", "top", "bottom", "left", "right"):
                self.active_handle = t
                break

    def _onHandleDrag(self, event: tk.Event) -> None:
        # Adjust ROI coordinates based on dragged handle
        if not self.active_handle:
            return
        # Update coordinates; clamp to canvas bounds
        x, y = event.x, event.y
        if self.active_handle == "tl":
            self.x0, self.y0 = x, y
        elif self.active_handle == "tr":
            self.x1, self.y0 = x, y
        elif self.active_handle == "bl":
            self.x0, self.y1 = x, y
        elif self.active_handle == "br":
            self.x1, self.y1 = x, y
        elif self.active_handle == "top":
            self.y0 = y
        elif self.active_handle == "bottom":
            self.y1 = y
        elif self.active_handle == "left":
            self.x0 = x
        elif self.active_handle == "right":
            self.x1 = x
        # Normalize coordinates
        if self.x0 > self.x1:
            self.x0, self.x1 = self.x1, self.x0
        if self.y0 > self.y1:
            self.y0, self.y1 = self.y1, self.y0
        # Update rectangle
        self.canvas.coords(self.crop_rect, self.x0, self.y0, self.x1, self.y1)
        # Update handle positions
        r = 4
        # Corners
        self.canvas.coords(self.crop_handles["tl"], self.x0 - r, self.y0 - r, self.x0 + r, self.y0 + r)
        self.canvas.coords(self.crop_handles["tr"], self.x1 - r, self.y0 - r, self.x1 + r, self.y0 + r)
        self.canvas.coords(self.crop_handles["bl"], self.x0 - r, self.y1 - r, self.x0 + r, self.y1 + r)
        self.canvas.coords(self.crop_handles["br"], self.x1 - r, self.y1 - r, self.x1 + r, self.y1 + r)
        # Edges
        self.canvas.coords(self.crop_handles["top"], (self.x0 + self.x1) // 2 - r, self.y0 - r, (self.x0 + self.x1) // 2 + r, self.y0 + r)
        self.canvas.coords(self.crop_handles["bottom"], (self.x0 + self.x1) // 2 - r, self.y1 - r, (self.x0 + self.x1) // 2 + r, self.y1 + r)
        self.canvas.coords(self.crop_handles["left"], self.x0 - r, (self.y0 + self.y1) // 2 - r, self.x0 + r, (self.y0 + self.y1) // 2 + r)
        self.canvas.coords(self.crop_handles["right"], self.x1 - r, (self.y0 + self.y1) // 2 - r, self.x1 + r, (self.y0 + self.y1) // 2 + r)

    def _onCropApply(self, event: tk.Event) -> None:
        # Apply crop to the current image and reset editing state
        if not self.images or not hasattr(self, "x0") or not hasattr(self, "x1"):
            return
        # Convert canvas coords back to image coords
        current_img = self.images[self.current_index]
        h, w = current_img.shape[:2]
        canvas_w = self.canvas.winfo_width() or 1
        canvas_h = self.canvas.winfo_height() or 1
        # Determine scaling factor used in displayImage
        scale = min(canvas_w / w, canvas_h / h)
        # Map ROI to image coordinates
        img_x0 = int(self.x0 / scale)
        img_y0 = int(self.y0 / scale)
        img_x1 = int(self.x1 / scale)
        img_y1 = int(self.y1 / scale)
        # Clamp
        img_x0 = max(0, min(w, img_x0))
        img_y0 = max(0, min(h, img_y0))
        img_x1 = max(0, min(w, img_x1))
        img_y1 = max(0, min(h, img_y1))
        if img_x1 <= img_x0 or img_y1 <= img_y0:
            return
        # Crop grayscale and update lists
        cropped = current_img[img_y0:img_y1, img_x0:img_x1]
        self.images[self.current_index] = cropped
        self.binaries[self.current_index] = None  # reset binary since image changed
        # Clean up ROI overlay
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        for handle_id in self.crop_handles.values():
            self.canvas.delete(handle_id)
        self.crop_handles.clear()
        # Unbind cropping events
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.unbind("<Double-Button-1>")
        self.crop_rect = None
        self.active_handle = None
        # Display the cropped image
        self.displayImage(cropped)

    # ------------------------------------------------------------------
    # Thresholding helper
    # ------------------------------------------------------------------
    def ensureBinary(self) -> None:
        """Ensure the current image has a binary mask; threshold if needed."""
        if not self.images:
            return
        idx = self.current_index
        if self.binaries[idx] is None:
            # Apply robust thresholding; use Otsu by default
            gray = self.images[idx]
            binary = imageProcessing.thresholdImage(gray, method="otsu")
            # Store and display binary
            self.binaries[idx] = binary
            self.displayImage(binary)

    # ------------------------------------------------------------------
    # Manual editing tools
    # ------------------------------------------------------------------
    def activatePaintbrush(self) -> None:
        """Activate paintbrush mode to draw black pixels on the binary."""
        if not self.images:
            return
        self.ensureBinary()
        self.current_tool = "paint"
        self.canvas.bind("<Button-1>", self._startPaint)
        self.canvas.bind("<B1-Motion>", self._doPaint)
        self.canvas.bind("<ButtonRelease-1>", self._stopPaint)

    def _startPaint(self, event: tk.Event) -> None:
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def _doPaint(self, event: tk.Event) -> None:
        if not self.drawing:
            return
        x, y = event.x, event.y
        idx = self.current_index
        binary = self.binaries[idx]
        if binary is None:
            return
        # Map canvas coords to image coords
        img = binary
        h, w = img.shape[:2]
        canvas_w = self.canvas.winfo_width() or 1
        canvas_h = self.canvas.winfo_height() or 1
        scale = min(canvas_w / w, canvas_h / h)
        ix0, iy0 = int(self.last_x / scale), int(self.last_y / scale)
        ix1, iy1 = int(x / scale), int(y / scale)
        # Draw line on the binary array
        cv2.line(img, (ix0, iy0), (ix1, iy1), color=self.paint_color, thickness=self.brush_size)
        self.last_x, self.last_y = x, y
        # Update display
        self.displayImage(img)

    def _stopPaint(self, event: tk.Event) -> None:
        self.drawing = False

    def activateBucket(self) -> None:
        """Activate bucket fill mode to flood fill connected regions."""
        if not self.images:
            return
        self.ensureBinary()
        self.current_tool = "bucket"
        self.canvas.bind("<Button-1>", self._bucketFill)

    def _bucketFill(self, event: tk.Event) -> None:
        idx = self.current_index
        binary = self.binaries[idx]
        if binary is None:
            return
        # Map click to image coordinates
        h, w = binary.shape[:2]
        canvas_w = self.canvas.winfo_width() or 1
        canvas_h = self.canvas.winfo_height() or 1
        scale = min(canvas_w / w, canvas_h / h)
        ix, iy = int(event.x / scale), int(event.y / scale)
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return
        target_val = binary[iy, ix]
        # Determine fill value: invert pixel value
        new_val = 255 if target_val == 0 else 0
        # Flood fill using OpenCV
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(binary, mask, (ix, iy), new_val)
        self.binaries[idx] = binary
        self.displayImage(binary)

    def processIslands(self) -> None:
        """Remove small isolated islands from the binary mask."""
        if not self.images:
            return
        self.ensureBinary()
        idx = self.current_index
        binary = self.binaries[idx]
        if binary is None:
            return
        # Invert to treat small white speckles as objects, remove them via opening
        inv = cv2.bitwise_not(binary)
        cleaned = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self.binaries[idx] = cv2.bitwise_not(cleaned)
        self.displayImage(self.binaries[idx])

    def activateErode(self) -> None:
        """Erode the binary mask to shrink objects or separate thin connections."""
        if not self.images:
            return
        self.ensureBinary()
        idx = self.current_index
        binary = self.binaries[idx]
        if binary is None:
            return
        # Use our erode function (single iteration) from imageProcessing
        eroded = imageProcessing.erodeBinary(binary, kernel_size=3)
        self.binaries[idx] = eroded
        self.displayImage(eroded)

    # ------------------------------------------------------------------
    # Saving labelled outputs
    # ------------------------------------------------------------------
    def saveLabeledImages(self) -> None:
        """Label each binary image and save colour images for further analysis."""
        if not self.images:
            return
        # Ensure output directory exists
        out_dir = "LabeledOutputs"
        os.makedirs(out_dir, exist_ok=True)
        for idx, gray in enumerate(self.images):
            # Ensure binary exists
            if self.binaries[idx] is None:
                binary = imageProcessing.thresholdImage(gray, method="otsu")
            else:
                binary = self.binaries[idx]
            # Clean and separate touching objects
            cleaned = imageProcessing.cleanBinary(binary)
            separated = imageProcessing.separateVesicles(cleaned, dist_threshold=0.7)
            # If background is white (objects dark), invert
            if np.count_nonzero(separated == 0) < np.count_nonzero(separated == 255):
                separated = cv2.bitwise_not(separated)
            # Connected component labelling
            num_labels, labels = cv2.connectedComponents(separated)
            # Prepare random colours (skip label 0 which is background)
            h, w = labels.shape
            output = np.zeros((h, w, 3), dtype=np.uint8)
            # Predefine background as white for clarity
            output[labels == 0] = (255, 255, 255)
            for lab in range(1, num_labels):
                # Generate random colour
                colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                output[labels == lab] = colour
            # Save image
            out_path = os.path.join(out_dir, f"image_{idx + 1}_labeled.png")
            cv2.imwrite(out_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        print(f"Saved {len(self.images)} labelled images to {out_dir}")


def main() -> None:
    """Entry point for running the PyFOAMS GUI."""
    root = tk.Tk()
    app = PyFOAMS_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()