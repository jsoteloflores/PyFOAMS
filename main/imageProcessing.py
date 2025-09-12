"""
Image processing utilities for PyFOAMS.

This module provides functions to load, preprocess and segment grayscale images of
vesicular samples.  The functions are designed to be reusable both from the
graphical user interface (GUI) and in headless scripts for batch processing.

Key functionality includes:

* Loading images as 8‑bit grayscale arrays.
* Optional cropping of images to remove annotations or scale bars.
* Robust thresholding using Otsu's method with automatic polarity detection or
  manual/percentile thresholds.  Preprocessing (median blur, Gaussian blur
  and CLAHE) is applied before thresholding to improve robustness to noise and
  uneven illumination【332621880016666†L124-L147】.
* Morphological cleaning via opening and closing to remove speckle noise and
  fill small gaps【523771896140784†screenshot】.
* Watershed‑based separation of touching vesicles using a distance transform to
  generate seed markers【225933462397427†L42-L55】.
* Convenience functions for eroding binary masks and cropping arrays.

The default behaviour aims to produce a binary mask where vesicles are white
(255) on a black (0) background.  The thresholding logic will automatically
invert the mask if the majority of edge pixels are detected as foreground.

Example usage:

```
from imageProcessing import loadGreyscaleImage, thresholdImage, cleanBinary, separateVesicles

img = loadGreyscaleImage("sample.tif")
binary = thresholdImage(img, method="otsu")
cleaned = cleanBinary(binary, open_size=3, close_size=3)
separated = separateVesicles(cleaned, dist_threshold=0.7)
```

This pipeline will produce a cleaned, separated binary mask suitable for
contour extraction and measurement.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

def loadGreyscaleImage(path: str) -> np.ndarray:
    """Load an image from disk as an 8‑bit grayscale numpy array.

    Parameters
    ----------
    path : str
        The filesystem path to an image file.

    Returns
    -------
    np.ndarray
        A two–dimensional array of dtype ``uint8`` representing the grayscale
        image.  Raises ``FileNotFoundError`` if the file cannot be loaded.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def cropImage(img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop the image to a rectangular region of interest (ROI).

    The ROI is specified as ``(x, y, w, h)`` where (x, y) gives the top–left
    corner and (w, h) specify the width and height.  Coordinates outside the
    image bounds are clamped.

    Parameters
    ----------
    img : np.ndarray
        The input grayscale image.
    roi : tuple of int
        (x, y, width, height) of the crop rectangle.

    Returns
    -------
    np.ndarray
        The cropped image.
    """
    h, w = img.shape[:2]
    x, y, width, height = roi
    # Clamp coordinates to image extents
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + width)
    y1 = min(h, y + height)
    if x1 <= x0 or y1 <= y0:
        return img.copy()
    return img[y0:y1, x0:x1]


def _apply_clahe(img: np.ndarray, clipLimit: float = 2.0, tileGridSize: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to improve local contrast.

    CLAHE operates on small tiles of the image and rescales each tile's
    histogram independently, which helps mitigate illumination gradients.  It is
    particularly useful prior to automatic thresholding【332621880016666†L124-L147】.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image.
    clipLimit : float, optional
        Controls the contrast limiting.  Higher values allow more contrast.  The
        default is 2.0.
    tileGridSize : tuple of int, optional
        Size of the grid for the histogram equalisation.  The default is
        (8, 8).

    Returns
    -------
    np.ndarray
        Image with enhanced local contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)


def thresholdImage(
    img: np.ndarray,
    method: str = "otsu",
    manualThresh: Optional[int] = None,
    *,
    fg_threshold: Optional[float] = None,
    apply_clahe: bool = True,
) -> np.ndarray:
    """Convert a grayscale image to a binary mask using robust thresholding.

    This function performs several preprocessing steps before applying a
    threshold.  Images are first blurred to reduce noise and then optionally
    enhanced with CLAHE for better contrast.  Thresholding can then be
    performed either automatically (Otsu's method) or manually.  If the image
    histogram is dominated by a single mode (e.g. uneven lighting), a
    percentile threshold can be supplied via ``fg_threshold``.  After
    thresholding, the result is automatically inverted if more than half of
    the border pixels are classified as foreground.  The returned mask has
    dtype ``uint8`` and values {0,255}, with vesicles as 255 (white).

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image.
    method : str, optional
        Either ``"otsu"`` for Otsu's automatic threshold or ``"manual"``
        to use the supplied ``manualThresh``.  Default is ``"otsu"``.
    manualThresh : int, optional
        Threshold value (0–255) to use when ``method="manual"``.  Ignored
        otherwise.
    fg_threshold : float, optional
        Fraction between 0 and 1 specifying the percentile of pixel
        intensities to treat as foreground.  If provided, this overrides
        Otsu's threshold.  For example, ``0.1`` selects approximately the
        darkest 10% of pixels as foreground if pores are dark.  The same
        fraction is applied for bright pores when inverted.  Default is
        ``None`` (disabled).
    apply_clahe : bool, optional
        If ``True`` (default), apply CLAHE to improve local contrast before
        thresholding.  Set to ``False`` if images are already well
        illuminated.

    Returns
    -------
    np.ndarray
        Binary image where vesicles are white (255) and background is black (0).
    """
    # Copy to avoid modifying caller data
    proc = img.copy()
    # Apply median and Gaussian blurs to suppress noise
    proc = cv2.medianBlur(proc, 3)
    proc = cv2.GaussianBlur(proc, (5, 5), 0)
    # Optional CLAHE for local contrast enhancement
    if apply_clahe:
        proc = _apply_clahe(proc)

    invert = False
    binary: np.ndarray
    thr_val = None
    # Determine threshold value and binary mask
    if method == "manual":
        if manualThresh is None:
            raise ValueError("manualThresh must be provided when method='manual'")
        thr_val = int(manualThresh)
        _, temp = cv2.threshold(proc, thr_val, 255, cv2.THRESH_BINARY)
    else:
        # Otsu or percentile threshold
        if fg_threshold is not None:
            # Clamp to [0,1]
            frac = max(0.0, min(1.0, float(fg_threshold)))
            thr_val = int(255 * frac)
            _, temp = cv2.threshold(proc, thr_val, 255, cv2.THRESH_BINARY)
        else:
            # Automatic Otsu threshold
            thr_val, temp = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine if the binary needs inversion: check border pixels
    # If more than half of the border pixels are white, invert mask
    border = np.concatenate([
        temp[0, :], temp[-1, :], temp[:, 0], temp[:, -1]
    ])
    white_ratio = np.mean(border == 255)
    if white_ratio > 0.5:
        invert = True

    if invert:
        _, binary = cv2.threshold(proc, thr_val, 255, cv2.THRESH_BINARY_INV)
    else:
        binary = temp

    # Ensure output is uint8 with values {0,255}
    return cv2.convertScaleAbs(binary)


def cleanBinary(binary: np.ndarray, *, open_size: int = 3, close_size: int = 3) -> np.ndarray:
    """Apply morphological opening and closing to clean a binary mask.

    Opening removes small white noise by eroding then dilating, while closing
    fills small black gaps inside objects by dilating then eroding【523771896140784†screenshot】.
    Both operations use a square structuring element.  The opening is applied
    first, followed by closing.

    Parameters
    ----------
    binary : np.ndarray
        Binary image where objects are white (255) on a black background (0).
    open_size : int, optional
        Size of the structuring element for the opening operation.  Default is
        3.
    close_size : int, optional
        Size of the structuring element for the closing operation.  Default is
        3.

    Returns
    -------
    np.ndarray
        Cleaned binary mask.
    """
    kernel_open = np.ones((open_size, open_size), np.uint8)
    kernel_close = np.ones((close_size, close_size), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    return closed


def erodeBinary(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Perform a single erosion on a binary mask.

    Erosion shrinks foreground regions and can help separate objects that are
    connected by thin bridges【523771896140784†screenshot】.  It also removes small
    foreground features entirely.  Use with caution: repeated erosions can
    erode away valid vesicles.

    Parameters
    ----------
    binary : np.ndarray
        Binary image where objects are white (255).
    kernel_size : int, optional
        Size of the square structuring element.  Default is 3.

    Returns
    -------
    np.ndarray
        Eroded binary mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return eroded


def separateVesicles(binary: np.ndarray, dist_threshold: float = 0.7) -> np.ndarray:
    """Separate touching vesicles in a binary mask using the watershed algorithm.

    The mask must contain vesicles as white (255) on a black background (0).
    A distance transform is computed on the foreground pixels.  Pixels whose
    distance is above a fraction ``dist_threshold`` of the maximum distance
    constitute seed markers【225933462397427†L42-L55】.  Watershed then floods from
    these markers to carve out individual vesicles.  Boundaries between
    vesicles are returned as black pixels.

    Parameters
    ----------
    binary : np.ndarray
        Binary image with vesicles as white (255).
    dist_threshold : float, optional
        Fraction (0–1) of the maximum distance value to use when selecting
        distance peaks as seed markers.  Default is 0.7.  Smaller values
        create more markers and thus more aggressive splitting; larger values
        produce fewer splits.

    Returns
    -------
    np.ndarray
        A binary image where separated vesicles are white (255) and background
        is black (0).  Regions where watershed assigned boundary pixels remain
        black.
    """
    # Ensure input is uint8 and objects are white
    binary = cv2.convertScaleAbs(binary)
    # Compute distance transform (only inside foreground)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_val = dist.max()
    if max_val == 0:
        return binary.copy()
    # Threshold to obtain markers; threshold relative to max distance
    _, seeds = cv2.threshold(dist, dist_threshold * max_val, 255, cv2.THRESH_BINARY)
    seeds = seeds.astype(np.uint8)
    # Label connected components of the seed image
    num_labels, markers = cv2.connectedComponents(seeds)
    # Add one so background is not 0
    markers = markers + 1
    # Mark unknown regions (background) as 0 for watershed
    markers[binary == 0] = 0
    # Prepare 3‑channel image for watershed
    color_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)
    # Create separated mask: watershed labels > 1 correspond to separated vesicles
    separated = np.zeros_like(binary)
    separated[markers > 1] = 255
    return separated


def process_images(
    image_paths: List[str],
    *,
    crop: Optional[Tuple[int, int, int, int]] = None,
    apply_clahe: bool = True,
    fg_threshold: Optional[float] = None,
    dist_threshold: float = 0.7,
) -> List[np.ndarray]:
    """Process a list of image paths and return binary masks for each.

    This helper function loops over a list of image file paths, loads each
    image, optionally crops it, applies thresholding and cleaning, and
    separates touching objects.  The result is a list of binary masks with
    vesicles separated.  Use this in batch processing scripts rather than the
    GUI.

    Parameters
    ----------
    image_paths : list of str
        Paths to image files.
    crop : tuple of int, optional
        Uniform crop region applied to all images as (x, y, w, h).  If None,
        no cropping is performed.  Default is None.
    apply_clahe : bool, optional
        If True (default), apply CLAHE for local contrast before thresholding.
    fg_threshold : float, optional
        Percentile threshold fraction passed through to ``thresholdImage``.
    dist_threshold : float, optional
        Distance threshold fraction passed to ``separateVesicles``.  Default
        is 0.7.

    Returns
    -------
    list of np.ndarray
        List of separated binary masks.
    """
    masks = []
    for path in image_paths:
        img = loadGreyscaleImage(path)
        if crop is not None:
            img = cropImage(img, crop)
        binary = thresholdImage(img, method="otsu", fg_threshold=fg_threshold, apply_clahe=apply_clahe)
        cleaned = cleanBinary(binary)
        separated = separateVesicles(cleaned, dist_threshold=dist_threshold)
        masks.append(separated)
    return masks

import cv2
import numpy as np

def _forceOdd(k: int) -> int:
    k = int(max(1, k))
    return k if k % 2 == 1 else k + 1

def thresholdImageAdvanced(
    gray,
    method: str = "otsu",              # "otsu" | "adaptive" | "percentile" | "pick"
    polarity: str = "auto",            # "auto" | "poresDarker" | "poresBrighter"
    useCLAHE: bool = False,
    claheClip: float = 2.0,
    claheTile: int = 8,                # tileGridSize
    medianK: int = 3,                  # 0 to disable, else odd >=3
    gaussianK: int = 0,                # 0 to disable, else odd >=3
    adaptiveBlock: int = 51,           # odd >=3
    adaptiveC: int = 2,
    percentile: float = 50.0,          # 0..100
    pickValue: int | None = None,      # required for "pick"
    pickTolerance: int = 10,           # +/- tolerance for "pick"
    applyOpenClose: bool = False,      # optional quick cleanup for preview
    morphK: int = 3
):
    """
    Robust thresholding wrapper with preprocessing and polarity control.
    Returns (binary_uint8, meta_dict).
    """
    if gray is None:
        raise ValueError("thresholdImageAdvanced: gray is None")
    img = gray.copy()

    # Ensure 8-bit grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Preprocessing
    if medianK and medianK >= 3:
        img = cv2.medianBlur(img, _forceOdd(medianK))
    if gaussianK and gaussianK >= 3:
        img = cv2.GaussianBlur(img, (_forceOdd(gaussianK), _forceOdd(gaussianK)), 0)
    if useCLAHE:
        clahe = cv2.createCLAHE(clipLimit=float(claheClip), tileGridSize=(int(claheTile), int(claheTile)))
        img = clahe.apply(img)

    meta = {"method": method, "polarity": polarity}

    # Thresholding modes
    if method == "otsu":
        # Try normal first
        _, binA = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Inverted version
        _, binB = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = _choosePolarity(img, binA, binB, polarity)
        meta["otsuThresh"] = _
    elif method == "adaptive":
        blk = max(3, _forceOdd(adaptiveBlock))
        # THRESH_BINARY means brighter->FG; INV means darker->FG
        binA = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, adaptiveC)
        binB = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, adaptiveC)
        binary = _choosePolarity(img, binA, binB, polarity)
        meta["block"] = blk
        meta["C"] = adaptiveC
    elif method == "percentile":
        t = np.clip(percentile, 0, 100)
        thresh = float(np.percentile(img, t))
        _, binA = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        _, binB = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
        binary = _choosePolarity(img, binA, binB, polarity)
        meta["percentile"] = float(t)
        meta["thresh"] = float(thresh)
    elif method == "pick":
        if pickValue is None:
            raise ValueError("thresholdImageAdvanced(method='pick'): pickValue is required")
        tol = int(max(0, pickTolerance))
        lo = max(0, int(pickValue) - tol)
        hi = min(255, int(pickValue) + tol)
        mask = (img >= lo) & (img <= hi)
        binary = (mask.astype(np.uint8) * 255)
        meta["pickValue"] = int(pickValue)
        meta["pickTolerance"] = tol
        # No polarity concept here (we directly define FG)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Optional quick morph cleanup (mostly for nicer preview)
    if applyOpenClose:
        k = max(1, int(morphK))
        kernel = np.ones((_forceOdd(k), _forceOdd(k)), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary.astype(np.uint8), meta

def _choosePolarity(img, binA, binB, polarity: str):
    """Decide which binary to use based on requested polarity."""
    if polarity == "poresDarker":
        # foreground should be darker than background → choose the mask whose FG mean is lower
        return _pickDarkerForeground(img, binA, binB)
    if polarity == "poresBrighter":
        return _pickBrighterForeground(img, binA, binB)
    # auto: prefer darker-foreground if it makes sense; otherwise brighter
    # We test both and choose where |mean_fg - mean_bg| is larger, biased to darker
    A_darker = _isDarkerForeground(img, binA)
    B_darker = _isDarkerForeground(img, binB)
    if A_darker and not B_darker:
        return binA
    if B_darker and not A_darker:
        return binB
    # both or neither → pick by contrast separation
    cA = _contrastGap(img, binA)
    cB = _contrastGap(img, binB)
    return binA if cA >= cB else binB

def _isDarkerForeground(img, binary):
    fg = img[binary == 255]
    bg = img[binary == 0]
    if fg.size == 0 or bg.size == 0:
        return False
    return float(np.mean(fg)) < float(np.mean(bg))

def _contrastGap(img, binary):
    fg = img[binary == 255]
    bg = img[binary == 0]
    if fg.size == 0 or bg.size == 0:
        return 0.0
    return abs(float(np.mean(fg)) - float(np.mean(bg)))

def _pickDarkerForeground(img, binA, binB):
    return binA if _isDarkerForeground(img, binA) else binB

def _pickBrighterForeground(img, binA, binB):
    # pick the orientation where FG mean > BG mean
    return binA if not _isDarkerForeground(img, binA) else binB
