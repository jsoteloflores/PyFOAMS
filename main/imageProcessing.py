# main/imageProcessing.py

import cv2
import numpy as np

def loadGreyscaleImage(path):
    """Load image as grayscale"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def thresholdImage(img, method="otsu", manualThresh=None):
    """Convert grayscale image to binary using thresholding"""
    if method == "otsu":
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "manual" and manualThresh is not None:
        _, binary = cv2.threshold(img, manualThresh, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("Invalid thresholding method or missing manual threshold.")
    return binary

def cleanBinary(binary, kernelSize=3):
    """Apply morphological opening to remove noise"""
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cleaned

def separateVesicles(binary):
    """Apply distance transform and watershed to separate touching vesicles"""
    # Sure foreground area
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Sure background area
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)

    # Convert to clean binary mask again
    separated = np.zeros_like(binary)
    separated[markers > 1] = 255
    return separated
