# main/imageProcessing.py

import cv2
import numpy as np
import os

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

def cleanBinary(binary, *, kernel_size=3):
    """Apply morphological opening to remove noise"""
    import cv2
    import numpy as np
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cleaned


def separateVesicles(binary):
    """Apply distance transform and watershed to separate touching vesicles"""
    # Compute distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Measure average equivalent diameter (D_eq)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    avg_diameter = np.sqrt(4 * np.mean(areas) / np.pi) if areas else 0

    # Threshold seed points at 0.6 * D_eq
    seed_threshold = 0.6 * avg_diameter
    _, markers = cv2.threshold(dist_transform, seed_threshold, 255, cv2.THRESH_BINARY)

    # Convert markers to integer type
    markers = np.uint8(markers)

    # Perform watershed
    markers = cv2.connectedComponents(markers)[1]
    markers = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)

    # Create separated image
    separated = np.zeros_like(binary)
    separated[markers > 1] = 255

    return separated
