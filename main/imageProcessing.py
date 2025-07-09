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
