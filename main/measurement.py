import cv2
import numpy as np

def findContours(binary):
    """Find contours in a binary image with refined preprocessing"""
    # Apply Gaussian smoothing to reduce noise
    smoothed = cv2.GaussianBlur(binary, (5, 5), 0)

    # Use morphological closing to fill gaps
    closed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Detect external contours only
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and circularity
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20 and isCircular(cnt)]

    return filtered_contours

def isCircular(contour):
    """Check if a contour is approximately circular"""
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return 0.7 <= circularity <= 1.2

def measureVesicles(contours, pixelScale=1.0):
    """
    Measure vesicle properties from contours.
    pixelScale: pixels per mm (used to convert to physical units)
    Returns a list of dicts for each vesicle.
    """
    vesicleData = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area == 0:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0
        equivalentDiameter = np.sqrt(4 * area / np.pi)

        vesicleData.append({
            "area": area / (pixelScale ** 2),
            "perimeter": perimeter / pixelScale,
            "circularity": circularity,
            "equivalentDiameter": equivalentDiameter / pixelScale,
            "boundingBox": cv2.boundingRect(cnt)
        })

    return vesicleData

