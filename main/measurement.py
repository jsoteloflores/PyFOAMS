import cv2
import numpy as np

def findContours(binaryImage):
    """Find contours of vesicles in binary image."""
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

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
