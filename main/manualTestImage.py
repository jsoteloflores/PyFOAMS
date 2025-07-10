# main/manualTestImage.py

import cv2
import os
from imageProcessing import loadGreyscaleImage, thresholdImage, cleanBinary
from measurement import findContours, measureVesicles

# 🔧 Change this to your test image path
imagePath = "main/unitTest/test_images/my_test_image.png"
outputDir = "main/unitTest/outputs"
os.makedirs(outputDir, exist_ok=True)

# Step 1: Load and preprocess
grey = loadGreyscaleImage(imagePath)
cv2.imwrite(os.path.join(outputDir, "1_greyscale.png"), grey)

binary = thresholdImage(grey, method="otsu")
cv2.imwrite(os.path.join(outputDir, "2_binary.png"), binary)

cleaned = cleanBinary(binary)
cv2.imwrite(os.path.join(outputDir, "3_cleaned.png"), cleaned)

# Step 2: Measure vesicles
contours = findContours(cleaned)
vesicleData = measureVesicles(contours, pixelScale=1000)

# Step 3: Draw results
contourOverlay = cv2.cvtColor(grey.copy(), cv2.COLOR_GRAY2BGR)
cv2.drawContours(contourOverlay, contours, -1, (0, 255, 0), 1)
cv2.imwrite(os.path.join(outputDir, "4_contours.png"), contourOverlay)

boxed = cv2.cvtColor(cleaned.copy(), cv2.COLOR_GRAY2BGR)
for vesicle in vesicleData:
    x, y, w, h = vesicle["boundingBox"]
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 1)
cv2.imwrite(os.path.join(outputDir, "5_bounding_boxes.png"), boxed)

print(f"[✓] Processed image saved to: {outputDir}")
