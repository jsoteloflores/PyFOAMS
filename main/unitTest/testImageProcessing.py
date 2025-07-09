# main/unitTest/testImageProcessing.py

import unittest
import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imageProcessing import loadGreyscaleImage, thresholdImage, cleanBinary

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        self.testImg = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(self.testImg, (50, 50), 20, 200, -1)
        self.testPath = "dummy_test_image.png"
        cv2.imwrite(self.testPath, self.testImg)

    def tearDown(self):
        if os.path.exists(self.testPath):
            os.remove(self.testPath)

    def testLoadGreyscaleImage(self):
        img = loadGreyscaleImage(self.testPath)
        self.assertEqual(img.shape, (100, 100))
        self.assertEqual(img.dtype, np.uint8)

    def testThresholdImageOtsu(self):
        binary = thresholdImage(self.testImg, method="otsu")
        uniqueVals = np.unique(binary)
        self.assertTrue(set(uniqueVals).issubset({0, 255}))

    def testCleanBinary(self):
        noisy = self.testImg.copy()
        noisy[10, 10] = 255  # simulate noise
        binary = thresholdImage(noisy, method="otsu")
        cleaned = cleanBinary(binary)
        self.assertEqual(cleaned.dtype, np.uint8)

if __name__ == "__main__":
    unittest.main()
