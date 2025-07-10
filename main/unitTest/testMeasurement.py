import unittest
import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from measurement import findContours, measureVesicles

class TestMeasurement(unittest.TestCase):

    def setUp(self):
        self.testBinary = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(self.testBinary, (50, 50), 20, 255, -1)  # white vesicle
        self.pixelScale = 1000  # e.g. 1000 pixels/mm

    def testFindContours(self):
        contours = findContours(self.testBinary)
        self.assertTrue(len(contours) > 0)

    def testMeasureVesicles(self):
        contours = findContours(self.testBinary)
        data = measureVesicles(contours, pixelScale=self.pixelScale)
        self.assertTrue(len(data) > 0)
        for vesicle in data:
            self.assertIn("area", vesicle)
            self.assertIn("circularity", vesicle)
            self.assertGreater(vesicle["area"], 0)

if __name__ == "__main__":
    unittest.main()
