import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import os
import numpy as np
from imageProcessing import loadGreyscaleImage, thresholdImage, cleanBinary, separateVesicles
from measurement import findContours, measureVesicles

class FOAMSTestGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("PyFOAMS Testing GUI")
        self.imagePath = None
        self.pixelScale = 1000  # px/mm

        self.frame = ttk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        # Warning label
        warning = ttk.Label(self.frame, text="This is a testing GUI. Not for final use.", foreground="red")
        warning.grid(row=0, column=0, columnspan=3, pady=(0, 5))

        self.btnLoad = ttk.Button(self.frame, text="📂 Load Image", command=self.loadImage)
        self.btnLoad.grid(row=1, column=0, padx=5, pady=5)

        self.methodVar = tk.StringVar(value="otsu")
        ttk.Label(self.frame, text="Threshold Method:").grid(row=1, column=1)
        self.dropdown = ttk.OptionMenu(self.frame, self.methodVar, "otsu", "otsu", "manual", command=self.toggleSlider)
        self.dropdown.grid(row=1, column=2)

        self.manualThresh = tk.IntVar(value=128)
        self.slider = ttk.Scale(self.frame, from_=0, to=255, orient="horizontal", variable=self.manualThresh)
        self.slider.grid(row=2, column=1, columnspan=2, sticky="ew")
        self.slider.configure(state="disabled")

        self.btnProcess = ttk.Button(self.frame, text="▶ Run Pipeline", command=self.processImage)
        self.btnProcess.grid(row=3, column=0, columnspan=3, pady=10)

    def toggleSlider(self, choice):
        self.slider.configure(state="normal" if choice == "manual" else "disabled")

    def loadImage(self):
        self.imagePath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.tif")])
        if self.imagePath:
            print(f"[✓] Loaded image: {self.imagePath}")
    def processImage(self):
        if not self.imagePath:
            print("No image loaded.")
            return

        outDir = "main/gui_outputs"
        os.makedirs(outDir, exist_ok=True)

        print("[*] Processing image...")

        grey = loadGreyscaleImage(self.imagePath)
        cv2.imwrite(os.path.join(outDir, "1_grey.png"), grey)

        # Thresholding
        if self.methodVar.get() == "manual":
            binary = thresholdImage(grey, method="manual", manualThresh=self.manualThresh.get())
        else:
            binary = thresholdImage(grey, method="otsu")
        cv2.imwrite(os.path.join(outDir, "2_binary.png"), binary)

        # Morphological cleaning
        cleaned = cleanBinary(binary)
        cv2.imwrite(os.path.join(outDir, "3_cleaned.png"), cleaned)

        # Watershed separation
        separated = separateVesicles(cleaned)
        cv2.imwrite(os.path.join(outDir, "4_separated.png"), separated)

        # Contour detection and measurement
        contours = findContours(separated)
        vesicles = measureVesicles(contours, pixelScale=self.pixelScale)

        # Draw contours
        contourOverlay = cv2.cvtColor(grey.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contourOverlay, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(outDir, "5_contours.png"), contourOverlay)

        # Draw bounding boxes
        boxed = cv2.cvtColor(separated.copy(), cv2.COLOR_GRAY2BGR)
        for v in vesicles:
            x, y, w, h = v["boundingBox"]
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(outDir, "6_boxes.png"), boxed)

        print(f"[✓] Output saved to {outDir}")

        for fname in ["1_grey", "2_binary", "3_cleaned", "4_separated", "5_contours", "6_boxes"]:
            img = cv2.imread(os.path.join(outDir, f"{fname}.png"))
            cv2.imshow(fname, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FOAMSTestGUI(root)
    root.mainloop()
