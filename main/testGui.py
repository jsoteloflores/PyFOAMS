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

        ttk.Label(self.frame, text="Cleaning Method:").grid(row=4, column=0, sticky="w")
        self.cleanVar = tk.StringVar(value="morph")
        self.cleanDropdown = ttk.OptionMenu(self.frame, self.cleanVar, "morph", "morph", "blur", "none", command=self.toggleKernelSlider)
        self.cleanDropdown.grid(row=4, column=1, columnspan=2, sticky="ew")

        # Kernel size slider (for morph)
        ttk.Label(self.frame, text="Kernel Size:").grid(row=5, column=0, sticky="w")
        self.kernelSize = tk.IntVar(value=3)
        self.kernelSlider = ttk.Scale(self.frame, from_=1, to=7, orient="horizontal", variable=self.kernelSize)
        self.kernelSlider.grid(row=5, column=1, columnspan=2, sticky="ew")


    def toggleSlider(self, choice):
        self.slider.configure(state="normal" if choice == "manual" else "disabled")

    def toggleKernelSlider(self, choice):
        if choice == "morph":
            self.kernelSlider.configure(state="normal")
        else:
            self.kernelSlider.configure(state="disabled")

    def softCleanBinary(self, binary):
        blurred = cv2.GaussianBlur(binary, (3, 3), 0)
        _, cleaned = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        return cleaned



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
        # Clean according to dropdown
        cleanMethod = self.cleanVar.get()
        if cleanMethod == "none":
            cleaned = binary.copy()
        elif cleanMethod == "morph":
            k = int(self.kernelSize.get())
            cleaned = cleanBinary(binary, kernel_size=k)
        elif cleanMethod == "blur":
            cleaned = self.softCleanBinary(binary)
        else:
            print(f"[!] Unknown cleaning method: {cleanMethod}")
            cleaned = binary.copy()


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
