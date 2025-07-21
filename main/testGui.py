import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
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

        self.btnProcess = ttk.Button(self.frame, text="▶ Run Pipeline", command=self.updatePipeline)
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

        # Sensitivity sliders for thresholds
        ttk.Label(self.frame, text="Distance Transform Sensitivity:").grid(row=6, column=0, sticky="w")
        self.distSensitivity = tk.DoubleVar(value=0.3)
        self.distSlider = ttk.Scale(self.frame, from_=0.1, to=1.0, orient="horizontal", variable=self.distSensitivity, command=self.updatePipeline)
        self.distSlider.grid(row=6, column=1, columnspan=2, sticky="ew")

        ttk.Label(self.frame, text="Foreground Threshold:").grid(row=7, column=0, sticky="w")
        self.fgThreshold = tk.DoubleVar(value=0.3)
        self.fgSlider = ttk.Scale(self.frame, from_=0.1, to=1.0, orient="horizontal", variable=self.fgThreshold, command=self.updatePipeline)
        self.fgSlider.grid(row=7, column=1, columnspan=2, sticky="ew")


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
            self.updatePipeline()

    def updatePipeline(self, *args):
        if not self.imagePath:
            print("No image loaded.")
            return

        outDir = "main/gui_outputs"
        debugDir = os.path.join(outDir, "debugSteps")
        os.makedirs(debugDir, exist_ok=True)

        print("[*] Processing image...")

        # Clear previous images
        for widget in self.frame.winfo_children():
            if isinstance(widget, tk.Label) or isinstance(widget, tk.Canvas):
                widget.destroy()

        # Ensure image is saved in compatible format
        grey = loadGreyscaleImage(self.imagePath)
        grey = cv2.convertScaleAbs(grey)  # Convert to CV_8U
        grey_path = os.path.join(outDir, "1_grey.png")
        cv2.imwrite(grey_path, grey)

        # Thresholding
        if self.methodVar.get() == "manual":
            binary = thresholdImage(grey, method="manual", manualThresh=self.manualThresh.get())
        else:
            binary = thresholdImage(grey, method="otsu")
        binary = cv2.convertScaleAbs(binary)  # Convert to CV_8U
        binary_path = os.path.join(outDir, "2_binary.png")
        cv2.imwrite(binary_path, binary)

        # Morphological cleaning
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
        cleaned = cv2.convertScaleAbs(cleaned)  # Convert to CV_8U
        cleaned_path = os.path.join(debugDir, "3_cleaned.png")
        cv2.imwrite(cleaned_path, cleaned)

        # Watershed separation
        separated = separateVesicles(cleaned)
        separated = cv2.convertScaleAbs(separated)  # Convert to CV_8U
        separated_path = os.path.join(debugDir, "4_separated.png")
        cv2.imwrite(separated_path, separated)

        # Contour detection and measurement
        contours = findContours(separated)
        vesicles = measureVesicles(contours, pixelScale=self.pixelScale)

        # Draw contours
        contourOverlay = cv2.cvtColor(grey.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contourOverlay, contours, -1, (0, 255, 0), 1)
        contour_path = os.path.join(debugDir, "5_contours.png")
        cv2.imwrite(contour_path, contourOverlay)

        # Draw bounding boxes
        boxed = cv2.cvtColor(separated.copy(), cv2.COLOR_GRAY2BGR)
        for v in vesicles:
            x, y, w, h = v["boundingBox"]
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 1)
        boxed_path = os.path.join(debugDir, "6_boxes.png")
        cv2.imwrite(boxed_path, boxed)

        print(f"[✓] Output saved to {outDir}")

        # Display images in a grid layout
        steps = [
            ("Step 1: Greyscale", grey_path),
            ("Step 2: Binary", binary_path),
            ("Step 3: Cleaned", cleaned_path),
            ("Step 4: Separated", separated_path),
            ("Step 5: Contours", contour_path),
            ("Step 6: Boxes", boxed_path),
            ("Debug: Cleaned", cleaned_path),
            ("Debug: Separated", separated_path),
            ("Debug: Contours", contour_path),
            ("Debug: Boxes", boxed_path)
        ]

        num_columns = 3  # Number of columns in the grid

        def resize_image(event, img_path, canvas):
            img = cv2.imread(img_path)
            if img is None:
                return

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((event.width, event.height), Image.ANTIALIAS)
            imgTk = ImageTk.PhotoImage(img)

            canvas.image = imgTk  # Keep a reference to avoid garbage collection
            canvas.create_image(0, 0, anchor="nw", image=imgTk)

        for i, (title, path) in enumerate(steps):
            row = i // num_columns + 8  # Start at row 8
            col = i % num_columns

            label = tk.Label(self.frame, text=title, font=("Arial", 12, "bold"))
            label.grid(row=row * 2, column=col, pady=(10, 0), padx=10)

            canvas = tk.Canvas(self.frame, width=200, height=200)
            canvas.grid(row=row * 2 + 1, column=col, pady=(0, 10), padx=10)
            canvas.bind("<Configure>", lambda event, img_path=path, canvas=canvas: resize_image(event, img_path, canvas))

if __name__ == "__main__":
    root = tk.Tk()
    app = FOAMSTestGUI(root)
    root.mainloop()
