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

        # Add a dropdown to select separation method
        self.separationMethodVar = tk.StringVar(value="watershed")
        ttk.Label(self.frame, text="Separation Method:").grid(row=8, column=0, sticky="w")
        self.separationDropdown = ttk.OptionMenu(self.frame, self.separationMethodVar, "watershed", "watershed", "distance", command=self.updatePipeline)
        self.separationDropdown.grid(row=8, column=1, columnspan=2, sticky="ew")


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

        # Save distance transform thresholding
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_transform_path = os.path.join(debugDir, "debug_dist_transform.png")
        cv2.imwrite(dist_transform_path, dist_transform)

        # Save watershed thresholding
        _, markers = cv2.threshold(dist_transform, self.distSensitivity.get() * 255, 255, cv2.THRESH_BINARY)
        watershed_path = os.path.join(debugDir, "debug_watershed.png")
        cv2.imwrite(watershed_path, markers)

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
        if self.separationMethodVar.get() == "distance":
            separated = dist_transform.copy()
        else:
            separated = separateVesicles(cleaned)
        separated = cv2.convertScaleAbs(separated)  # Convert to CV_8U
        separated_path = os.path.join(debugDir, "4_separated.png")
        cv2.imwrite(separated_path, separated)

        # Contour detection and measurement
        contours = findContours(separated)
        vesicles = measureVesicles(contours, pixelScale=self.pixelScale)

        # Draw contours
        contourOverlay = cv2.cvtColor(cleaned.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contourOverlay, contours, -1, (0, 255, 0), 1)
        contour_path = os.path.join(debugDir, "5_contours.png")
        cv2.imwrite(contour_path, contourOverlay)

        # Ensure bounding boxes are drawn correctly
        if vesicles:
            boxed = cv2.cvtColor(cleaned.copy(), cv2.COLOR_GRAY2BGR)
            for v in vesicles:
                x, y, w, h = v["boundingBox"]
                cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 1)
            boxed_path = os.path.join(debugDir, "6_boxes.png")
            cv2.imwrite(boxed_path, boxed)
        else:
            boxed_path = os.path.join(debugDir, "6_boxes.png")
            print("[!] No vesicles detected for bounding boxes.")

        print(f"[✓] Output saved to {outDir}")

        # Display images in a grid layout
        steps = [
            ("Step 1: Greyscale", grey_path),
            ("Step 2: Binary", binary_path),
            ("Step 3: Distance Transform", dist_transform_path),
            ("Step 4: Watershed", watershed_path),
            ("Step 5: Cleaned", cleaned_path),
            ("Step 6: Separated", separated_path),
            ("Step 7: Contours", contour_path),
            ("Step 8: Boxes", boxed_path)
        ]

        num_columns = 3  # Number of columns in the grid

        def resize_image(event, img_path, canvas):
            img = cv2.imread(img_path)
            if img is None:
                return

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((event.width, event.height), Image.Resampling.LANCZOS)  # Updated attribute
            imgTk = ImageTk.PhotoImage(img)

            canvas.image = imgTk  # Keep a reference to avoid garbage collection
            canvas.create_image(0, 0, anchor="nw", image=imgTk)

        def open_image_window(img_path):
            viewer = ZoomPanImageViewer(self.master, img_path)
            viewer.grab_set()  # Make the viewer modal

        for i, (title, path) in enumerate(steps):
            row = i // num_columns + 8  # Start at row 8
            col = i % num_columns

            label = tk.Label(self.frame, text=title, font=("Arial", 12, "bold"))
            label.grid(row=row * 2, column=col, pady=(10, 0), padx=10)

            canvas = tk.Canvas(self.frame, width=200, height=200)
            canvas.grid(row=row * 2 + 1, column=col, pady=(0, 10), padx=10)
            canvas.bind("<Configure>", lambda event, img_path=path, canvas=canvas: resize_image(event, img_path, canvas))
            canvas.bind("<Double-Button-1>", lambda event, img_path=path: open_image_window(img_path))

    def autoSearchSensitivities(self):
        print("[*] Searching for optimal sensitivities...")

        grey = loadGreyscaleImage(self.imagePath)

        # Define ranges for sensitivity search
        dist_range = np.linspace(0.1, 1.0, 10)
        fg_range = np.linspace(0.1, 1.0, 10)

        best_dist = 0.3
        best_fg = 0.3
        best_metric = float('-inf')

        for dist in dist_range:
            for fg in fg_range:
                binary = thresholdImage(grey, method="otsu", distSensitivity=dist, fgThreshold=fg)
                metric = self.evaluateImageQuality(binary)

                if metric > best_metric:
                    best_metric = metric
                    best_dist = dist
                    best_fg = fg

        print(f"[✓] Optimal sensitivities found: Distance Transform = {best_dist}, Foreground Threshold = {best_fg}")

        # Update sliders with optimal values
        self.distSensitivity.set(best_dist)
        self.fgThreshold.set(best_fg)

        print("[*] Sliders updated. You can fine-tune the settings if needed.")

    def evaluateImageQuality(self, binary):
        # Example metric: Count of edges detected
        edges = cv2.Canny(binary, 100, 200)
        return np.sum(edges > 0)

# Add zooming and panning functionality to the image viewer
class ZoomPanImageViewer(tk.Toplevel):
    def __init__(self, master, img_path):
        super().__init__(master)
        self.title("Image Viewer")

        self.img = cv2.imread(img_path)
        if self.img is None:
            print(f"[!] Image not found: {img_path}")
            self.destroy()
            return

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = Image.fromarray(self.img)
        self.imgTk = ImageTk.PhotoImage(self.img)

        self.canvas = tk.Canvas(self, width=self.img.width, height=self.img.height)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.create_image(0, 0, anchor="nw", image=self.imgTk)

        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.canvas.bind("<MouseWheel>", self.zoom)

        self.offset_x = 0
        self.offset_y = 0
        self.scale = 1.0

    def start_pan(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def pan(self, event):
        dx = event.x - self.start_x
        dy = event.y - self.start_y

        self.offset_x += dx
        self.offset_y += dy

        self.canvas.scan_dragto(-dx, -dy, gain=1)

        self.start_x = event.x
        self.start_y = event.y

    def zoom(self, event):
        scale_factor = 1.1 if event.delta > 0 else 0.9
        self.scale *= scale_factor

        self.canvas.scale("all", event.x, event.y, scale_factor, scale_factor)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    app = FOAMSTestGUI(root)
    root.mainloop()
