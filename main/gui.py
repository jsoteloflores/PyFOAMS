import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from imageProcessing import loadGreyscaleImage, thresholdImage, cleanBinary

class PyFOAMS_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("PyFOAMS GUI")
        self.imagePath = None
        self.image = None

        # Create toolbar
        self.toolbar = ttk.Frame(master)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btnLoad = ttk.Button(self.toolbar, text="📂 Load Image", command=self.loadImage)
        self.btnLoad.pack(side=tk.LEFT, padx=5, pady=5)

        self.paintbrushTool = ttk.Button(self.toolbar, text="🖌 Paintbrush", command=self.activatePaintbrush)
        self.paintbrushTool.pack(side=tk.LEFT, padx=5, pady=5)

        self.bucketTool = ttk.Button(self.toolbar, text="🪣 Bucket", command=self.activateBucket)
        self.bucketTool.pack(side=tk.LEFT, padx=5, pady=5)

        self.processIslandsTool = ttk.Button(self.toolbar, text="Process Islands", command=self.processIslands)
        self.processIslandsTool.pack(side=tk.LEFT, padx=5, pady=5)

        self.erodeTool = ttk.Button(self.toolbar, text="Erode Tool", command=self.activateErode)
        self.erodeTool.pack(side=tk.LEFT, padx=5, pady=5)

        # Create canvas for image
        self.canvas = tk.Canvas(master, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def loadImage(self):
        self.imagePath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.tif")])
        if self.imagePath:
            raw_image = cv2.imread(self.imagePath, cv2.IMREAD_GRAYSCALE)
            binary = thresholdImage(raw_image, method="otsu")
            self.image = cleanBinary(binary, kernel_size=3)  # Process to cleaned black-and-white
            self.displayImage()
            self.btnLoad.config(text="Image Loaded", state=tk.DISABLED)

    def displayImage(self):
        if self.image is None:
            return

        img = Image.fromarray(self.image)
        imgTk = ImageTk.PhotoImage(img)

        self.canvas.image = imgTk  # Keep a reference to avoid garbage collection
        self.canvas.create_image(0, 0, anchor="nw", image=imgTk)

        # Enable drawing functionality
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.startPaint)
        self.canvas.bind("<ButtonRelease-1>", self.endPaint)

        self.painting = False
        self.brushSize = 5
        self.brushColor = 0  # Black for adding pore space

        self.eraseMode = False  # Toggle between paint and erase

    # Paintbrush functionality
    def startPaint(self, event):
        self.painting = True

    def endPaint(self, event):
        self.painting = False

    def paint(self, event):
        if not self.painting or self.image is None:
            return

        x, y = event.x, event.y
        brushSize = self.brushSize

        # Convert canvas coordinates to image coordinates
        imgX = int(x * self.image.shape[1] / self.canvas.winfo_width())
        imgY = int(y * self.image.shape[0] / self.canvas.winfo_height())

        # Apply brush or erase
        if self.eraseMode:
            cv2.circle(self.image, (imgX, imgY), brushSize, 255, -1)  # White for erasing
        else:
            cv2.circle(self.image, (imgX, imgY), brushSize, self.brushColor, -1)  # Black for painting

        self.displayImage()

    # Toggle erase mode
    def toggleEraseMode(self):
        self.eraseMode = not self.eraseMode
        print("Erase mode" if self.eraseMode else "Paint mode")

    def activatePaintbrush(self):
        print("Paintbrush tool activated.")

    def activateBucket(self):
        print("Bucket tool activated.")

    def processIslands(self):
        print("Processing islands...")

    def activateErode(self):
        print("Erode tool activated.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PyFOAMS_GUI(root)
    root.mainloop()
