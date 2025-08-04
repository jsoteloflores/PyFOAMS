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

        self.cropTool = ttk.Button(self.toolbar, text="Crop Tool", command=self.activateCrop)
        self.cropTool.pack(side=tk.LEFT, padx=5, pady=5)

        # Create canvas for image
        self.canvas = tk.Canvas(master, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def loadImage(self):
        self.imagePath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.tif")])
        if self.imagePath:
            self.image = cv2.imread(self.imagePath, cv2.IMREAD_GRAYSCALE)
            self.displayImage()
            self.btnLoad.config(text="Image Loaded", state=tk.DISABLED)

    def displayImage(self):
        if self.image is None:
            return

        img = Image.fromarray(self.image)
        imgTk = ImageTk.PhotoImage(img)

        self.canvas.image = imgTk  # Keep a reference to avoid garbage collection
        self.canvas.create_image(0, 0, anchor="nw", image=imgTk)

    def activatePaintbrush(self):
        print("Paintbrush tool activated.")

    def activateBucket(self):
        print("Bucket tool activated.")

    def processIslands(self):
        print("Processing islands...")

    def activateErode(self):
        print("Erode tool activated.")

    def activateCrop(self):
        print("Crop tool activated.")
        

if __name__ == "__main__":
    root = tk.Tk()
    app = PyFOAMS_GUI(root)
    root.mainloop()
