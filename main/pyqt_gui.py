import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint

class PyFOAMS_GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyFOAMS GUI")
        self.image = None
        self.drawing = False
        self.lastPoint = QPoint()
        self.brushSize = 5
        self.brushColor = Qt.black

        self.initUI()

    def initUI(self):
        self.canvas = QLabel(self)
        self.canvas.setStyleSheet("background-color: white;")
        self.canvas.setFixedSize(800, 600)

        loadButton = QPushButton("Load Image", self)
        loadButton.clicked.connect(self.loadImage)

        paintButton = QPushButton("Paintbrush", self)
        paintButton.clicked.connect(self.activatePaintbrush)

        bucketButton = QPushButton("Bucket Fill", self)
        bucketButton.clicked.connect(self.activateBucket)

        layout = QVBoxLayout()
        layout.addWidget(loadButton)
        layout.addWidget(paintButton)
        layout.addWidget(bucketButton)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def loadImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if filePath:
            self.image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
            self.updateCanvas()

    def updateCanvas(self):
        if self.image is None:
            return

        height, width = self.image.shape
        bytesPerLine = width
        qImg = QImage(self.image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        self.canvas.setPixmap(QPixmap.fromImage(qImg))

    def activatePaintbrush(self):
        self.canvas.mousePressEvent = self.startDrawing
        self.canvas.mouseMoveEvent = self.draw
        self.canvas.mouseReleaseEvent = self.stopDrawing

    def startDrawing(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def draw(self, event):
        if self.drawing and self.image is not None:
            painter = QPainter(self.canvas.pixmap())
            pen = QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            painter.end()
            self.update()

    def stopDrawing(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def activateBucket(self):
        self.canvas.mousePressEvent = self.bucketFill

    def bucketFill(self, event):
        if self.image is None:
            return

        x = event.pos().x()
        y = event.pos().y()

        # Convert canvas coordinates to image coordinates
        imgX = int(x * self.image.shape[1] / self.canvas.width())
        imgY = int(y * self.image.shape[0] / self.canvas.height())

        mask = np.zeros((self.image.shape[0] + 2, self.image.shape[1] + 2), np.uint8)
        cv2.floodFill(self.image, mask, (imgX, imgY), 255)
        self.updateCanvas()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyFOAMS_GUI()
    window.show()
    sys.exit(app.exec_())
