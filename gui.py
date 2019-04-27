import cv2
import time
import numpy as np
from tomograph import Radon
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtWidgets import QLabel, QGridLayout
from PyQt5.QtWidgets import QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QImage, QPixmap, qRgb
from PyQt5.QtCore import QRectF, Qt

PYFORMS_STYLESHEET = 'style.css'

gray_color_table = [qRgb(i, i, i) for i in range(256)]

class TomographGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.interface()
        self.show()
        
    def interface(self):
        #labels
        self._det_l     = QLabel("Detectors number:")
        self._aSpread_l = QLabel("Angular spread:")
        self._aIter_l   = QLabel("Iteration angle:")
        self._fSize_l   = QLabel("Filter size:")
        self._image_l   = QLabel("Input image:")
        self._sin_l     = QLabel("Sinogram image:")
        self._out_l     = QLabel("Output image:")
        self._fOut_l     = QLabel("Filtered output image:")

        #textboxes
        self._det_t     = QLineEdit()
        self._aSpread_t = QLineEdit()
        self._aIter_t   = QLineEdit()
        self._fSize_t   = QLineEdit()

        #buttons
        self._fOpen_b   = QPushButton("&Open File")
        self._fOpen_b.clicked.connect(self.file_open)
        self._start_b   = QPushButton("&Start")
        self._start_b.clicked.connect(self.start)
        self._start_b.setEnabled(False)

        #canvas
        self._image_c   = FigureCanvas(plt.Figure(figsize=(5, 5)))
        self._sin_c     = FigureCanvas(plt.Figure(figsize=(5, 5)))
        self._out_c     = FigureCanvas(plt.Figure(figsize=(5, 5)))
        self._fOut_c    = FigureCanvas(plt.Figure(figsize=(5, 5)))

        #axes
        self._image_ax  = self._image_c.figure.subplots()
        self._sin_ax    = self._sin_c.figure.subplots()
        self._out_ax    = self._out_c.figure.subplots()
        self._fOut_ax    = self._fOut_c.figure.subplots()

        #layout
        self._layout    = QGridLayout()

        self._layout.addWidget(self._det_l, 1, 0)
        self._layout.addWidget(self._aSpread_l, 2, 0)
        self._layout.addWidget(self._aIter_l, 3, 0)
        self._layout.addWidget(self._fSize_l, 4, 0)
        self._layout.addWidget(self._image_l, 5, 0)
        self._layout.addWidget(self._sin_l, 5, 1)
        self._layout.addWidget(self._out_l, 5, 2)
        self._layout.addWidget(self._fOut_l, 5, 3)

        self._layout.addWidget(self._det_t, 1, 1)
        self._layout.addWidget(self._aSpread_t, 2, 1)
        self._layout.addWidget(self._aIter_t, 3, 1)
        self._layout.addWidget(self._fSize_t, 4, 1)

        self._layout.addWidget(self._fOpen_b, 0, 0)
        self._layout.addWidget(self._start_b, 0, 1)

        self._layout.addWidget(self._image_c, 6, 0)
        self._layout.addWidget(self._sin_c, 6, 1)
        self._layout.addWidget(self._out_c, 6, 2)
        self._layout.addWidget(self._fOut_c, 6, 3)

        self.setLayout(self._layout)
        self.setGeometry(20, 20, 1500, 1000)
        self.setWindowTitle("TomographGUI")

    def file_open(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')
        self._cvImg = cv2.imread(name[0], cv2.IMREAD_GRAYSCALE)
        self.update_canvas(self._cvImg, self._image_ax)
        self._start_b.setEnabled(True)

    def update_canvas(self, image, ax):
        ax.clear()
        ax.imshow(image, aspect='auto', cmap='gray')
        ax.figure.canvas.draw()

    def animate_sin(self, data):
        self._sin_ax.set_data(data)

    def start(self):
        det_nr      = int(self._det_t.text())
        ang_spread  = int(self._aSpread_t.text())
        it_ang      = int(self._aIter_t.text())
        filter_size = int(self._fSize_t.text())
        radon = Radon(self._cvImg, det_nr, ang_spread, it_ang, filter_size)
        
        sinogram = radon.transform()

        #self._sin_an = FuncAnimation(self._sin_c.figure, self.animate_sin, sinogram, interval=100, blit=True)
        #result = radon.transform(inverse=True)
        radon.transform()
        sinogram = radon.getSinogram()
        self.update_canvas(sinogram, self._sin_ax)
        radon.transform(inverse=True)
        result = radon.getResult()
        self.update_canvas(result,self._out_ax)
        self.update_canvas(cv2.GaussianBlur(radon.getResult(), (3, 3), 0), self._fOut_ax)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = TomographGUI()
    sys.exit(app.exec_())