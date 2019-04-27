import cv2
import numpy as np
from tomograph import Radon
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
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

        #scenes
        self._image_s   = QGraphicsScene()
        self._sin_s     = QGraphicsScene()
        self._out_s     = QGraphicsScene()

        #images
        self._image_v   = QGraphicsView(self._image_s)
        self._sin_v     = QGraphicsView(self._sin_s)
        self._out_v     = QGraphicsView(self._out_s)

        #holders (needed to not crash)
        self._image_h   = None
        self._sin_h     = None
        self._out_h     = None

        #layout
        self._layout    = QGridLayout()

        self._layout.addWidget(self._det_l, 1, 0)
        self._layout.addWidget(self._aSpread_l, 2, 0)
        self._layout.addWidget(self._aIter_l, 3, 0)
        self._layout.addWidget(self._fSize_l, 4, 0)
        self._layout.addWidget(self._image_l, 5, 0)
        self._layout.addWidget(self._sin_l, 5, 1)
        self._layout.addWidget(self._out_l, 5, 2)

        self._layout.addWidget(self._det_t, 1, 1)
        self._layout.addWidget(self._aSpread_t, 2, 1)
        self._layout.addWidget(self._aIter_t, 3, 1)
        self._layout.addWidget(self._fSize_t, 4, 1)

        self._layout.addWidget(self._fOpen_b, 0, 0)
        self._layout.addWidget(self._start_b, 0, 1)

        self._layout.addWidget(self._image_v, 6, 0)
        self._layout.addWidget(self._sin_v, 6, 1)
        self._layout.addWidget(self._out_v, 6, 2)

        self.setLayout(self._layout)
        self.setGeometry(20, 20, 1500, 1000)
        self.setWindowTitle("TomographGUI")

    def file_open(self):
        name = QFileDialog.getOpenFileName(self, 'Open File')
        self._cvImg = cv2.imread(name[0], cv2.IMREAD_GRAYSCALE)
        print(type(self._cvImg))
        
        height, width = self._cvImg.shape
        self._image_i = QImage(self._cvImg.data, width, height, QImage.Format_Indexed8)
        self._image_s.clear()
        pixMap = QPixmap.fromImage(self._image_i)
        self._image_s.addPixmap(pixMap)
        self._image_v.fitInView(QRectF(0, 0, width, height), Qt.KeepAspectRatio)
        self._image_s.update()
        self._start_b.setEnabled(True)

    def update_scene(self, scene, view, image, holder):
        height, width = image.shape
        holder = QImage(image.data, width, height, image.strides[0], QImage.Format_Indexed8)
        holder.setColorTable(gray_color_table)
        scene.clear()
        pixMap = QPixmap.fromImage(holder)
        scene.addPixmap(pixMap)
        view.fitInView(QRectF(0, 0, width, height), Qt.KeepAspectRatio)
        scene.update()

    def __normalize(self, image):
        min = np.min(image)
        max = np.max(image)
        normalized = (image - min) / (max - min)
        return normalized

    def start(self):
        det_nr      = int(self._det_t.text())
        ang_spread  = int(self._aSpread_t.text())
        it_ang      = int(self._aIter_t.text())
        filter_size = int(self._fSize_t.text())
        radon = Radon(self._cvImg, det_nr, ang_spread, it_ang, filter_size)
        

        radon.transform()
        sinogram = self.__normalize(np.transpose(radon.getSinogram()).copy())
        print(type(sinogram))
        self.update_scene(self._sin_s, self._sin_v, sinogram, self._sin_h)
        radon.transform(inverse=True)
        result = self.__normalize(np.transpose(radon.getResult()).copy())
        self.update_scene(self._out_s, self._out_v, result, self._out_h)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = TomographGUI()
    sys.exit(app.exec_())