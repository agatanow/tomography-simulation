import pyforms
import cv2
import numpy as np
from tomograph import Radon
from   pyforms          import BaseWidget
from   pyforms.controls import ControlText
from   pyforms.controls import ControlButton
from   pyforms.controls import ControlFile
from   pyforms.controls import ControlImage
from   pyforms.controls import ControlMatplotlib

PYFORMS_STYLESHEET = 'style.css'

class TomographGUI(BaseWidget):

    def __init__(self):
        super(TomographGUI,self).__init__('TomographGUI')

        self.formset = [('_file','_loadButton'),
                         '_det_nr', '_angl_spread', 
                         '_angl_it', '_filter_size', 
                         ('_image', '_sinogram', '_output'), '_startButton',] 
                        
        #Definition of the forms fields
        self._file          = ControlFile()
        self._loadButton    = ControlButton('Load')
        self._det_nr        = ControlText('Detectors number')
        self._angl_spread   = ControlText('Angular spread')
        self._angl_it       = ControlText('Angluar step')
        self._filter_size   = ControlText('Filter size')
        self._image         = ControlImage()
        self._sinogram      = ControlImage()
        self._output        = ControlImage()
        self._startButton   = ControlButton('Start')

        self._loadButton.value  = self.__load
        self._startButton.value = self.__start

    def __load(self):
        if (self._file.value):
            self._image.value = cv2.imread(self._file.value, cv2.IMREAD_GRAYSCALE); 
            self._image.repaint()
    
    def __normalize(self, image):
        min = np.min(image)
        max = np.max(image)
        normalized = (image - min) / (max - min)
        return normalized

    def __start(self):
        if (self._image.value is None):
            return False
        det_nr      = int(self._det_nr.value)
        ang_spread  = int(self._angl_spread.value)
        it_ang      = int(self._angl_it.value)
        filter_size = int(self._filter_size.value)
        image = cv2.imread(self._file.value, cv2.IMREAD_GRAYSCALE)
        radon = Radon(image, det_nr, ang_spread, it_ang, filter_size)
        
        radon.transform()
        sinogram = radon.getSinogram()
        self._sinogram.value =  self.__normalize(sinogram)
        self._sinogram.repaint()
        radon.transform(inverse=True)
        result = radon.getResult()
        self._output.value = self.__normalize(result)
        self._output.repaint()


        # sinogram = radon.transform()
        # for iteration in sinogram:
        #     self._sinogram.value = iteration
        #     self._sinogram.repaint()
        # result = radon.transform(inverse=True)
        # for iteration in result:
        #     self._output.value = iteration
        #     self._output.repaint()
    
        


#Execute the application
if __name__ == "__main__":   pyforms.start_app( TomographGUI )