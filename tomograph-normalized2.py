from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
import matplotlib.pyplot as plt
import math
import numpy as np
import random as rd
import time

DETECTORS_NR = 51
ANGULAR_SPREAD = 90 # <180
ITERATION_ANGLE = 2 # <360
FILTER_SIZE = 100

class BrasenhamLine:
    def __init__(self, x1, y1, x2, y2):
         line = []
         x = x1
         y = y1
         if x1 < x2:
             xi = 1
             dx = x2 - x1
         else:
             xi = -1
             dx = x1 - x2
         if y1 < y2:
             yi = 1
             dy = y2 - y1
         else:
             yi = -1
             dy = y1 - y2
         line.append(np.array([x,y]))
         if (dx > dy):
             ai = (dy - dx) * 2
             bi = dy * 2
             d = bi - dx
             while (x != x2):
                if (d >= 0):
                     x += xi
                     y += yi
                     d += ai
                else:
                     d += bi
                     x += xi
                line.append(np.array([x,y]))
         else:
             ai = ( dx - dy ) * 2
             bi = dx * 2
             d = bi - dy
             while (y != y2):
                 if (d >= 0):
                     x += xi
                     y += yi
                     d += ai
                 else:
                     d += bi
                     y += yi
                 line.append(np.array([x,y]))
         self.points = np.array(line)

    def mapToValue(self, image):
        val = 0
        for p in self.points:
            val += image[p[0],p[1]]
        return val/self.points.shape[0]

    def addValue(self, image, value):
        for point in self.points:
            image[point[0], point[1]] += value

    def draw(self, image, color):
        for p in self.points:
            image[p[0],p[1]] = color

    def count(self, freq):
        for p in self.points:
            freq[p[0],p[1]] += 1

class Circle:
    def __init__(self, det_nr, ang_spread, it_ang):
        self.detectors_nr = det_nr
        self.angular_spread = ang_spread
        self.iteration_angle = it_ang

    def getAngle(self,p):# of point ON A BORDER
        x=p[0]
        y=p[1]
        #angle = math.acos(x)
        angle2 = math.fabs(math.asin(y))
        a=x
        b=y
        if a>0 and b>=0: #pierwsza
            alfa = angle2
        if a<=0 and b>0: #druga
            alfa = math.pi - angle2
        if a<0 and b<=0: #trzecia
            alfa = math.pi + angle2
        if a>=0 and b<0: #czwarta
            alfa = math.pi*2 - angle2

        return math.degrees(alfa)

    def getPoint(self,a): #of angle
        a = math.radians(a)
        return np.array([math.cos(a), math.sin(a)])

    def nextPoint(self, p, alfa):
        tem = self.getAngle(p) + alfa
        return self.getPoint(tem)

    def getDetectors(self, x):
        kr1 = (x + 180) - self.angular_spread
        move = self.angular_spread*2 / (self.detectors_nr - 1)
        detectors = np.empty([self.detectors_nr + 1, 2])
        for i in range(self.detectors_nr):
            detectors[i] = self.getPoint(kr1+move*i)
        detectors[self.detectors_nr]=self.getPoint(x)
        return detectors[::-1]

class Radon:
    def __init__(self, image,  det_nr, ang_spread, it_ang, filter_size):
        self.it_ang = it_ang
        self.det_nr = det_nr
        self.circle_model = Circle(det_nr, ang_spread, it_ang)
        self.image = image
        self.filter_size = filter_size
        self.filter = self.__createFilter(self.filter_size) if filter_size > 0 else None
        self.result = np.zeros(image.shape)
        self.sinogram = np.zeros([math.ceil(360/it_ang), det_nr])
        self.freq = np.zeros(image.shape)

    def mapPoints(self, points):
        return np.round((points + 1) * (self.image.shape[0]/2 - 1))

    def transform(self, inverse = False):
        for it, angle in enumerate(range(0, 360, self.it_ang)):
            detectors = self.mapPoints(self.circle_model.getDetectors(angle)).astype(int)
            emiter = detectors[0]
            for detId in range(0, self.det_nr):
                detector = detectors[detId + 1]
                line = BrasenhamLine(emiter[0], emiter[1], detector[0], detector[1])
                if (inverse == False):
                    value = line.mapToValue(self.image)
                    self.sinogram[it, detId] = value
                else:
                    line.count(self.freq)
                    line.addValue(self.result, self.sinogram[it, detId])
            if self.filter is not None:
                splot = self.__convolve(self.sinogram[it], self.filter)
                self.sinogram[it] = splot
            #yield self.result if inverse else self.sinogram

    def __convolve(self, values, filter):
        new_values = np.array(values)
        for x in range(values.shape[0]):
            neighbor_sum = 0
            for y in range(filter.size):
                if (x - y >= 0):
                    neighbor_sum += values[x - y] * filter[y]
                if (x + y <= values.shape[0]):
                    neighbor_sum += values[x - y] * filter[y]
            new_values[x] = values[x] + neighbor_sum
        return new_values

    def __createFilter(self, size):
        filter = np.zeros(size)
        for x in range(size):
            filter[x] = 0 if x%2 == 1 else -(2/(np.pi*(x+1)))**2
        return filter

    def normalize(self, image):
        peak = np.max(image)
        if peak == 0:
            return image
        return np.array([np.array([0 if item < 0 else item for item in row]) for row in image])        

    def getResult(self):
        self.freq = np.array([np.array([1 if item == 0 else item for item in row]) for row in self.freq])
        return self.result / self.freq

    def getSinogram(self):
        res = np.array(self.sinogram)
        return np.rot90(res)

def radon_transform(image = None):
    #if image == None:
    image = imread("kolo.jpg", as_grey=True)

    #image = imread(data_dir + "/phantom.png")#
    #image = np.zeros([501,501,3])
    #image = image/255

    radon = Radon(image, DETECTORS_NR, ANGULAR_SPREAD, ITERATION_ANGLE, FILTER_SIZE)
    radon.transform()
    image = radon.getSinogram()
    image = radon.normalize(image)
    #cir = Circle(DETECTORS_NR, ANGULAR_SPREAD, ITERATION_ANGLE)
    #print(cir.getPoint(0))

    fig, ax = plt.subplots(sharex=True, figsize=(5, 5))
    ax.imshow(image, aspect='auto', cmap='gray')
    plt.show()

    radon.transform(inverse=True)
    image = radon.getResult()
    image = radon.normalize(image)
    
    fig, ax = plt.subplots(sharex=True, figsize=(5, 5))
    ax.imshow(image, aspect='auto', cmap='gray')
    plt.show()

if __name__ == "__main__":
    radon_transform()
