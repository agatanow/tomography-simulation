from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
import matplotlib.pyplot as plt
import math
import numpy as np
import random as rd
import time

DETECTORS_NR = 100
ANGULAR_SPREAD = 90 # <180
ITERATION_ANGLE = 5 # <360


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

    def draw(self, image, color):
        for p in self.points:
            image[p[0],p[1]] = color

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
        move = self.angular_spread*2/(self.detectors_nr-1)
        detectors = np.empty([self.detectors_nr + 1, 2])
        for i in range(self.detectors_nr):
            detectors[i] = self.getPoint(kr1+move*i)
        detectors[self.detectors_nr]=self.getPoint(x)

        return detectors[::-1]

    def iterate(self):
        x = 0
        i = 0
        while(x<360):
            #p = self.getPoint(x)
            #print(self.getDetectors(x))
            #time.sleep(5)
            i+=1
            print(x, i )
            yield self.getDetectors(x)
            x += self.iteration_angle



class Radon:
    def __init__(self, image,  det_nr, ang_spread, it_ang):
        self.circle_model = Circle(det_nr, ang_spread, it_ang)
        #self.det_generator = self.circle_model.iterate()
        self.image = image
        self.result = np.zeros(image.shape)
        self.sinogram = [] #np.zeros([det_nr, 360//it_ang])

    def iterate(self):
        for x in self.circle_model.iterate():
            yield np.round((x+1)*(self.image.shape[0]/2-1))

    def transform(self):
        for x in self.iterate():
            sin_line = []
            x1 = (int)(x[0,0])
            y1 = (int)(x[0,1])
            for y in x[1:]:
                #print(x1,y1,(int)(y[0]),(int)(y[1]))
                b = BrasenhamLine(x1,y1,(int)(y[0]),(int)(y[1]))
                #it_img = b.draw(self.image, (1,0,0))
                o=b.mapToValue(self.image)
                #self.sinogram = 
                sin_line.append(o)
            #time.sleep(5)
            self.sinogram.append(sin_line)
            # fig, ax = plt.subplots(sharex=True, figsize=(5, 5))
            # print(self.sinogram)
            # ax.imshow(self.sinogram, aspect='auto', cmap='gray')
            # plt.show()

    def inverse(self):
        for x in self.iterate():
            
            yield
    
    def getResult(self):
        return self.result

    def getSinogram(self):
        res = np.array(self.sinogram)
        m =res.max()
        #print(res.max())
        # fig, ax = plt.subplots(sharex=True, figsize=(5, 5))
        # ax.imshow(self.image, aspect='auto')
        # plt.show()
        #self.sinogram.append(sin_line)
        return np.rot90(res/m)





def radon_transform(image = None):
    #if image == None:
    image = imread("pic_s.png", as_grey=True)

    #image = imread(data_dir + "/phantom.png")#
    #image = np.zeros([501,501,3])
    #image = image/255

    radon = Radon(image, DETECTORS_NR, ANGULAR_SPREAD, ITERATION_ANGLE)
    radon.transform()
    radon.inverse()
    image=radon.getSinogram()
    print(image.shape, math.ceil(360/ITERATION_ANGLE))
    #cir = Circle(DETECTORS_NR, ANGULAR_SPREAD, ITERATION_ANGLE)
    #print(cir.getPoint(0))

    #fig, ax = plt.subplots(sharex=True, figsize=(5, 5))
    #ax.imshow(image, aspect='auto')
    #plt.show()

if __name__ == "__main__":
    radon_transform()
