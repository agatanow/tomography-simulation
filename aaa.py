import cv2
from matplotlib import pyplot as plt
import numpy as np
import math as m
import datetime, time
from IPython.display import clear_output
import copy


def load(name):
    im = cv2.imread(name)
    return im


def plot(img):
    #     tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tmp = img
    plt.imshow(tmp)
    plt.xticks([]), plt.yticks([])
    plt.show()


def colorMean(color):
    return (0.11 * color[0] + 0.59 * color[1] + 0.3 * color[2])


def BresenhamLine(img, x1, y1, x2, y2, M):
    x, y = x1, y1
    sum, it = 0, 0
    xi = 1 if x1 < x2 else -1
    dx = xi * (x2 - x1)
    yi = 1 if y1 < y2 else -1
    dy = yi * (y2 - y1)

    col = colorMean(img[x][y])
    sum += col
    M[x][y] += 1
    it += 1
    if dx > dy:
        ai = (dy - dx) * 2
        bi = dy * 2
        d = bi - dx
        while x != x2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi
            col = colorMean(img[x][y])
            sum += col
            M[x][y] += 1
            it += 1
    else:
        ai = (dx - dy) * 2
        bi = dx * 2
        d = bi - dy
        while y != y2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                y += yi
            col = colorMean(img[x][y])
            sum += col
            M[x][y] += 1
            it += 1
    return sum / it


def normalize(sinog):
    if sinog:
        sinogram_max = max([i for row in sinog  for i in row])
        if sinogram_max != 0:
            sinog = [[0 if item < 0 else item / sinogram_max * 255 for item in row] for row in sinog]
    return sinog


def convertToUint8RGB(img):
    return [[[np.uint8(round(el))]*3 for el in row] for row in img]


def convolveSinogram(img, kernelSize=9):
    width = len(img)
    imgNew = [[0.0 for y in range(width)] for x in range(width)]
    for i in range(width):
        imgNew.append([])
        for j in range(width):
            imgNew[i].append(0.0)
    kernelCenter = int(kernelSize / 2)
    kernel = [0 if i % 2 == 0 else (-4 / pow(m.pi, 2)) / pow(i - kernelCenter, 2) for i in range(kernelSize)]
    kernel[kernelCenter] = 1.0
    for i in range(len(img)):
        j = kernelCenter
        while j < (width - kernelCenter):
            it = 0
            k = j - kernelCenter
            while it < kernelSize:
                imgNew[i][j] += img[i][k] * kernel[it]
                k += 1
                it += 1
            j += 1
    return imgNew


def sinogram(img, M, steps=12, l=45, detectors=5, alfa=180):
    alfa = alfa / steps
    alfaRadians = m.radians(alfa)
    center = (int)(img.shape[0] / 2)
    radius = center - 5
    tmpRadians = 0
    sinog = [[] for i in range(steps)]
    for i in range(steps):
        x = center + int(radius * np.cos(tmpRadians))
        y = center + int(radius * np.sin(tmpRadians))
        tmpDetectorRadians = m.radians(180 - l / 2) + tmpRadians
        detectorStep = m.radians(l / (detectors - 1))
        for j in range(detectors):
            xDet = center + int(radius * np.cos(tmpDetectorRadians))
            yDet = center + int(radius * np.sin(tmpDetectorRadians))
            tmpDetectorRadians += detectorStep
            sinog[i].append(BresenhamLine(img, x, y, xDet, yDet, M))
        tmpRadians += alfaRadians
    originalSinog = [[sinog[y][x] for y in range(detectors)] for x in range(steps)]
    originalSinog = convertToUint8RGB(originalSinog)
    originalSinog = np.asarray(originalSinog)
    plot(originalSinog)
    sinog = convolveSinogram(sinog)
    sinog = normalize(sinog)
    sinog = convertToUint8RGB(sinog)
    sinog = np.asarray(sinog)
    return sinog


def BresenhamLineInverse(img, x1, y1, x2, y2, brightness):
    x, y = x1, y1
    xi = 1 if x1 < x2 else -1
    dx = xi * (x2 - x1)
    yi = 1 if y1 < y2 else -1
    dy = yi * (y2 - y1)
    img[x][y] = img[x][y] + brightness
    if dx > dy:
        ai = (dy - dx) * 2
        bi = dy * 2
        d = bi - dx
        while x != x2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi
            img[x][y] = img[x][y] + brightness
    else:
        ai = (dx - dy) * 2
        bi = dx * 2
        d = bi - dy
        while y != y2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                y += yi
            img[x][y] = img[x][y] + brightness
    return img


def convolve(img, kernelSize):
    width = len(img)
    imgNew = []
    for i in range(width):
        imgNew.append([])
        for j in range(width):
            imgNew[i].append(0.0)
    kernel = np.ones((kernelSize, kernelSize))
    kernel /= 1.0 * kernelSize * kernelSize
    shift = int((kernelSize - 1) / 2)
    m = shift
    while m < (width - shift):
        n = shift
        while n < (width - shift):
            i = m - shift
            while i <= (m + shift):
                j = n - shift
                while j <= (n + shift):
                    imgNew[m][n] += img[i][j] * kernel[m - i][n - j]
                    j += 1
                i += 1
            n += 1
        m += 1
    return imgNew


def inverseRadonTransform(img, M, sinog, steps, l, detectors, alfa, kernelSize=5):
    alfa = alfa / steps
    alfaRadians = m.radians(alfa)
    center = (int)(img.shape[0] / 2)
    radius = center - 5
    centerTuple = (center, center)
    height, width, channels = img.shape
    img2 = [[0 for item in range(height)] for row in range(height)]
    tmpRadians = 0

    for i in range(steps):
        x = center + int(radius * np.cos(tmpRadians))
        y = center + int(radius * np.sin(tmpRadians))
        tmpDetectorRadians = m.radians(180 - l / 2) + tmpRadians
        detectorStep = m.radians(l / (detectors - 1))
        for j in range(detectors):
            xDet = center + int(radius * np.cos(tmpDetectorRadians))
            yDet = center + int(radius * np.sin(tmpDetectorRadians))
            tmpDetectorRadians += detectorStep
            brightness = colorMean(sinog[i][j])
            img2 = BresenhamLineInverse(img2, x, y, xDet, yDet, brightness)
        tmpRadians += alfaRadians
    for i, row in enumerate(img2):
        for j, el in enumerate(row):
            if M[i][j] != 0:
                img2[i][j] /= M[i][j]
    # for i, row in enumerate(img2):
    #     for j, el in enumerate(row):
    #         img2[i][j] = m.pow(el, 2)
    # img2 = convolve(img2, kernelSize)
    # for i, row in enumerate(img2):
    #     for j, el in enumerate(row):
    #         img2[i][j] = m.pow(el, (3 / 2))
    img2 = normalize(img2)
    img2 = convertToUint8RGB(img2)
    img2 = np.asarray(img2)
    return img2


def MeanSquaredError(img, img2):
    err = np.sum((img.astype("float") - img2.astype("float")) ** 2)
    err /= float(img.shape[0] * img.shape[1])
    return err


def PictureDifference(img, img2, height, width):
    imgD = [[0 for item in range(height)] for row in range(height)]
    for i, row in enumerate(img):
        for j, el in enumerate(row):
            imgD[i][j] = img[i][j] - img2[i][j]
            if img[i][j][0] < img2[i][j][0]:
                imgD[i][j] = [0, 0, 0]
    return np.asarray(imgD)


alfa = 360
steps = 180
l = 360
detectors = 250
kernelSize = 0
img = load("kolo.jpg")
plot(img)

width, height, channels = img.shape
M = [[0 for item in range(height)] for row in range(height)]
sinog = sinogram(img, M, steps, l, detectors, alfa)
newImg = inverseRadonTransform(img, M, sinog, steps, l, detectors, alfa, kernelSize)
plot(newImg)
err = MeanSquaredError(img, newImg)
imgD = PictureDifference(newImg, img, height, width)
plot(imgD)
imgD = PictureDifference(img, newImg, height, width)
plot(imgD)
print(err)
