import cv2
import numpy as np


class CutImg:
    def __init__(self, cut_range):
        """裁剪图像

        Arguments:
            cut_range {list(3*2)} -- 裁剪范围 e.g.[[0, 720], [0, 1280], [0, 3]]
        """
        self.cut_range = cut_range

    def __call__(self, img):
        if img is None:
            return img
        img = img[self.cut_range[0][0]:self.cut_range[0][1], self.
                  cut_range[1][0]:self.cut_range[1][1], self.
                  cut_range[2][0]:self.cut_range[2][1]]
        return img


class Resize:
    def __init__(self, size):
        """调整尺寸

        Arguments:
            size {tuple(2)} -- 图像尺寸 e.g.(320, 320)
        """
        self.size = size

    def __call__(self, img):
        if img is None:
            return img
        img = cv2.resize(img, self.size)
        return img


class PerspectiveProject:
    def __init__(self, project_matrix, size=None):
        """透视变换

        Arguments:
            project_matrix {np.float} -- cv2.getPerspectiveTransform得来的透视变换矩阵

        Keyword Arguments:
            size {tuple(2)} -- 透视变换后的图像尺寸 (default: {None})
        """
        self.project_matrix = project_matrix
        self.size = size

    def __call__(self, img):
        if self.size is None:
            size = (img.shape[1], img.shape[0])
        else:
            size = self.size

        img = cv2.warpPerspective(img.copy(), self.project_matrix, size)
        return img


class ComputeHog:
    def __call__(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32))
        winSize = (16, 16)
        blockSize = (8, 8)
        blockStride = (4, 4)
        cellSize = (4, 4)
        nbins = 9
        derivAperture = 1
        winSigma = -1
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                                nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold,
                                gammaCorrection, nlevels)
        descriptor = hog.compute(img)
        return descriptor[:, 0]


class SobelX:
    def __call__(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=-1)
        img = cv2.resize(img, (16, 16))
        img = np.float32(img)
        return img


class SobelY:
    def __call__(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=-1)
        img = cv2.resize(img, (16, 16))
        img = np.float32(img)
        return img
