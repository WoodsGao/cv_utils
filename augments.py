import cv2
import random
import numpy as np
import math


# bigger better
class PerspectiveProject:
    def __init__(self, rate=0.1, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, img, det=None, seg=None):
        if random.random() > self.p:
            return img, det, seg
        src = np.float32([[0, 0], [0, img.shape[0]], [img.shape[1], 0],
                          [img.shape[1], img.shape[0]]])
        dst = src + np.float32(
            (np.random.rand(4, 2) * 2 - 1) *
            np.float32([img.shape[0], img.shape[1]]) * self.rate)
        p_matrix = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, p_matrix, (img.shape[1], img.shape[0]))

        if det is not None:
            new_det = list()
            # detection n*(x1 y1 x2 y2 x3 y3 x4 y4)
            for i in range(4):
                point = np.concatenate(
                    [det[:, 2 * i:2 * i + 2],
                     np.ones([det.shape[0], 1])], 1)
                point = np.dot(p_matrix, point.transpose(1, 0)).transpose(1, 0)
                point[:, :2] /= point[:, 2:]
                new_det.append(point[:, :2])
            det = np.concatenate(new_det, 1)
        if seg is not None:
            seg = cv2.warpPerspective(seg, p_matrix,
                                      (seg.shape[1], seg.shape[0]))
        return img, det, seg


class H_Flap:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, det=None, seg=None):
        if random.random() > self.p:
            return img, det, seg
        img = img[:, ::-1, :]
        if det is not None:
            # detection n*(x1 y1 x2 y2 x3 y3 x4 y4)
            det[:, 0::2] = img.shape[1] - det[:, 0::2]
        if seg is not None:
            seg = seg[:, ::-1, :]
        return img, det, seg


class V_Flap:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, det=None, seg=None):
        if random.random() > self.p:
            return img, det, seg
        img = img[::-1, :, :]
        if det is not None:
            # detection n*(x1 y1 x2 y2 x3 y3 x4 y4)
            det[:, 1::2] = img.shape[0] - det[:, 1::2]
        if seg is not None:
            seg = seg[::-1, :, :]
        return img, det, seg


class HSV_H:
    def __init__(self, rate=0.1, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, img, det=None, seg=None):
        if random.random() > self.p:
            return img, det, seg
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = np.float32(img)
        img[:, :, 0] += 255 * (self.rate * random.uniform(-1, 1))
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img, det, seg


class HSV_S:
    def __init__(self, rate=0.1, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, img, det=None, seg=None):
        if random.random() > self.p:
            return img, det, seg
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = np.float32(img)
        img[:, :, 1] += 255 * (self.rate * random.uniform(-1, 1))
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img, det, seg


class HSV_V:
    def __init__(self, rate=0.1, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, img, det=None, seg=None):
        if random.random() > self.p:
            return img, det, seg
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = np.float32(img)
        img[:, :, 2] += 255 * (self.rate * random.uniform(-1, 1))
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img, det, seg


class Rotate:
    def __init__(self, rate=0.1, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, img, det=None, seg=None):
        if random.random() > self.p:
            return img, det, seg
        d = random.uniform(-1, 1) * self.rate * 180
        scale = max(abs(math.sin(d / 180 * math.pi)),
                    abs(math.cos(d / 180 * math.pi)))
        r_matrix = cv2.getRotationMatrix2D(
            (img.shape[1] / 2, img.shape[0] / 2), d, scale)
        img = cv2.warpAffine(img, r_matrix, (img.shape[1], img.shape[0]))
        if det is not None:
            new_det = list()
            # detection n*(x1 y1 x2 y2 x3 y3 x4 y4)
            for i in range(4):
                point = np.concatenate(
                    [det[:, 2 * i:2 * i + 2],
                     np.ones([det.shape[0], 1])], 1)
                point = np.dot(r_matrix, point.transpose(1, 0)).transpose(1, 0)
                # point[:, :2] /= point[:, 2:]
                new_det.append(point[:, :2])
            det = np.concatenate(new_det, 1)
        if seg is not None:
            seg = cv2.warpAffine(seg, r_matrix, (seg.shape[1], seg.shape[0]))
        return img, det, seg


class Blur:
    def __init__(self, rate=0.1, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, img, det=None, seg=None):
        if random.random() > self.p:
            return img, det, seg
        rate = self.rate * random.random()
        ksize = (int(
            (rate * img.shape[0]) // 2) * 2 + 1, int(
                (rate * img.shape[1]) // 2) * 2 + 1)
        img = cv2.blur(img, ksize)
        return img, det, seg


class Noise:
    def __init__(self, rate=0.05, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, img, det=None, seg=None):
        if random.random() > self.p:
            return img, det, seg
        size = img.shape[0] * img.shape[1]
        amount = int(size * self.rate / 2 * random.random())

        x = np.random.randint(0, img.shape[0], amount)
        y = np.random.randint(0, img.shape[1], amount)
        img[x, y] = [0, 0, 0]

        x = np.random.randint(0, img.shape[0], amount)
        y = np.random.randint(0, img.shape[1], amount)
        img[x, y] = [255, 255, 255]
        return img, det, seg


class Normalize:
    def __call__(self, img, det=None, seg=None):
        img = np.float32(img)
        img -= np.mean(img)
        img /= np.std(img)
        return img, det, seg


class BGR2RGB:
    def __call__(self, img, det=None, seg=None):
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        return img, det, seg


class NHWC2NCHW:
    def __call__(self, img, det=None, seg=None):
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        if seg is not None:
            seg = seg.transpose(2, 0, 1)
            seg = np.ascontiguousarray(seg)
        return img, det, seg