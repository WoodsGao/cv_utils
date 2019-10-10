import cv2
import numpy as np
import os
import random
from threading import Thread
import time
# from queue import Queue


class ClassifyDataloader(object):
    def __init__(self,
                 path,
                 img_size=224,
                 batch_size=8,
                 augments=[],
                 balance=False,
                 multi_scale=False):
        self.path = path
        self.img_size = img_size
        self.batch_size = batch_size
        self.augments = augments
        self.multi_scale = multi_scale
        self.classes = os.listdir(path)
        self.classes.sort()
        self.data_list = list()

        if balance:
            weights = [
                len(os.listdir(os.path.join(path, c))) for c in self.classes
            ]
            max_weight = max(weights)
        for i_c, c in enumerate(self.classes):
            names = os.listdir(os.path.join(path, c))
            if balance:
                names *= (max_weight // len(names)) + 1
                names = names[:max_weight]
            for name in names:
                self.data_list.append([os.path.join(path, c, name), i_c])
        self.iter_times = len(self.data_list) // self.batch_size + 1
        self.max_len = 50
        self.queue = []
        self.scale = img_size
        self.batch_list = []
        t = Thread(target=self.run)
        t.setDaemon(True)
        t.start()

    def __iter__(self):
        return self

    def worker(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.scale, self.scale))
        for aug in self.augments:
            img, _, __ = aug(img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)

        return img

    def run(self):
        while True:
            while len(self.batch_list) > self.max_len:
                time.sleep(0.1)
            if len(self.queue) == 0:
                random.shuffle(self.data_list)
                self.queue = self.data_list

            if self.multi_scale:
                self.scale = random.randint(min(self.img_size // 40, 1),
                                            self.img_size // 20) * 32
                # print(self.scale)
            its = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            imgs = [self.worker(it[0]) for it in its]
            self.batch_list.append(
                [np.float32(imgs),
                 np.int64([it[1] for it in its])])

    def next(self):
        while len(self.batch_list) == 0:
            time.sleep(0.1)
        batch = self.batch_list[0]
        self.batch_list = self.batch_list[1:]

        return batch[0], batch[1]