import cv2
import numpy as np
import os
import random
from threading import Thread
import time
from copy import deepcopy


class Dataloader:
    def __init__(self,
                 data_dir,
                 img_size=224,
                 batch_size=8,
                 augments=[],
                 max_len=50):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.augments = augments
        self.data_list = list()
        self.max_len = max_len
        self.queue = []
        self.batch_list = []
        self.iter_times = 0
        self.classes = []
        self.build_data_list()
        self.iter_times = (len(self.data_list) - 1) // self.batch_size + 1
        self.run_thread()

    def build_data_list(self):
        pass

    def run_thread(self):
        t = Thread(target=self.run)
        t.setDaemon(True)
        t.start()

    def __iter__(self):
        return self

    def worker(self, message):
        return False, False

    def run(self):
        while True:
            while len(self.batch_list) > self.max_len:
                time.sleep(0.1)
            if len(self.queue) == 0:
                self.queue = deepcopy(self.data_list)
                random.shuffle(self.queue)
            its = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            imgs = []
            labels = []
            for it in its:
                message = self.worker(it)
                imgs.append(message[0])
                labels.append(message[1])
            self.batch_list.append([np.float32(imgs), np.float32(labels)])

    def next(self):
        while len(self.batch_list) == 0:
            time.sleep(0.1)
        batch = self.batch_list.pop(0)
        return batch[0], batch[1]


class ClassifyDataloader(Dataloader):
    def build_data_list(self):
        self.classes = os.listdir(self.data_dir)
        self.classes.sort()
        for ci, c in enumerate(self.classes):
            names = os.listdir(os.path.join(self.data_dir, c))
            for name in names:
                target = np.zeros(len(self.classes))
                target[ci] = 1
                self.data_list.append(
                    [os.path.join(self.data_dir, c, name), target])

    def worker(self, message):
        img = cv2.imread(message[0])
        img = cv2.resize(img, (self.img_size, self.img_size))
        for aug in self.augments:
            img, _, __ = aug(img)
        return img, message[1]


# class SegmentDataloader:
