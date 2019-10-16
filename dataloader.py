import cv2
import numpy as np
import os
import random
from threading import Thread
import time
from copy import deepcopy
from .config import IMG_EXT


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
            names = [
                name for name in names if os.path.splitext(name)[1] in IMG_EXT
            ]
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


class SegmentDataloader(Dataloader):
    def build_data_list(self):
        with open(os.path.join(self.data_dir, 'labels', 'classes.csv'), 'r') as f:
            lines = [l.split(',') for l in f.readlines()]
            lines = [[l[0], np.uint8(l[1:])] for l in lines if len(l) == 4]
            self.classes = lines
        image_dir = os.path.join(self.data_dir, 'images')
        label_dir = os.path.join(self.data_dir, 'labels')
        names = os.listdir(image_dir)
        names = [
            name for name in names if os.path.splitext(name)[1] in IMG_EXT
        ]
        for name in names:
            if os.path.exists(os.path.join(label_dir, names)):
                self.data_dir.append([
                    os.path.join(image_dir, names),
                    os.path.join(label_dir, names)
                ])

    def worker(self, message):
        img = cv2.imread(message[0])
        img = cv2.resize(img, (self.img_size, self.img_size))
        seg_rgb = cv2.imread(message[1])
        seg = np.zeros([seg_rgb.shape[0], seg_rgb.shape[1], self.classes])
        for ci, c in enumerate(self.classes):
            seg[(seg_rgb == c[1]).all(2), ci] = 1
        seg = cv2.resize(seg, (self.img_size, self.img_size))
        for aug in self.augments:
            img, _, seg = aug(img)
        seg[seg > 0.5] = 1
        seg[seg < 1] = 0
        return img, seg
