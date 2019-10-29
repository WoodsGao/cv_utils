import cv2
import numpy as np
import os
import random
from threading import Thread
# from concurrent.futures import ProcessPoolExecutor
import time
from copy import deepcopy


class Dataloader:
    def __init__(self,
                 path,
                 img_size=224,
                 batch_size=8,
                 augments=[],
                 max_len=50, 
                 multi_scale=False,
                 *args,
                 **kargs):
        self.path = path
        self.img_size = img_size
        self.scale = img_size
        self.multi_scale = multi_scale
        self.batch_size = batch_size
        self.augments = augments
        self.data_list = list()
        self.max_len = max_len
        self.queue = []
        self.batch_list = []
        self.iter_times = 0
        self.classes = []
        self.args = args
        self.kargs = kargs
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

    def worker(self, message=None):
        return None

    def run(self):
        # TODO multi processing
        # pool = ProcessPoolExecutor()
        while True:
            while len(self.batch_list) > self.max_len:
                time.sleep(0.1)
            if len(self.queue) == 0:
                random.shuffle(self.data_list)
                self.queue = deepcopy(self.data_list)

            # multi scale (0.5x - 1.5x)
            if self.multi_scale and random.random() > 0.5:
                self.scale = int(random.uniform(self.img_size // 64, self.img_size // 21))
                self.scale = self.scale if self.scale > 0 else 1
                self.scale *= 32

            its = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            # messages = pool.map(self.worker, its)
            messages = [self.worker(it) for it in its]
            self.batch_list.append([np.float32([m[0] for m in messages]), np.float32([m[1] for m in messages])])

    def next(self):
        while len(self.batch_list) == 0:
            time.sleep(0.1)
        batch = self.batch_list.pop(0)
        return batch[0], batch[1]

