import numpy as np
import matplotlib.pyplot as plt
import os
import time
import shutil
import cv2
from config import *


def rebuild_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    while os.path.exists(path):
        time.sleep(0.01)
    os.makedirs(path)


def move_dir(src, dst):
    for name in os.listdir(src):
        shutil.move(os.path.join(src, name), dst)
    while os.listdir(src):
        time.sleep(0.01)
    shutil.rmtree(src)
    while os.path.exists(src):
        time.sleep(0.01)


def read_2d_list(path):
    """读取n*m的2d列表

    Arguments:
        path {str} -- txt路径

    Returns:
        list -- 2d列表
    """
    if not os.path.exists(path):
        return list()
    with open(path, 'r') as f:
        lines = [line for line in f.readlines() if line.replace(' ', '')]
    lines = [line.split() for line in lines]
    lines = [[float(i) for i in line if i] for line in lines]
    return lines


def write_2d_list(path, objs, mode='w'):
    """将n*m的2d列表写入文件

    Arguments:
        path {str} -- 写入txt路径
        objs {list} -- 2d列表

    Returns:
        bool -- 写入成功返回True
    """
    if len(objs) == 0:
        return False
    with open(path, mode) as f:
        for obj in objs:
            f.write(' '.join([str(o) for o in obj]) + '\n')
    return True


def normalize(data):
    data = data.copy()
    data -= np.mean(data)
    data /= np.std(data)
    return data


def get_features(img, processor_list=[]):
    features = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for processor in processor_list:
        feature = processor(img.copy())
        feature = normalize(feature).reshape(-1)
        features.append(feature)
    features = np.concatenate(features)
    return features


def simple_dataloader(data_dir, processor_list=[]):
    inputs = []
    targets = []
    class_names = os.listdir(data_dir)
    for ci, class_name in enumerate(class_names):
        c_dir = os.path.join(data_dir, class_name)
        names = os.listdir(c_dir)
        names = [name for name in names if os.path.splitext(name)[1] in IMG_EXT]
        names.sort()
        for name in names:
            img = cv2.imread(os.path.join(c_dir, name))
            inputs.append(get_features(img, processor_list))
            targets.append(ci)
    inputs = np.float32(inputs)
    targets = np.int64(targets)
    return inputs, targets
