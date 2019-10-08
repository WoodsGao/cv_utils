from config import IMG_EXT
import os
import cv2
import numpy as np
from utils import normalize


def get_features(img, processor_list=[], linear=True):
    features = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for processor in processor_list:
        feature = processor(img.copy())
        feature = normalize(feature)
        if linear:
            feature = feature.reshape(-1)
        features.append(feature)
    features = np.concatenate(features)
    return features


def simple_dataloader(data_dir, processor_list=[], linear=True):
    inputs = []
    targets = []
    class_names = os.listdir(data_dir)
    for ci, class_name in enumerate(class_names):
        c_dir = os.path.join(data_dir, class_name)
        names = os.listdir(c_dir)
        names = [
            name for name in names if os.path.splitext(name)[1] in IMG_EXT
        ]
        names.sort()
        for name in names:
            img = cv2.imread(os.path.join(c_dir, name))
            inputs.append(get_features(img, processor_list, linear))
            targets.append(ci)
    inputs = np.float32(inputs)
    targets = np.int64(targets)
    return inputs, targets
