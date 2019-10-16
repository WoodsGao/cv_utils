import os
import cv2
import numpy as np
from config import IMG_EXT
from tqdm import tqdm


def find_segment_classes(data_dir):
    classes = np.zeros([0, 3])
    names = os.listdir(data_dir)
    names = [name for name in names if os.path.splitext(name)[1] in IMG_EXT]
    for name in tqdm(names):
        img = cv2.imread(os.path.join(data_dir, name)).reshape(-1, 3)
        classes = np.unique(np.concatenate([classes, np.unique(img, axis=0)], 0), axis=0)
    output = []
    for ci, c in enumerate(classes):
        output.append(', '.join(['%d'] * 4) % (ci, *c))
    output = '\n'.join(output)
    with open(os.path.join(data_dir, 'classes.csv'), 'w') as f:
        f.write(output)


if __name__ == "__main__":
    find_segment_classes('/home/uisee/Downloads/data_road/training/labels')