import os
import cv2
from tqdm import tqdm
from augments import PerspectiveProject, HSV_S, HSV_H, HSV_V
from augments import Rotate, Blur, Noise
from .config import IMG_EXT


def cls_generator(data_dir, augment_list=[]):
    img_paths = []
    class_names = os.listdir(data_dir)
    for class_name in class_names:
        names = os.listdir(os.path.join(data_dir, class_name))
        names = [
            name for name in names if os.path.splitext(name)[1] in IMG_EXT
        ]
        img_paths += [
            os.path.join(data_dir, class_name, name) for name in names
        ]
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        for aug in augment_list:
            img, _, __ = aug(img)
        if img is not None:
            cv2.imwrite(img_path, img)


if __name__ == "__main__":
    cls_generator('/home/uisee/Datasets/road_mark/train', [
        PerspectiveProject(),
        HSV_S(),
        HSV_H(),
        HSV_V(),
        Rotate(),
        Blur(),
        Noise()
    ])
