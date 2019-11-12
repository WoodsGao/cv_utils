import os
import cv2
import shutil
from tqdm import tqdm
from utils import *

def run(src_path, dst_path, horizontal=True):
    rebuild_dir(dst_path)
    names = os.listdir(src_path)
    def worker(name):
        img = cv2.imread(os.path.join(src_path, name))
        if horizontal:
            img = img[:, :img.shape[1]//2, :]
        else:
            img = img[:img.shape[0]//2, :, :]
        cv2.imwrite(os.path.join(dst_path, name.replace(
            '.tiff', '.png')), img)

    for name in tqdm(names):
        if '.tiff' not in name and '.png' not in name:
            continue
        worker(name)


if __name__ == "__main__":
    run('/home/uisee/Projects/yolov3/data/samples', '/home/uisee/Projects/yolov3/data/samples1')
