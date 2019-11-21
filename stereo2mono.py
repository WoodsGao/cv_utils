import os
import cv2
import shutil
from tqdm import tqdm
import argparse
from utils import *
from concurrent.futures import ThreadPoolExecutor

def run(src_path, dst_path, vertical=False):
    rebuild_dir(dst_path)
    names = os.listdir(src_path)
    def worker(name):
        img = cv2.imread(os.path.join(src_path, name))
        if not vertical:
            img = img[:, :img.shape[1]//2, :]
        else:
            img = img[:img.shape[0]//2, :, :]
        cv2.imwrite(os.path.join(dst_path, name.replace(
            '.tiff', '.png')), img)
        return True

    names = [name for name in names if os.path.splitext(name)[-1] in ['.tiff', '.png', '.jpg', '.jpeg']]
    for name in tqdm(names):
        worker(name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='src file path')
    parser.add_argument('--dst', type=str, help='dst file path')
    parser.add_argument('--vertical', action='store_true', help='cut in axis y')
    opt = parser.parse_args()
    run(opt.src, opt.dst, opt.vertical)
