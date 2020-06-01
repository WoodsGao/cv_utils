import argparse
import os
import os.path as osp
import random
import shutil


def random_select_file(path, outputs, ratio=1e-2):
    os.makedirs(outputs, exist_ok=True)
    for root, folders, files in os.walk(path):
        for f in files:
            if osp.splitext(f)[-1] in ['.jpg', '.png', '.tiff']:
                if random.random() < ratio:
                    shutil.copy(osp.join(root, f), osp.join(outputs, f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('outputs', type=str)
    parser.add_argument('-r', '--ratio', type=float, default=1e-2)
    opt = parser.parse_args()
    random_select_file(opt.path, opt.outputs, opt.ratio)
