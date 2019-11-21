import os
import argparse
import random
import cv2
from tqdm import tqdm


def run(img_path, dst_dir, cut_times=10):
    os.makedirs(dst_dir, exist_ok=True)
    src = cv2.imread(img_path)
    img_path = os.path.basename(img_path)
    img_path = os.path.splitext(img_path)[0]
    h = src.shape[0] - 1
    w = src.shape[1] - 1
    for i in tqdm(range(cut_times)):
        min_h = random.randint(0, h - 1)
        max_h = random.randint(min_h + 1, h)
        min_w = random.randint(0, w - 1)
        max_w = random.randint(min_w + 1, w)
        c = src[min_h:max_h, min_w:max_w]
        name = img_path + '_%d.png' % i
        name = os.path.join(dst_dir, name)
        cv2.imwrite(name, c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='src path')
    parser.add_argument('--dst', type=str, help='dst file path')
    parser.add_argument('--times', '-t', type=int, default=10)
    opt = parser.parse_args()
    run(opt.src, opt.dst, opt.times)