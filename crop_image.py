#!/usr/bin/python3
import argparse
import json
import os
import os.path as osp
from copy import deepcopy

import cv2

from coco_utils import create_coco, find_anns, insert_img_anns, sort_coco


def crop_images(img, output, img_size, steps):
    print(img_size)
    os.makedirs(output, exist_ok=True)
    img = cv2.imread(img)
    for i in range(0, img.shape[1] - img_size[0], steps[0]):
        for j in range(0, img.shape[0] - img_size[1], steps[1]):
            cv2.imwrite(osp.join(output, '%d_%d.png' % (i, j)),
                        img[j:j + img_size[1], i:i + img_size[0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--img-size', default=[1000, 1000], type=int, nargs=2)
    parser.add_argument('--steps', default=[500, 500], type=int, nargs=2)
    opt = parser.parse_args()

    crop_images(opt.img, opt.output, opt.img_size, opt.steps)
