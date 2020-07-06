#!/usr/bin/python3
import argparse
import os
import os.path as osp
import shutil

import cv2
from tqdm import tqdm


def labelme2seg(labelme_dir, output_dir):
    classes = []
    img_dir = osp.join(output_dir, 'images')
    label_dir = osp.join(output_dir, 'labels')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    names = os.listdir(labelme_dir)
    names = [n for n in names if osp.splitext(n)[-1] == '.json']
    for name in names:
        name = osp.join(labelme_dir, name)
        os.system('labelme_json_to_dataset %s && rm %s' % (name, name))
    names = os.listdir(labelme_dir)
    names = [n for n in names if osp.isdir(osp.join(labelme_dir, n))]
    for name in tqdm(names):
        shutil.move(osp.join(labelme_dir, name, 'label.png'),
                                osp.join(label_dir, name[:-5] + '.png'))
        shutil.move(osp.join(labelme_dir, name, 'img.png'),
                                osp.join(img_dir, name[:-5] + '.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('labelme', type=str, help='labelme file path')
    parser.add_argument('output', type=str, help='output file path')
    opt = parser.parse_args()
    labelme2seg(opt.labelme, opt.output)
