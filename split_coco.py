#!/usr/bin/python3
import argparse
import json
import os
import os.path as osp
import random

import cv2
from tqdm import tqdm

from coco_utils import (create_coco, find_all_img_anns, insert_img_anns,
                        sort_coco)


def split_coco(coco_path, val_ratio, shuffle):
    with open(coco_path, 'r') as f:
        coco = f.read()
    coco = json.loads(coco)
    train_coco = create_coco(coco)
    val_coco = create_coco(coco)

    img_info_list, anns_list = find_all_img_anns(coco)
    indexs = list(range(len(img_info_list)))
    if shuffle:
        random.shuffle(indexs)
    for ii, i in enumerate(indexs):
        img_info = img_info_list[i]
        anns = anns_list[i]
        if ii < len(img_info_list) * (1 - val_ratio):
            train_coco = insert_img_anns(train_coco, img_info, anns)
        else:
            val_coco = insert_img_anns(val_coco, img_info, anns)

    save_path = osp.join(osp.dirname(coco_path), 'train.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(train_coco, indent=4, sort_keys=True))

    save_path = osp.join(osp.dirname(coco_path), 'val.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(val_coco, indent=4, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('coco', type=str)
    parser.add_argument('-v', '--val_ratio', default=0.3, type=float)
    parser.add_argument('-s', '--shuffle', action='store_true')
    opt = parser.parse_args()
    split_coco(opt.coco, opt.val_ratio, opt.shuffle)
