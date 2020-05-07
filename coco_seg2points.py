#!/usr/bin/python3
import argparse
import json
import os
import os.path as osp
import random
from copy import deepcopy

import cv2
import numpy as np

from coco_utils import create_coco, find_anns, insert_img_anns, sort_coco


def coco_seg2points(coco_path, img_root, output):
    save_path = osp.join(output, 'images')
    os.makedirs(save_path, exist_ok=True)
    with open(coco_path, 'r') as f:
        coco = f.read()
    coco = json.loads(coco)
    new_coco = create_coco(coco)
    for img_info in coco['images']:
        anns = find_anns(coco, img_info)
        if len(anns) == 0:
            continue
        print(osp.join(img_root, img_info['file_name']))
        new_anns = []
        for ai, ann in enumerate(anns):
            points = np.int32(ann['segmentation']).reshape(-1, 2)
            for p in points:
                x, y = p.tolist()
                x1 = max(0, x - 5)
                y1 = max(0, y - 5)
                x2 = min(img_info['width'], x + 5)
                y2 = min(img_info['height'], y + 5)
                w = x2 - x1
                h = y2 - y1
                new_anns.append({
                    'area':
                    w * h,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'category_id':
                    ann['category_id'],
                    'iscrowd':
                    0,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
        insert_img_anns(new_coco, img_info, new_anns)
    save_path = osp.join(output, 'coco_seg2points.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(new_coco, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('coco', type=str)
    parser.add_argument('--img-root', type=str, default='')
    parser.add_argument('--output', type=str)
    opt = parser.parse_args()
    if not opt.img_root:
        opt.img_root = osp.dirname(opt.coco)
    coco_seg2points(opt.coco, opt.img_root, opt.output)
