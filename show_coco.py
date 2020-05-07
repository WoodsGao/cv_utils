import argparse
import json
import os
import os.path as osp
import random
from copy import deepcopy

import cv2
import numpy as np

from coco_utils import find_all_img_anns


def show_coco(coco_path, img_root, output):
    save_path = osp.join(output, 'show')
    os.makedirs(save_path, exist_ok=True)
    with open(coco_path, 'r') as f:
        coco = f.read()
    coco = json.loads(coco)
    img_info_list, anns_list = find_all_img_anns(coco)
    for img_info, anns in zip(img_info_list, anns_list):
        if len(anns) == 0:
            continue
        print(osp.join(img_root, img_info['file_name']))
        img = cv2.imread(osp.join(img_root, img_info['file_name']))
        img_name = osp.splitext(osp.basename(img_info['file_name']))[0]
        for ai, ann in enumerate(anns):
            seg = np.float32(ann['segmentation'])
            p = seg.reshape(-1, 2).astype(np.int32)
            if len(p) == 1:
                cv2.circle(img, tuple(p[0]), 2, (0, 0, 255), -1)
            elif len(p) == 2:
                cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 0, 255), 2)
            else:
                contour = p.reshape(-1, 1, 2)
                cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
        cv2.imwrite(osp.join(save_path, img_name + '.png'), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('coco', type=str)
    parser.add_argument('--img-root', type=str, default='')
    parser.add_argument('--output', type=str)
    opt = parser.parse_args()
    if not opt.img_root:
        opt.img_root = osp.dirname(opt.coco)
    show_coco(opt.coco, opt.img_root, opt.output)
