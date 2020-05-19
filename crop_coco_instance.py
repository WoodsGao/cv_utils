#!/usr/bin/python3
import argparse
import json
import os
import os.path as osp
import random
from copy import deepcopy

import cv2
import numpy as np

from coco_utils import create_coco, find_all_img_anns, insert_img_anns, sort_coco


def crop_coco_instance(coco_path, img_root, output, square=False):
    save_path = osp.join(output, 'images')
    os.makedirs(save_path, exist_ok=True)
    with open(coco_path, 'r') as f:
        coco = f.read()
    coco = json.loads(coco)
    new_coco = create_coco(coco)
    img_info_list, anns_list = find_all_img_anns(coco)
    for img_info, anns in zip(img_info_list, anns_list):
        if len(anns) == 0:
            continue
        print(osp.join(img_root, img_info['file_name']))
        img = cv2.imread(osp.join(img_root, img_info['file_name']))
        img_name = osp.splitext(osp.basename(img_info['file_name']))[0]
        for ai, ann in enumerate(anns):
            iname = img_name + '_%05d.png' % ai
            seg = np.float32(ann['segmentation'])
            p = seg.reshape(-1, 2).transpose(1, 0).astype(np.int32)
            x1 = p[0].min()
            x2 = p[0].max()
            y1 = p[1].min()
            y2 = p[1].max()
            # square crop - it may be better for segmentation
            if square:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                r = max(x2 - x1, y2 - y1) // 2
                x1 = center[0] - r
                x2 = center[0] + r
                y1 = center[1] - r
                y2 = center[1] + r
            x1_ = max(random.randint(x1 - 50, x1 - 10), 0)
            y1_ = max(random.randint(y1 - 50, y1 - 10), 0)
            x2_ = min(random.randint(x2 + 10, x2 + 50), img.shape[1])
            y2_ = min(random.randint(y2 + 10, y2 + 50), img.shape[0])
            cut = img[y1_:y2_, x1_:x2_]
            seg[:, ::2] -= x1_
            seg[:, 1::2] -= y1_
            ann['bbox'] = np.array([x1 - x1_, y1 - y1_, x2 - x1,
                                    y2 - y1]).tolist()
            ann['segmentation'] = seg.tolist()
            cv2.imwrite(osp.join(save_path, iname), cut)
            print(osp.join('images', iname))
            img_info = {
                'file_name': osp.join('images', iname),
                'width': cut.shape[1],
                'height': cut.shape[0],
            }
            insert_img_anns(new_coco, img_info, [ann])
    save_path = osp.join(output, 'coco_instance.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(new_coco, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('coco', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--img-root', type=str, default='')
    parser.add_argument('-s', '--square', action='store_true')
    opt = parser.parse_args()
    if not opt.img_root:
        opt.img_root = osp.dirname(opt.coco)
    crop_coco_instance(opt.coco, opt.img_root, opt.output, opt.square)
