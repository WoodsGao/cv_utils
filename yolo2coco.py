import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np

from coco_utils import create_coco, insert_img_anns


def yolo2coco(img_root):
    coco = create_coco()
    cls_txt = osp.join(img_root, 'class_names.txt')
    if osp.exists(cls_txt):
        with open(cls_txt, 'r') as f:
            classes = [c for c in f.read().split('\n') if c]
        for ci in range(len(coco['categories'])):
            if ci < len(classes):
                coco['categories'][ci]['name'] = classes[ci]
    coco['categories'] = [c for c in coco['categories'] if c['name'] is not None]
    img_list = os.listdir(osp.join(img_root, 'images'))
    for img in img_list:
        ih, iw, _ = cv2.imread(osp.join(img_root, 'images', img)).shape
        label_name = osp.join(img_root, 'labels',
                              ''.join(*(osp.splitext(img)[:-1])) + '.txt')
        anns = []
        if osp.exists(label_name):
            labels = np.loadtxt(label_name, delimiter=' ', ndmin=2)
            for (c, x, y, w, h) in labels:
                x *= iw
                y *= ih
                w *= iw
                h *= ih
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                anns.append({
                    'area':
                    w * h,
                    'bbox': [x1, y1, w, h],
                    'category_id':
                    int(c),
                    'iscrowd':
                    0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
        img_info = {
            'file_name': osp.join('images', img),
            'width': iw,
            'height': ih
        }
        insert_img_anns(coco, img_info, anns)
    with open(osp.join(img_root, 'coco.json'), 'w') as f:
        f.write(json.dumps(coco, indent=4, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-root', type=str)
    opt = parser.parse_args()
    yolo2coco(opt.img_root)
