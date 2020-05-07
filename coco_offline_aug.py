#!/usr/bin/python3
import argparse
import json
import os
import os.path as osp
from copy import deepcopy

import cv2
import numpy as np
from imgaug import augmenters as ia
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from coco_utils import create_coco, find_anns, insert_img_anns, sort_coco

AUGS = ia.Sequential([
    ia.SomeOf(
        [0, 3],
        [
            ia.Dropout([0.015, 0.1]),  # drop 5% or 20% of all pixels
            # ia.Sharpen((0.0, 1.0)),  # sharpen the image
            # rotate by -45 to 45 degrees (affects heatmaps)
            # ia.ElasticTransformation(
            # alpha=(0, 10),
            # sigma=(0, 10)),  # apply water effect (affects heatmaps)
            # ia.PiecewiseAffine(scale=(0, 0.03), nb_rows=(2, 6), nb_cols=(2, 6)),
            # ia.GaussianBlur((0, 3)),
            ia.Fliplr(0.1),
            ia.Flipud(0.1),
            # ia.LinearContrast((0.5, 1)),
            # ia.AdditiveGaussianNoise(loc=(0, 10), scale=(0, 10))
        ],
        random_state=True),
    ia.Affine(scale=(0.8, 1.2),
              translate_percent=(-0.1, 0.1),
              rotate=(-90, 90),
              shear=(-0.1, 0.1))
])


def aug_img_anns(img, anns):
    augments = AUGS.to_deterministic()
    polygons = []
    for ann in anns:
        polygons.append(
            Polygon(
                np.float32(ann['segmentation'][0]).reshape(-1, 2),
                ann['category_id']))
    polygons = PolygonsOnImage(polygons, img.shape)
    img = augments.augment_image(img)
    polygons = augments.augment_polygons(polygons).polygons
    anns = []
    for p in polygons:
        seg = p.exterior.reshape(-1).tolist()
        xs = seg[::2]
        ys = seg[1::2]
        if len([x for x in xs if x >= img.shape[1] - 1 or x <= 0]) > 0:
            continue
        if len([y for y in ys if y >= img.shape[0] - 1 or y <= 0]) > 0:
            continue
        x1 = min(xs)
        y1 = min(ys)
        w = max(xs) - x1
        h = max(ys) - y1
        anns.append({
            'area': w * h,
            'bbox': [x1, y1, w, h],
            'category_id': p.label,
            'iscrowd': 0,
            'segmentation': [seg]
        })
    return img, anns


def coco_offline_aug(coco_path, img_root, output, repeats=1):
    with open(coco_path, 'r') as f:
        coco = f.read()
    coco = json.loads(coco)
    new_coco = create_coco(coco)
    save_dir = osp.join(output, 'images')
    os.makedirs(save_dir, exist_ok=True)
    for img_info in coco['images']:
        anns = find_anns(coco, img_info)
        img = cv2.imread(osp.join(img_root, img_info['file_name']))
        img_name = osp.splitext(osp.basename(img_info['file_name']))[0]
        anns = find_anns(coco, img_info)
        for ri in range(repeats):
            aug_img, aug_anns = aug_img_anns(img.copy(), anns)
            iname = img_name + '_%05d.png' % ri
            cv2.imwrite(osp.join(save_dir, iname), aug_img)
            print(osp.join(save_dir, iname))
            img_info = {
                'file_name': osp.join('images', iname),
                'width': aug_img.shape[1],
                'height': aug_img.shape[0],
            }
            insert_img_anns(new_coco, img_info, aug_anns)
    save_path = osp.join(output, 'coco_aug.json')
    # print(new_coco)
    with open(save_path, 'w') as f:
        f.write(json.dumps(new_coco, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('coco', type=str)
    parser.add_argument('--img-root', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--repeats', type=int, default=1)
    opt = parser.parse_args()
    if not opt.img_root:
        opt.img_root = osp.dirname(opt.coco)
    coco_offline_aug(opt.coco, opt.img_root, opt.output, opt.repeats)
