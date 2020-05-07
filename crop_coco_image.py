#!/usr/bin/python3
import argparse
import json
import os
import os.path as osp
from copy import deepcopy

import cv2

from coco_utils import create_coco, find_anns, insert_img_anns, sort_coco


def crop_img_ann(img, anns, img_size, steps):
    img_array = []
    anns_array = []
    for i in range(0, img.shape[1] - img_size[0], steps[0]):
        for j in range(0, img.shape[0] - img_size[1], steps[1]):
            if img[j:j + img_size[1], i:i + img_size[0]].sum() == 0:
                continue
            img_array.append(img[j:j + img_size[1], i:i + img_size[0]])
            new_anns = []
            for ann in anns:
                xs = ann['segmentation'][0][::2]
                ys = ann['segmentation'][0][1::2]
                if len([x for x in xs if x >= i + img_size[0] - 1 or x <= i
                        ]) > 0:
                    continue
                if len([y for y in ys if y >= j + img_size[1] - 1 or y <= j
                        ]) > 0:
                    continue
                ann = deepcopy(ann)
                for si in range(0, len(ann['segmentation'][0]), 2):
                    ann['segmentation'][0][si] -= i
                    ann['segmentation'][0][si + 1] -= j
                xs = ann['segmentation'][0][::2]
                ys = ann['segmentation'][0][1::2]
                xmin = min(xs)
                ymin = min(ys)
                w = max(xs) - xmin
                h = max(ys) - ymin
                ann['bbox'] = [xmin, ymin, w, h]
                new_anns.append(ann)
            anns_array.append(new_anns)
    return img_array, anns_array


def crop_coco_image(coco_path, img_root, output, img_size, steps):
    print(img_size)
    save_path = osp.join(output, 'images')
    os.makedirs(save_path, exist_ok=True)
    with open(coco_path, 'r') as f:
        coco = f.read()
    coco = json.loads(coco)
    new_coco = create_coco(coco)
    for img_info in coco['images']:
        if img_info['width'] <= img_size[0] and img_info['height'] <= img_size[
                1]:
            anns = find_anns(coco, img_info)
            insert_img_anns(new_coco, img_info, anns)
            continue
        print(osp.join(img_root, img_info['file_name']))
        img = cv2.imread(osp.join(img_root, img_info['file_name']))
        img_name = osp.splitext(osp.basename(img_info['file_name']))[0]
        anns = find_anns(coco, img_info)
        imgs_split, anns_split = crop_img_ann(img, anns, img_size, steps)
        for si, (img, anns) in enumerate(zip(imgs_split, anns_split)):
            iname = img_name + '_%05d.png' % si
            cv2.imwrite(osp.join(save_path, iname), img)
            print(osp.join('images', iname))
            img_info = {
                'file_name': osp.join('images', iname),
                'width': img.shape[1],
                'height': img.shape[0],
            }
            insert_img_anns(new_coco, img_info, anns)
    save_path = osp.join(output, 'coco_split.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(new_coco, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('coco', type=str)
    parser.add_argument('--img-root', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--img-size', default='1000', type=str)
    parser.add_argument('--steps', default='', type=str)
    opt = parser.parse_args()
    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0])] * 2
    else:
        img_size = [int(x) for x in img_size]
    steps = opt.steps.split(',') if opt.steps else []
    if len(steps) == 1:
        steps = [int(steps[0])] * 2
    elif len(steps) == 2:
        steps = [int(x) for x in steps]
    else:
        steps = [img_size[0] // 2, img_size[1] // 2]
    if not opt.img_root:
        opt.img_root = osp.dirname(opt.coco)
    crop_coco_image(opt.coco, opt.img_root, opt.output, img_size, steps)
