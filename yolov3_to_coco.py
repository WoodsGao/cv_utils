import cv2
import os
import os.path as osp
import json
import argparse


def build_json(src, img_list):
    coco_json = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    for img in img_list:
        ih, iw, _ = cv2.imread(osp.join(src, 'images', img)).shape
        label_name = osp.join(src, 'labels',
                              ''.join(*(osp.splitext(img)[:-1])) + '.txt')
        if osp.exists(label_name):
            with open(label_name, 'r') as f:
                labels = f.read().split('\n')
                labels = [l.split(' ') for l in labels]
                labels = [l for l in labels if len(l) == 5]
            for label in labels:
                if not label:
                    False
                c, x, y, w, h = [float(l) for l in label]
                x *= iw
                y *= ih
                w *= iw
                h *= ih
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                coco_json['annotations'].append({
                    'area':
                    ih * iw,
                    'bbox': [x1, y1, w, h],
                    'category_id':
                    int(c),
                    'id':
                    len(coco_json['annotations']),
                    'image_id':
                    len(coco_json['images']),
                    'iscrowd':
                    0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
        coco_json['images'].append({
            'file_name': img,
            'id': len(coco_json['images']),
            'width': iw,
            'height': ih
        })
    return coco_json


def run(src, dst, val_ratio=0.3):
    os.makedirs(dst, exist_ok=True)
    if osp.exists(osp.join(src, 'train.txt')) and osp.exists(
            osp.join(src, 'val.txt')):
        with open(osp.join(src, 'train.txt'), 'r') as f:
            train_list = f.read().split('\n')
        with open(osp.join(src, 'val.txt'), 'r') as f:
            val_list = f.read().split('\n')
    else:
        names = os.listdir(osp.join(src, 'images'))
        split_idx = int(len(names) * val_ratio)
        train_list = names[:-split_idx]
        val_list = names[-split_idx:]
    train_json = build_json(src, train_list)
    val_json = build_json(src, val_list)
    cls_txt = osp.join(src, 'class_names.txt')
    if osp.exists(cls_txt):
        with open(cls_txt, 'r') as f:
            classes = [c for c in f.read().split('\n') if c]
    else:
        classes = [str(c) for c in range(80)]
    classes = [{
        "supercategory": c,
        "id": i,
        "name": c
    } for i, c in enumerate(classes)]
    train_json['categories'] = classes
    val_json['categories'] = classes
    with open(osp.join(dst, 'train.json'), 'w') as f:
        f.write(json.dumps(train_json))
    with open(osp.join(dst, 'val.json'), 'w') as f:
        f.write(json.dumps(val_json))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='src path')
    parser.add_argument('--dst', type=str, help='dst file path')
    parser.add_argument('--val-ratio',
                        type=float,
                        default=0.3,
                        help='dst file path')
    opt = parser.parse_args()
    run(opt.src, opt.dst, opt.val_ratio)
