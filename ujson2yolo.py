# coding=utf-8
import argparse
import json
import os
import os.path as osp
import shutil

import cv2
from tqdm import tqdm


def ujson2yolo(ann, cls_file, dst, lines=False):
    os.makedirs(osp.join(dst, 'images'), exist_ok=True)
    os.makedirs(osp.join(dst, 'labels'), exist_ok=True)
    with open(ann, 'r') as f:
        if lines:
            ann_list = [json.loads(line)
                        for line in f.read().split('\n') if line]
        else:
            ann_list = json.loads(f.read())
    with open(cls_file, 'r', encoding='utf-8') as f:
        lines = f.read()
        cls_list = json.loads(lines)
    if osp.exists(osp.join(dst, 'classes.names')):
        with open(osp.join(dst, 'classes.names'), 'r') as f:
            lines = f.read().split('\n')
            lines = [line for line in lines if line]
            yolo_cls = lines
    else:
        yolo_cls = []
    for a in tqdm(ann_list):
        img_path = a['url_image'].split('/')
        img_name = img_path[-1]
        label_name = osp.splitext(img_name)[0] + '.txt'
        img_path = osp.join(osp.dirname(ann), *img_path[3:])
        shutil.copy(img_path, osp.join(dst, 'images', img_name))
        img = cv2.imread(img_path)
        bbox_list = []
        for result in a['result']:
            c = result['tagtype']
            if cls_list.get(c):
                if cls_list[c]['datatype'] not in ['polygon', 'box']:
                    continue
            else:
                continue
            if cls_list[c]['datatype'] == 'box':
                box = True
            else:
                box = False 
            if c not in yolo_cls:
                yolo_cls.append(c)
                c = len(yolo_cls) - 1
            else:
                c = yolo_cls.index(c)
            if box:
                bbox = result['data']
                if isinstance(bbox, str):
                    bbox = json.loads(bbox)
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = xmin + bbox[2]
                ymax = ymin + bbox[3]
            else:
                bbox = json.loads(result['data'])
                bbox = [b[1:] for b in bbox if b[0] == 'L']
                xs = [b[0] for b in bbox]
                ys = [b[1] for b in bbox]
                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)
            x = (xmin + xmax) / 2 / img.shape[1]
            y = (ymin + ymax) / 2 / img.shape[0]
            w = (xmax - xmin) / img.shape[1]
            h = (ymax - ymin) / img.shape[0]
            bbox_list.append('%d %lf %lf %lf %lf' % (c, x, y, w, h))
        if len(bbox_list):
            with open(osp.join(dst, 'labels', label_name), 'w') as f:
                f.write('\n'.join(bbox_list))
    print(yolo_cls)
    with open(osp.join(dst, 'classes.names'), 'w') as f:
        f.write('\n'.join(yolo_cls))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', type=str)
    parser.add_argument('--cls', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--lines', action='store_true')
    opt = parser.parse_args()
    ujson2yolo(opt.ann, opt.cls, opt.dst, opt.lines)
