import os
import os.path as osp
import argparse
import json
import numpy as np
import cv2
from copy import deepcopy
from coco_utils import find_anns, create_coco, insert_img_anns, sort_coco
from tqdm import tqdm


def coco_seg2det(coco_path, img_root, scale=5):
    with open(coco_path, 'r') as f:
        coco = f.read()
    coco = json.loads(coco)
    new_coco = create_coco(coco)
    for img_info in tqdm(coco['images']):
        anns = find_anns(coco, img_info)
        if len(anns) == 0:
            continue
        print(osp.join(img_root, img_info['file_name']))
        img = cv2.imread(osp.join(img_root, img_info['file_name']))
        new_anns = []
        for ai, ann in enumerate(anns):
            seg_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
            points = ann['segmentation']
            points = np.int32(points).reshape(-1, 2)
            seg_canvas = cv2.fillPoly(seg_canvas, [points], (255, 255, 255), 0)
            down_size = (img.shape[1] // scale, img.shape[0] // scale)
            seg_canvas = cv2.resize(seg_canvas, down_size)
            for j, i in np.stack(np.where(seg_canvas == 255), 1).tolist():
                x1 = i * scale
                x2 = x1 + scale
                y1 = j * scale
                y2 = y1 + scale
                new_anns.append({
                    'area':
                    scale**2,
                    'bbox': [x1, y1, scale, scale],
                    'category_id':
                    ann['category_id'],
                    'iscrowd':
                    0,
                    'segmentation': [[x1, y1, x1, y2, x2, y2, x2, y1]]
                })
                # img[y1:y2, x1:x2] = 255
        insert_img_anns(new_coco, img_info, new_anns)
    save_path = osp.join(osp.dirname(coco_path), 'coco_seg2det.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(new_coco, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', type=str)
    parser.add_argument('--img-root', type=str, default='')
    parser.add_argument('--scale', type=int, default=8)
    opt = parser.parse_args()
    if not opt.img_root:
        opt.img_root = osp.dirname(opt.coco)
    coco_seg2det(opt.coco, opt.img_root, opt.scale)
