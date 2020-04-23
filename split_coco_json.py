import os
import os.path as osp
import argparse
import json
import cv2
import random
from coco_utils import find_anns, create_coco, insert_img_anns, sort_coco


def split_coco_image(coco_path, val_ratio, shuffle):
    with open(coco_path, 'r') as f:
        coco = f.read()
    coco = json.loads(coco)
    train_coco = create_coco(coco)
    val_coco = create_coco(coco)
    if shuffle:
        random.shuffle(coco['images'])
    img_len = len(coco['images'])
    for img_i in range(img_len):
        img_info = coco['images'][img_i]
        anns = find_anns(coco, img_info)
        if img_i > img_len * (1-val_ratio):
            train_coco = insert_img_anns(train_coco, img_info, anns)
        else:
            val_coco = insert_img_anns(val_coco, img_info, anns)

    train_coco = sort_coco(train_coco)
    save_path = osp.join(osp.dirname(coco_path), 'train.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(coco, indent=4, sort_keys=True))

    val_coco = sort_coco(val_coco)
    save_path = osp.join(osp.dirname(coco_path), 'val.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(coco, indent=4, sort_keys=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', type=str)
    parser.add_argument('-v', '--val_ratio', default=0.3, type=float)
    parser.add_argument('-s', '--shuffle', action='store_true')
    opt = parser.parse_args()
    split_coco_image(opt.coco, opt.val_ratio, opt.shuffle)
