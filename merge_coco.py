import argparse
import json
import os
import os.path as osp
import random

import cv2
from tqdm import tqdm

from coco_utils import (create_coco, find_all_img_anns, insert_img_anns,
                        sort_coco)


def merge_coco(path):
    merge_coco = None
    coco_list = os.listdir(path)
    coco_list = [c for c in coco_list if osp.splitext(c)[-1] == '.json']
    for coco_path in coco_list:
        coco_path = osp.join(path, coco_path)
        with open(coco_path, 'r') as f:
            coco = f.read()
        coco = json.loads(coco)
        if merge_coco is None:
            merge_coco = create_coco(coco)

        img_info_list, anns_list = find_all_img_anns(coco)
        for i in range(len(img_info_list)):
            img_info = img_info_list[i]
            anns = anns_list[i]
            merge_coco = insert_img_anns(merge_coco, img_info, anns)

    save_path = osp.join(osp.dirname(coco_path), 'merge.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(merge_coco, indent=4, sort_keys=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    opt = parser.parse_args()
    merge_coco(opt.path)
