import os
import os.path as osp
import argparse
import json
import numpy as np
import cv2
from copy import deepcopy
from coco_utils import find_anns, create_coco, insert_img_anns, sort_coco


def crop_coco_instance(coco_path, img_root, output):
    save_path = osp.join(output, 'images')
    os.makedirs(save_path, exist_ok=True)
    with open(coco_path, 'r') as f:
        coco = f.read()
    coco = json.loads(coco)
    new_coco = create_coco(coco)
    for img_info in coco['images']:
        anns = find_anns(coco, img_info)
        if len(anns) == 0:
            continue
        print(osp.join(img_root, img_info['file_name']))
        img = cv2.imread(osp.join(img_root, img_info['file_name']))
        img_name = osp.splitext(osp.basename(img_info['file_name']))[0]
        for ai, ann in enumerate(anns):
            x1, y1, w, h = np.int32(ann['bbox'])
            cut = img[y1:y1 + h, x1:x1 + w]
            iname = img_name + '_%05d.png' % ai
            ann['bbox'][0] -= x1
            ann['bbox'][1] -= y1
            seg = np.float32(ann['segmentation'])
            seg[:, ::2] -= x1
            seg[:, 1::2] -= y1
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
    parser.add_argument('--coco', type=str)
    parser.add_argument('--img-root', type=str, default='')
    parser.add_argument('--output', type=str)
    opt = parser.parse_args()
    if not opt.img_root:
        opt.img_root = osp.dirname(opt.coco)
    crop_coco_instance(opt.coco, opt.img_root, opt.output)
