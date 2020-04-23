import argparse
import json
import cv2
import numpy as np
import os
import os.path as osp
from coco_utils import create_coco, insert_img_anns


def labelme2coco(path, img_root=''):
    if not img_root:
        img_root = path
    coco = create_coco()
    categories = []
    for data in os.listdir(path):
        with open(osp.join(path, data), 'r') as f:
            data = json.loads(f.read())
        img = cv2.imread(osp.join(img_root, data['imagePath']))
        h, w, c = img.shape
        img_info = {'file_name': data['imagePath'], 'width': w, 'height': h}
        anns = []
        for shapes in data['shapes']:
            label = shapes['label']
            if label not in categories:
                coco['categories'][len(categories)]['name'] = label
                categories.append(label)
            cid = categories.index(label)

            points = np.float32(shapes['points']).reshape(-1).tolist()
            xs = points[::2]
            ys = points[1::2]
            anns.append({
                'area':
                w * h,
                'bbox':
                [min(xs),
                 min(ys),
                 max(xs) - min(xs),
                 max(ys) - min(ys)],
                'category_id':
                cid,
                'iscrowd':
                0,
                'segmentation': [points]
            })
        coco = insert_img_anns(coco, img_info, anns)
    save_path = osp.join(osp.dirname(path), '../coco.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(coco, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--img-root', default='', type=str)
    opt = parser.parse_args()
    labelme2coco(opt.path, opt.img_root)
