#!/usr/bin/python3
import argparse
import json
import os
import os.path as osp

import numpy as np
from sklearn.cluster import KMeans

from coco_utils import find_all_img_anns


def kmeans_anchor(coco_path, n_clusters, img_size):
    wh_list = []
    with open(coco_path, 'r') as f:
        coco = f.read()
    coco = json.loads(coco)

    img_info_list, anns_list = find_all_img_anns(coco)
    for i in range(len(img_info_list)):
        img_info = img_info_list[i]
        anns = anns_list[i]
        for ann in anns:
            wh_list.append([ann['bbox'][2] / img_info['width'], ann['bbox'][3] / img_info['height']])

    wh_list = np.float32(wh_list)
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(wh_list)
    anchors = np.float32(estimator.cluster_centers_)
    anchors[:, 0] *= img_size[0]
    anchors[:, 1] *= img_size[1]
    area = anchors[:, 0] * anchors[:, 1]
    
    output = 'anchors: '
    for i in range(n_clusters):
        index = np.argmax(area)
        area[index] = 0
        output += '[%d,%d], ' % (int(anchors[index][0]), int(anchors[index][1]))
    print(output[:-2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', type=str)
    parser.add_argument('-n', type=int, default=9)
    parser.add_argument('-s', '--img-size', type=str, default='416')
    opt = parser.parse_args()
    print(opt)
    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0])] * 2
    else:
        img_size = [int(x) for x in img_size]
    kmeans_anchor(opt.coco, opt.n, img_size)
