from sklearn.cluster import KMeans
import os
import numpy as np
from tqdm import tqdm
from .utils import read_2d_list


def kmeans_anchor(label_dir, n_clusters=9, img_height=320):
    names = os.listdir(label_dir)
    wh_list = []
    for name in tqdm(names):
        label = read_2d_list(os.path.join(label_dir, name))
        for line in label:
            if len(line) != 5:
                continue
            wh_list.append(line[3:])
    wh_list = np.float32(wh_list)
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(wh_list)
    anchors = np.int64(estimator.cluster_centers_ * img_height)
    output = 'anchors: '
    for anchor in anchors:
        output += '%d,%d, ' % (anchor[0], anchor[1])
    print(output[:-2])


if __name__ == "__main__":
    kmeans_anchor('/home/uisee/Datasets/mark-small-2classes/labels')
