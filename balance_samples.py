import numpy as np
import os
import shutil
from utils import *
from config import *


def balance_yolo_data(data_dir, max_rate=1000.):
    """统计yolo数据中各个分类的样本数量, 穷举出使得各分类出现次数方差最小的组合

    Arguments:
        img_dir {str} -- yolo数据路径

    Keyword Arguments:
        max_rate {float} -- 新生成的样本中种类的最大数量不能超过原样本中种类的最大数量的倍数 (default: {1000.})
    """
    img_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    names = os.listdir(img_dir)
    names = [name for name in names if os.path.splitext(name)[1] in IMG_EXT]
    # 各文件中各class的数量
    labels = np.zeros((len(names), 1000))
    # 各文件复制数量, 默认为1
    file_dist = np.ones(len(names))
    for ni, name in enumerate(names):
        with open(os.path.join(label_dir, os.path.splitext(name)[0]+'.txt'), 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                if line == '':
                    continue
                label_index = int(line.split(' ')[0])
                labels[ni, label_index] += 1
    # 去除数量为0的class
    labels = labels[:, np.where(np.sum(labels, axis=0) > 0)[0]]
    # 各class的数量
    first_dist = np.dot(file_dist, labels)
    # 各class最终数量的上限
    max_len = max_rate * np.max(first_dist)
    # class数量的方差
    last_score = np.var(first_dist)
    while True:
        # 该轮方差是否减小过
        flag = False
        for ni, name in enumerate(names):
            # 如果增加该文件数量, 是否能减小方差
            tmp_file_dist = file_dist.copy()
            tmp_file_dist[ni] += 1
            tmp_dist = np.dot(tmp_file_dist, labels)
            if np.max(tmp_dist) > max_len:
                print('beyond max length')
                continue
            score = np.var(tmp_dist)
            # 能减小,  更新file_dist
            if last_score > score:
                print('reduce var: {:10g}    var: {:10g}'.format(
                    last_score - score, score))
                file_dist[ni] += 1
                last_score = score
                flag = True
        if not flag:
            print('no reduction')
            break
    # final_dist = np.dot(file_dist, labels)
    for ni, name in enumerate(names):
        for ci in range(int(file_dist[ni] - 1)):
            shutil.copy(os.path.join(label_dir, name[:-3]+'txt'),
                        os.path.join(label_dir, str(ci)+'_'+name[:-3]+'txt'))
            shutil.copy(os.path.join(img_dir, name),
                        os.path.join(img_dir, str(ci)+'_'+name))


if __name__ == "__main__":
    balance_yolo_data('/home/uisee/Datasets/mark-small-2classes')
