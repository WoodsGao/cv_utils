import os
import os.path as osp
from utils import read_2d_list, write_2d_list


def run(label_dir):
    names = os.listdir(label_dir)
    for name in names:
        name = osp.join(label_dir, name)
        label = read_2d_list(name)
        new_label = []
        for l in label:
            # key points
            if l[0] == 0:
                l[3] = 0.025
                l[4] = 0.025
            new_label.append(l)
        write_2d_list(name, new_label)


if __name__ == "__main__":
    run('/home/uisee/Datasets/mark-small-2classes/labels')