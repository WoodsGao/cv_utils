#!/usr/bin/python3
import argparse
import os
import os.path as osp
import time


def rename(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for i, f in enumerate(files):
            print(f)
            ext = osp.splitext(f)[-1]
            if ext in ['.png', '.jpg', '.tiff']:
                new_f = '%0.6lf' % time.time() + ext
                os.rename(osp.join(root, f), osp.join(root, new_f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    rename(args.path)
