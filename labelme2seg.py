import os
import os.path as osp
import cv2
import shutil
import argparse
from tqdm import tqdm


def labelme2seg(src_dir, dst_dir):
    classes = []
    img_dir = osp.join(dst_dir, 'images')
    label_dir = osp.join(dst_dir, 'labels')
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    names = os.listdir(src_dir)
    names = [n for n in names if osp.splitext(n)[-1] == '.json']
    for name in names:
        name = osp.join(src_dir, name)
        os.system('labelme_json_to_dataset %s && rm %s' % (name, name))
    names = os.listdir(src_dir)
    names = [n for n in names if osp.isdir(osp.join(src_dir, n))]
    for name in tqdm(names):
        shutil.move(osp.join(src_dir, name, 'label.png'),
                                osp.join(label_dir, name[:-5] + '.png'))))
        shutil.move(osp.join(src_dir, name, 'img.png'),
                                osp.join(img_dir, name[:-5] + '.png'))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='src file path')
    parser.add_argument('--dst', type=str, help='dst file path')
    opt = parser.parse_args()
    labelme2seg(opt.src, opt.dst)
