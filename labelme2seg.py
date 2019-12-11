import os
import cv2
import argparse
from tqdm import tqdm


def run(src_dir, dst_dir):
    classes = []
    img_dir = os.path.join(dst_dir, 'images')
    label_dir = os.path.join(dst_dir, 'labels')
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    names = os.listdir(src_dir)
    names = [n for n in names if os.path.splitext(n)[-1] == '.json']
    for name in names:
        name = os.path.join(src_dir, name)
        os.system('labelme_json_to_dataset %s && rm %s' % (name, name))
    names = os.listdir(src_dir)
    names = [n for n in names if os.path.isdir(os.path.join(src_dir, n))]
    for name in tqdm(names):
        os.system('mv %s %s' % (os.path.join(src_dir, name, 'label.png'),
                                os.path.join(label_dir, name[:-5] + '.png')))
        os.system('mv %s %s' % (os.path.join(src_dir, name, 'img.png'),
                                os.path.join(img_dir, name[:-5] + '.png')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='src file path')
    parser.add_argument('--dst', type=str, help='dst file path')
    opt = parser.parse_args()
    run(opt.src, opt.dst)
