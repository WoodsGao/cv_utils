import os
import cv2
import argparse
from tqdm import tqdm


def run(data_dir, dst_dir):
    classes = []
    os.makedirs(dst_dir, exist_ok=True)
    img_dir = os.path.join(data_dir, 'images')
    txt_dir = os.path.join(data_dir, 'labels')
    txt_names = os.listdir(txt_dir)
    txt_names = [os.path.splitext(name)[0] for name in txt_names]
    names = os.listdir(img_dir)
    names = [name for name in names if os.path.splitext(name)[0] in txt_names]
    for name in tqdm(names):
        basename = os.path.basename(name)
        basename = os.path.splitext(basename)[0]
        with open(os.path.join(txt_dir, basename + '.txt'), 'r') as f:
            bboxes = [[float(x) for x in l.split(' ')] for l in f.readlines()
                      if l]
        img = cv2.imread(os.path.join(img_dir, name))
        for i, (c, x, y, w, h) in enumerate(bboxes):
            if c not in classes:
                os.makedirs(os.path.join(dst_dir, str(c)), exist_ok=True)
            xmin = x - w
            xmax = x + w
            ymin = y - h
            ymax = y + h
            xmin *= img.shape[1]
            xmax *= img.shape[1]
            ymin *= img.shape[0]
            ymax *= img.shape[0]
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)
            cut = img[ymin:ymax, xmin:xmax]
            if cut.size < 1:
                continue
            cv2.imwrite(
                os.path.join(dst_dir, str(c), basename + '_%d.png' % i),
                cut)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='src path')
    parser.add_argument('--dst', type=str, help='dst file path')
    opt = parser.parse_args()
    run(opt.src, opt.dst)