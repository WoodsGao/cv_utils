import os
import cv2
import argparse
from tqdm import tqdm


def run(img_dir, txt_dir, dst_dir):
    classes = []
    os.makedirs(dst_dir, exist_ok=True)
    txt_names = os.listdir(txt_dir)
    txt_names = [os.path.splitext(name)[0] for name in txt_names]
    names = os.listdir(img_dir)
    names = [name for name in names if os.path.splitext(name)[0] in txt_names]
    for name in tqdm(names):
        basename = os.path.basename(name)
        basename = os.path.splitext(basename)[0]
        with open(os.path.join(txt_dir, basename + '.txt'), 'r') as f:
            bboxes = [[int(x) for x in l.split(' ')[:-2] if x] for l in f.read().split('\n')
                      if l]
        img = cv2.imread(os.path.join(img_dir, name))
        for i, (xmin, ymin, xmax, ymax, c) in enumerate(bboxes):
            if c not in classes:
                os.makedirs(os.path.join(dst_dir, str(c)), exist_ok=True)
            w = (xmax - xmin) // 3
            h = (ymax - ymin) // 3
            cut = img[ymin - h:ymax + h, xmin - w:xmax + w]
            if cut.size < 1:
                continue
            cv2.imwrite(
                os.path.join(dst_dir, str(c), basename + '_%d.png' % i),
                cut)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, help='img path')
    parser.add_argument('--txt-dir', type=str, help='txt path')
    parser.add_argument('--dst', type=str, help='dst file path')
    opt = parser.parse_args()
    run(opt.img_dir, opt.txt_dir, opt.dst)
