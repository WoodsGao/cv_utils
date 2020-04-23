import os
import cv2
import argparse


def draw_bin_seg(img, seg, dst):
    img = cv2.imread(img)[:, :1280]
    seg = cv2.imread(seg)
    seg = cv2.resize(seg, (img.shape[1], img.shape[0]))
    img[(seg > 0).any(2), :2] //= 2
    cv2.imwrite(dst, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str)
    parser.add_argument('seg', type=str)
    parser.add_argument('dst', type=str)
    opt = parser.parse_args()
    draw_bin_seg(opt.img, opt.seg, opt.dst)
