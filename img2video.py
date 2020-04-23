import os
import os.path as osp
import cv2
from tqdm import tqdm
import argparse


def img2video(img_dir, fps, img_size):
    names = os.listdir(img_dir)
    names = [name for name in names if osp.splitext(name)[1] in ['.png', '.tiff', '.jpg', '.jpeg']]
    names.sort()
    video_writer = cv2.VideoWriter(img_dir + '.avi',
                                   cv2.VideoWriter_fourcc(*'H264'), fps, img_size)
    for name in tqdm(names):
        img = cv2.imread(osp.join(img_dir, name))
        if img is None:
            continue
        if img.shape[0] != img_size[1] or img.shape[1] != img_size[0]:
            img = cv2.resize(img, img_size)
        video_writer.write(img)
    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--img-size', type=str)
    opt = parser.parse_args()
    img_size = opt.img_size.split(',')
    assert len(img_size) in [1, 2]
    if len(img_size) == 1:
        img_size = [int(img_size[0])] * 2
    else:
        img_size = [int(x) for x in img_size]
    img2video(opt.img_dir, opt.fps, img_size)
