#!/usr/bin/python3
import argparse
import os
import os.path as osp
import shutil

import cv2


def video2img(video_path, ratio=0.):
    """视频转图片, 会在视频所在文件夹创建同名路径

    Arguments:
        video_path {str} -- 视频路径

    Keyword Arguments:
        ratio {float} -- 跳帧比例(0.-1.) (default: {0.})
    """
    output_dir = osp.splitext(video_path)[0]
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    video_cap = cv2.VideoCapture(video_path)
    fi = 0
    while video_cap.isOpened():
        fi += 1
        rect, frame = video_cap.read()
        if frame is None:
            break
        if fi % int(1 / (1 - ratio)) > 0:
            continue
        cv2.imwrite(osp.join(output_dir, '%010d.png' % fi), frame)
    video_cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str)
    parser.add_argument('--ratio', type=float, default=0.)
    opt = parser.parse_args()
    video2img(opt.video, opt.ratio)
