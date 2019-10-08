import os
import cv2
from .utils import rebuild_dir


def video2img(video_path, rate=0.):
    """视频转图片, 会在视频所在文件夹创建同名路径

    Arguments:
        video_path {str} -- 视频路径

    Keyword Arguments:
        rate {float} -- 跳帧比例(0.-1.) (default: {0.})
    """
    output_dir = os.path.splitext(video_path)[0]
    rebuild_dir(output_dir)
    video_cap = cv2.VideoCapture(video_path)
    fi = 0
    while video_cap.isOpened():
        fi += 1
        rect, frame = video_cap.read()
        if frame is None:
            break
        if fi % int(1 / (1 - rate)) != 0:
            continue
        cv2.imwrite(os.path.join(output_dir, '%010d.png' % fi), frame)
    video_cap.release()


if __name__ == "__main__":
    video2img(
        '/home/uisee/Datasets/20190715/20190715_100027_backupdata/log/dump_images/mono1.avi'
    )
