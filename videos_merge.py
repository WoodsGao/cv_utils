import os
import cv2
from .config import VIDEO_EXT


def videos_merge(video_dir, fps, size):
    """按照文件名拼接视频, 生成在上级文件夹中

    Arguments:
        video_dir {str} -- 视频文件夹
        fps {float} -- 帧率
        size {tuple} -- (width, height)
    """
    names = os.listdir(video_dir)
    names = [name for name in names if os.path.splitext(name)[1] in VIDEO_EXT]
    names.sort()
    video_writer = cv2.VideoWriter(
        os.path.normpath(video_dir) + '.avi', cv2.VideoWriter_fourcc(*'H264'),
        float(fps), size)
    for name in names:
        video_path = os.path.join(video_dir, name)
        video_cap = cv2.VideoCapture(video_path)
        while video_cap.isOpened():
            rect, frame = video_cap.read()
            if frame is None:
                break
            video_writer.write(frame)
        video_cap.release()
    video_writer.release()


if __name__ == "__main__":
    videos_merge(
        '/home/uisee/Datasets/20190715/20190715_100027_backupdata/log/dump_images/',
        25, (1280, 720))
