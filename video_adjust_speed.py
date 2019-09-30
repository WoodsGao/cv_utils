import os
import cv2


def video2img(video_path, rate=0., fps=None):
    """视频加速/减速, 会在视频所在文件夹创建'speed_adjusted_'前缀的视频

    Arguments:
        video_path {str} -- 视频路径

    Keyword Arguments:
        rate {float} -- 跳帧比例(0.-1.), 越大越快 (default: {0.})
        fps {float} -- 调整fps来调节速度 (default: {None})
    """
    video_cap = cv2.VideoCapture(video_path)
    size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if fps is None:
        fps = video_cap.get(cv2.CAP_PROP_FPS)
    output_path = os.path.join(
        os.path.dirname(video_path),
        'speed_adjusted_' + os.path.basename(video_path))
    video_writer = cv2.VideoWriter(
        os.path.splitext(output_path) + '.avi',
        cv2.VideoWriter_fourcc(*'H264'), fps, size)
    fi = 0
    while video_cap.isOpened():
        fi += 1
        rect, frame = video_cap.read()
        if frame is None:
            break
        if fi % int(1 / (1 - rate)) != 0:
            continue
        video_writer.write(frame)
    video_writer.release()
    video_cap.release()


if __name__ == "__main__":
    video2img(
        '/home/uisee/Datasets/20190715/20190715_100027_backupdata/log/dump_images/mono1.avi',
        0.5, 60)
