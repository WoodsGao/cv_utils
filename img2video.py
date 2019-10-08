import os
import cv2
from .config import IMG_EXT
from tqdm import tqdm


def img2video(img_dir, fps, size):
    names = os.listdir(img_dir)
    names = [name for name in names if os.path.splitext(name)[1] in IMG_EXT]
    names.sort()
    video_writer = cv2.VideoWriter(img_dir + '.avi',
                                   cv2.VideoWriter_fourcc(*'H264'), fps, size)
    for name in tqdm(names):
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            continue
        if img.shape[0] != size[1] or img.shape[1] != size[0]:
            img = cv2.resize(img, size)
        video_writer.write(img)
    video_writer.release()


if __name__ == "__main__":
    img2video(
        '/home/uisee/Datasets/20190715/20190715_100027_backupdata/log/dump_images/mono',
        25, (1280, 720))
