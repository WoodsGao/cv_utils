import os
import cv2
from .utils import rebuild_dir
from .config import IMG_EXT
from .processors import CutImg
from tqdm import tqdm


def process_img(img_dir, output_dir, processors=[]):
    """图像批处理
    
    Arguments:
        img_dir {str} -- 输入文件夹
        output_dir {str} -- 输出文件夹
    
    Keyword Arguments:
        processors {list} -- processors中的实例 (default: {[]})
    
    Returns:
        int -- status
    """
    if not os.path.isdir(img_dir):
        return -1
    rebuild_dir(output_dir)
    names = os.listdir(img_dir)
    names = [name for name in names if os.path.splitext(name)[1] in IMG_EXT]
    for name in tqdm(names):
        img = cv2.imread(os.path.join(img_dir, name))
        if img is None:
            continue

        for p in processors:
            img = p(img)

        if img is None:
            continue
        cv2.imwrite(
            os.path.join(output_dir,
                         os.path.splitext(name)[0] + '.png'), img)
    return 0


if __name__ == "__main__":
    process_img(
        '/home/uisee/Datasets/20190715/20190715_100027_backupdata/log/dump_images/image_capturer_0',
        '/home/uisee/Datasets/20190715/20190715_100027_backupdata/log/dump_images/mono',
        processors=[CutImg([[0, 720], [0, 1280], [0, 3]])])
