import os
import random
from config import IMG_EXT


def generate_yolo_train_valid_txt(data_dir, train_rate=0.7, shuffle=True):
    """根据数据文件夹中的images生成yolov3所需要的train.txt和valid.txt
    
    Arguments:
        data_dir {str} -- 数据文件夹路径
    
    Keyword Arguments:
        train_rate {float} -- train所占比例 (default: {0.7})
        shuffle {bool} -- 是否打乱顺序 (default: {True})
    """
    img_dir = os.path.join(data_dir, 'images')
    names = os.listdir(img_dir)
    names = [name for name in names if os.path.splitext(name)[1] in IMG_EXT]
    names.sort()
    if shuffle:
        random.shuffle(names)
    names = [os.path.join(img_dir, name) for name in names]
    names = [os.path.abspath(name) for name in names]
    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(names[:int(train_rate * len(names))]))
    with open(os.path.join(data_dir, 'valid.txt'), 'w') as f:
        f.write('\n'.join(names[int(train_rate * len(names)):]))


if __name__ == "__main__":
    generate_yolo_train_valid_txt('/home/uisee/Datasets/mark-small-2classes')