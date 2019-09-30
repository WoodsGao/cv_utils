import random
import os
import shutil


def select_samples(src_dir, dst_dir, rate=0.01, shuffle=False):
    """抽取一部分文件

    Arguments:
        src_dir {str} -- 输入路径
        dst_dir {str} -- 输出路径

    Keyword Arguments:
        rate {float} -- [description] (default: {0.01})
        shuffle {bool} -- [description] (default: {False})
    """
    names = os.listdir(src_dir)
    names.sort()
    for ni, name in enumerate(names):
        if shuffle:
            if random.random() > rate:
                continue
        else:
            if ni % (1 / rate) > 0:
                continue
        shutil.copy(os.path.join(src_dir, name), os.path.join(dst_dir, name))


if __name__ == "__main__":
    select_samples(
        '/home/uisee/Datasets/20190715/20190715_100027_backupdata/log/dump_images/mono',
        '/home/uisee/Datasets/20190715/20190715_100027_backupdata/log/dump_images/mono1'
    )
