import os
import cv2
import numpy as np

src = np.float32([[573, 429], [676, 429], [301, 720], [958, 720]])
dst = np.float32([[390, 462], [520, 462], [390, 832], [520, 832]])
p_matrix = cv2.getPerspectiveTransform(src, dst)
inv_p_matrix = cv2.getPerspectiveTransform(dst, src)


def project_points(points_array, M):
    """根据变换矩阵将点的坐标投影

    Arguments:
        points_array {np.float} -- (n, 2)N个点的xy坐标
        M {np.float} -- opencv透视变换矩阵

    Returns:
        np.float -- (n, 2)N个点的xy坐标
    """
    points = points_array.copy()
    points = np.concatenate([points, np.ones(len(points)).reshape(-1, 1)], 1)
    points = np.dot(M, points)
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], 1)
    return points


if __name__ == "__main__":
    src_dir = 'D:/20190715/mono'
    output_dir = 'D:/20190715/projected'
    names = os.listdir(src_dir)
    for name in names:
        img = cv2.imread(os.path.join(src_dir, name))
        img = cv2.warpPerspective(img, p_matrix, (832, 832))
        cv2.imwrite(os.path.join(output_dir, name), img)
        # cv2.imshow('img', img)
        # cv2.waitKey(10)
