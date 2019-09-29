import cv2
import numpy as np
import os


def get_orientation(img):
    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    # blur = cv2.blur(v, (3, 3))
    blur = v
    # cv2.imshow('blur', blur)
    norm = np.zeros_like(blur)
    cv2.normalize(blur, norm, 0, 255, cv2.NORM_MINMAX, -1)
    # cv2.imshow('norm', norm)
    _, thres = cv2.threshold(norm, 160, 255, cv2.THRESH_BINARY)
    _, contours, __ = cv2.findContours(
        thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # canny = cv2.Canny(norm, 50, 150)
    # cv2.imshow('canny', canny)
    max_area = 0
    main_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            main_contour = contour
    canvas = np.zeros_like(v)
    cv2.drawContours(canvas, [main_contour], 0, 255, -1)
    x_max = int(max(main_contour[:, :, 0]))
    x_min = int(min(main_contour[:, :, 0]))
    y_max = int(max(main_contour[:, :, 1]))
    y_min = int(min(main_contour[:, :, 1]))
    canvas = canvas[y_min:y_max, x_min:x_max]
    canvas = cv2.resize(canvas, (32, 32))
    moments = cv2.moments(main_contour)
    center = [moments['m10'] / moments['m00'], moments['m01'] / moments['m00']]
    y_orientation = center[1] / (y_max/2+y_min/2)
    # print(orientation)
    y_orientation = min(y_orientation, 1.1)
    y_orientation = max(y_orientation, 0.9)
    x_orientation = center[0] / (x_max/2+x_min/2)
    # print(orientation)
    x_orientation = min(x_orientation, 1.1)
    x_orientation = max(x_orientation, 0.9)
    return canvas, [y_orientation, x_orientation]


if __name__ == "__main__":
    path = 'D:/woods/Desktop/mark-small-split/sout/train/sr'
    names = os.listdir(path)
    for name in names:
        img = cv2.imread(os.path.join(path, name))
        canvas, orientation = get_orientation(img)
        print(orientation)
        cv2.imshow('img', canvas)
        cv2.waitKey(0)
