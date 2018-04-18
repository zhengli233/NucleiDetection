import cv2
import numpy as np


def split(height, width, rects, y, file_path):
    seq = 0
    for rect in rects:
        new_masks = np.zeros((height, width, 3))
        for i in range(rect[0], rect[0]+rect[2]):
            for j in range(rect[1], rect[1]+rect[3]):
                new_masks[j, i, :] = y[j, i, :]
        if seq==0:
            masks_path = file_path + str(seq) + '.png'
            cv2.imwrite(masks_path, y)
        seq = seq + 1
        masks_path = file_path + str(seq) + '.png'
        cv2.imwrite(masks_path, new_masks)