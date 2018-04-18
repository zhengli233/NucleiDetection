import numpy as np
import cv2


def preprocessing(original_img):
    m, n = original_img.shape
    num = np.zeros((256, 1))
    for i in range(m):
        for j in range(n):
            temp = original_img[i, j]
            num[temp, 0] = num[temp, 0] + 1
    if np.where(num == num.max())[0] > 50:
        original_img = 255 - original_img
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    return original_img
