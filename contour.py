'''
get the contour
'''
from __future__ import division

import cv2
import numpy as np

# directory = './Data/stage1_train/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf/masks/dd404644bdf6d796aee4823214deb736de9d0bacea79fb91ede87ff7ffdef57d.png'
def get_contours(directory):
	im = cv2.imread(directory)
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	rect = cv2.boundingRect(contours[0])
	rect = list(rect)
	for i in range(4):
		rect[i] = rect[i] / 256
	return rect


# print(get_contours(directory))
'''
cv2.rectangle(image, (x, y), (x + w, y + h), (128, 255, 0), 2)
cv2.imwrite('contour.png', image)


# print(contours)
cv2.drawContours(image, contours, -1, (128,255,0), 3)


# cv2.imwrite('origin.png', im)
'''
