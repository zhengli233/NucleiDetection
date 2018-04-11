'''
get the contour
'''
from __future__ import division

import cv2
import numpy as np

# directory = './Data/stage1_train/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf/masks/dd404644bdf6d796aee4823214deb736de9d0bacea79fb91ede87ff7ffdef57d.png'
def get_contours(directory, input_type):
	if input_type == 'directory':
		im = cv2.imread(directory)
		
	if input_type == 'image':
		im = directory

	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	thresh = cv2.convertScaleAbs(thresh)
	print(thresh.shape, thresh.dtype)
	image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	rects = []
	for contour in contours:
		rect = cv2.boundingRect(contour)
		rect = list(rect)
		rects.append(rect)
	return rects
# get_contours(directory, 'directory')