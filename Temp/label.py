import argparse
import imutils
import cv2
import numpy as np
import os
import sys

def get_label(args):
	if args == 'train':
		data_dir = "./Data/stage1_train/"
		Y = [[0 for i in range(256 * 256)] for j in range(334)]
	elif args == 'test':
		data_dir = './Data/stage1_test/'
		Y = [[0 for i in range(256 * 256)] for j in range(24)]
	else:
		return
	
	index = 0

	for name in os.listdir(data_dir):
		directory = data_dir + name + "/images/"
		directory = directory + os.listdir(directory)[0]
		image = cv2.imread(directory, 0)
		if len(image.ravel()) == 256 * 256:
			directory = data_dir + name + "/masks/"
			y_mask = [0 for x in range(256 * 256)]
			try:
				os.stat(directory)
			except:
				continue
			for mask in os.listdir(directory):
				directory = directory + mask
				mask_image = cv2.imread(directory, 0)
			
			# print(directory)
			# print(len(image))
				cnts = cv2.findContours(mask_image, cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
				cnts = cnts[0] if imutils.is_cv2() else cnts[1]
				
				for c in cnts:
					# compute the center of the contour
					M = cv2.moments(c)
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])
					y_mask[cX + cY * 256] = 1
					# print(cX, cY)
			Y[index] = y_mask
			index = index + 1
	return Y
	# print(index)