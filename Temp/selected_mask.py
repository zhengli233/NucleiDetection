import os
import cv2
import random
from contour import get_contours
import numpy as np

def get_X():
	data_dir = "./Data/stage1_train/"
	size = 5
	X_train = []
	y_train = []
	for name in os.listdir(data_dir):
		img_dir = data_dir + name + "/images/"
		img_file = img_dir + os.listdir(img_dir)[0]
		mask_dir = data_dir + name + '/masks/'
		original_img = cv2.imread(img_file)
		
		if original_img.shape != (256, 256, 3):
			resized_img = cv2.resize(original_img, (256, 256))
		normalized_img = cv2.normalize(resized_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		X_train.append(normalized_img)
		masks = os.listdir(mask_dir)
		first_read = True
		for mask in masks:
			mask = cv2.imread(mask_dir + mask)
			mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
			if mask.shape != (256, 256):
				mask = cv2.resize(mask, (256, 256))
			if first_read:
				combined_mask = mask
				first_read = False
			else:
				combined_mask |= mask
		combined_mask = cv2.normalize(combined_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		combined_mask = np.array(combined_mask)
		combined_mask = np.reshape(combined_mask, (256, 256, 1))
		y_train.append(combined_mask)
		try:
			selected_masks = random.sample(masks, size)
		except ValueError:
			continue
		first_read = True
		for mask in selected_masks:
			mask = cv2.imread(mask_dir + mask)
			if first_read:
				combined_mask = mask
				first_read = False
			else:
				combined_mask |= mask
		original_img &= combined_mask
		if original_img.shape != (256, 256, 3):
			resized_img = cv2.resize(original_img, (256, 256))
		normalized_img = cv2.normalize(resized_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		X_train.append(normalized_img)
		if combined_mask.shape != (256, 256):
			combined_mask = cv2.resize(mask, (256, 256))
		combined_mask = cv2.cvtColor(combined_mask, cv2.COLOR_BGR2GRAY)
		combined_mask = cv2.normalize(combined_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		combined_mask = np.array(combined_mask)
		combined_mask = np.reshape(combined_mask, (256, 256, 1))
		y_train.append(combined_mask)

		
	print('created data')
	np.savez('/home/zhengli/ECE523/Project/model/selected.npz', X_train=X_train, y_train=y_train)
	print('saved data!!!')
		

get_X()

