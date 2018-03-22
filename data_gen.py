import os
# from shutil import copy
import cv2
from contour import get_contours
import numpy as np
import pickle

def get_X():
	data_dir = "./Data/stage1_train/"
	# count = 0
	X_train = []
	y_train = []
	for name in os.listdir(data_dir):
		img_dir = data_dir + name + "/images/"
		img_file = img_dir + os.listdir(img_dir)[0]
		mask_dir = data_dir + name + '/masks/'
		original_img = cv2.imread(img_file)
		# print(count)

		'''
		try:
			os.stat('./KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
		except:
			os.mkdir('./KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
		copy(img_file, './KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
		'''
		mask_count = 0
		if original_img.shape == (256, 256, 3):
			masks = os.listdir(mask_dir)
			for mask in masks:
				y_train.append(get_contours(mask_dir + mask))
				mask = cv2.imread(mask_dir + mask)
				masked_img = original_img & mask
				# if masked_img.shape != (256, 256):
					# masked_img = cv2.resize(masked_img, (256, 256))
				cv2.normalize(masked_img, masked_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
				X_train.append(masked_img)
				print(mask_count)
				mask_count += 1

			# count += 1
			# if count == 10:
				# break	
		
	print('created data')
	with open('X_train.txt', 'wb') as fp:
		pickle.dump(X_train, fp)
	print('saved X_train!!!')
	with open('y_train.txt', 'wb') as fp:
		pickle.dump(y_train, fp)
	print('saved y_train!!!')
	



def get_y():
	mask_dir = data_dir + name + '/masks/'
	masks = os.listdir(mask_dir)
	y_train = []
	for mask in masks:
		y_train.append(get_contours(mask_dir + mask))
	print('created y_train')
	with open('y_train.txt', 'wb') as fp:
		pickle.dump(y_train, fp)
	print('saved y_train!!!')
		

get_X()

'''
	first_read = True
	for mask in masks:
		mask = cv2.imread(mask_dir + mask)
		if first_read:
			combined_mask = mask
			first_read = False
		else:
			combined_mask |= mask
	try:
		os.stat('./KerasData/Mask/' + os.path.splitext(os.listdir(img_dir)[0])[0])
	except:
		os.mkdir('./KerasData/Mask/' + os.path.splitext(os.listdir(img_dir)[0])[0])
	cv2.imwrite('./KerasData/Mask/' + os.path.splitext(os.listdir(img_dir)[0])[0] + '/' + os.listdir(img_dir)[0], combined_mask)
	count += 1
	print(count)
'''
