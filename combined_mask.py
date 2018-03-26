import os
# from shutil import copy
import cv2
from contour import get_contours
import numpy as np

def get_X():
	data_dir = "./Data/stage1_train/"
	
	X_train = []
	y_train = []
	for name in os.listdir(data_dir):
		img_dir = data_dir + name + "/images/"
		img_file = img_dir + os.listdir(img_dir)[0]
		mask_dir = data_dir + name + '/masks/'
		original_img = cv2.imread(img_file)
		
		'''
		try:
			os.stat('./KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
		except:
			os.mkdir('./KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
		copy(img_file, './KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
		'''
		# mask_count = 0
		if original_img.shape != (256, 256, 3):
			original_img = cv2.resize(original_img, (256, 256))
		cv2.normalize(original_img, original_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		X_train.append(original_img)
		masks = os.listdir(mask_dir)
		first_read = True
		for mask in masks:
			mask = cv2.imread(mask_dir + mask)
			if mask.shape != (256, 256, 3):
				mask = cv2.resize(mask, (256, 256))
			if first_read:
				combined_mask = mask
				first_read = False
			else:
				combined_mask |= mask
		cv2.normalize(combined_mask, combined_mask, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		y_train.append(combined_mask)
			
		
	print('created data')
	np.savez('combined.npz', X_train=X_train, y_train=y_train)
	print('saved data!!!')
	



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
	
	try:
		os.stat('./KerasData/Mask/' + os.path.splitext(os.listdir(img_dir)[0])[0])
	except:
		os.mkdir('./KerasData/Mask/' + os.path.splitext(os.listdir(img_dir)[0])[0])
	cv2.imwrite('./KerasData/Mask/' + os.path.splitext(os.listdir(img_dir)[0])[0] + '/' + os.listdir(img_dir)[0], combined_mask)
	count += 1
	print(count)
'''
