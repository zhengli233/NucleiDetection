import os
import cv2
from contour import get_contours
import numpy as np
from Preprocessing import preprocessing
from sklearn.model_selection import train_test_split

def get_X():
	data_dir = "./Data/stage1_train/"
	
	X_train = []
	y_train = []
	for name in os.listdir(data_dir):
		img_dir = data_dir + name + "/images/"
		img_file = img_dir + os.listdir(img_dir)[0]
		mask_dir = data_dir + name + '/masks/'
		original_img = cv2.imread(img_file)
		original_img = preprocessing(original_img)
		if original_img.shape != (256, 256, 3):
			original_img = cv2.resize(original_img, (256, 256))
		original_img = cv2.normalize(original_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		X_train.append(original_img)
		masks = os.listdir(mask_dir)
		first_read = True
		for mask in masks:
			mask = cv2.imread(mask_dir + mask)
			# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
			if mask.shape != (256, 256, 3):
				mask = cv2.resize(mask, (256, 256))
			if first_read:
				combined_mask = mask
				first_read = False
			else:
				combined_mask |= mask
		combined_mask = cv2.normalize(combined_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		# combined_mask = np.array(combined_mask)
		# combined_mask = np.reshape(combined_mask, (256, 256, 3))
		y_train.append(combined_mask)
			
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)	
	print('created data')
	np.savez('/home/zhengli/ECE523/Project/model/splited.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
	print('saved data!!!')
		

get_X()

