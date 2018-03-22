import os
# from shutil import copy
import cv2
from contour import get_contours
import numpy as np

def get_data():
	data_dir = "./Data/stage1_train/"
	# count = 0
	X_train = []
	y_train = []
	for name in os.listdir(data_dir):
		img_dir = data_dir + name + "/images/"
		mask_dir = data_dir + name + '/masks/'
		img_file = img_dir + os.listdir(img_dir)[0]
		original_img = cv2.imread(img_file)
		# print(count)

		'''
		try:
			os.stat('./KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
		except:
			os.mkdir('./KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
		copy(img_file, './KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
		'''
		# mask_count = 0
		masks = os.listdir(mask_dir)
		for mask in masks:
			y_train.append(get_contours(mask_dir + mask))
			mask = cv2.imread(mask_dir + mask)
			masked_img = original_img & mask
			if masked_img.shape != (256, 256):
				masked_img = cv2.resize(masked_img, (256, 256))
			X_train.append(masked_img)
			# print(mask_count)
			# mask_count += 1

		# count += 1
		# if count == 10:
			# break	
	return X_train, y_train
	print('created X_train and y_train')
'''
X, y = get_data()
X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)
'''
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
