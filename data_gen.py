import os
from shutil import copy
import cv2

data_dir = "./Data/stage1_train/"
count = 0
for name in os.listdir(data_dir):
	img_dir = data_dir + name + "/images/"
	mask_dir = data_dir + name + '/masks/'
	img_file = img_dir + os.listdir(img_dir)[0]
	try:
		os.stat('./KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
	except:
		os.mkdir('./KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
	copy(img_file, './KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
	masks = os.listdir(mask_dir)
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
