import numpy as np
import os
import cv2

def get_feature(args):
	if args == 'train':
		data_dir = "./Data/stage1_train/"
		X = [0 for x in range(334)]
	elif args == 'test':
		data_dir = './Data/stage1_test/'
		X = [0 for y in range(24)]
	else:
		return
	
	index = 0
	for name in os.listdir(data_dir):
			directory = data_dir + name + "/images/"
			directory = directory + os.listdir(directory)[0]
			image = cv2.imread(directory, 0).ravel()
			if len(image) == 256 * 256:
				X[index] = image
				index += 1
	return X

# print(X[0])

'''
directory = './Data/stage1_train/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf/images/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf.png'
image = cv2.imread(directory, 0)
image = image.ravel()
print(image)
'''