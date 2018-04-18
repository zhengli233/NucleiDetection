import numpy as np
import os
import cv2

train_data_dir = "./Data/stage1_test/"
static = {}
total = 0
for name in os.listdir(train_data_dir):

	directory = train_data_dir + name + "/masks/"
	try:
		os.stat(directory)
	except:
		total +=1
		print('no masks')
	'''
	directory = directory + os.listdir(directory)[0]
	image = np.asarray(cv2.imread(directory, 0))
	# print(directory)
	# print(len(image))
	
	if image.shape in static: 
		static[image.shape] += 1
	else:
		static[image.shape] = 1
	total = total + 1
	'''


print(static)
print(total)