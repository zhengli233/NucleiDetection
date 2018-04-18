from keras.models import model_from_yaml
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
from contour import get_contours

def filter(threshold):
	for i, row in enumerate(y):
		for j, chnl in enumerate(row):
			pixel = 0
			for k, channel in enumerate(chnl):
				pixel += channel
			if pixel > threshold * 3:
				y[i][j] = [1 , 1, 1]
			else:
				y[i][j] = [0, 0, 0]

file_path = 'combined_model'
yaml_file = open('/home/zhengli/ECE523/Project/model/' + file_path + '.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("/home/zhengli/ECE523/Project/model/" + file_path + ".h5")
print("Loaded model from" + file_path)
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
name = '0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732'
img_dir = './Data/stage1_test/' + name + '/images/' + name + '.png'
print(img_dir)
img = cv2.imread(img_dir)
height, width, channel = img.shape
print(img.shape)
fig, axe = plt.subplots(1)
plt.imshow(img)
resized_img = cv2.resize(img, (256, 256))
resized_img = cv2.normalize(resized_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

y = loaded_model.predict(np.expand_dims(resized_img, axis=0), steps=1)
y = cv2.resize(y[0], (width, height))
# y = np.reshape(y, (256, 256, 3))
filter(0.2)
print(y.shape)
y = cv2.normalize(y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
rects = get_contours(y, 'image')

for rect in rects:
	rectangle = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r',facecolor='none')
	axe.add_patch(rectangle)
fig, axe = plt.subplots(1)
plt.imshow(y)
for rect in rects:
	rectangle = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r',facecolor='none')
	axe.add_patch(rectangle)
plt.show()
