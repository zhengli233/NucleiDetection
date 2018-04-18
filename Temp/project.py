import numpy as np
import os
import cv2
from sklearn import svm
from label import get_label
from feature import get_feature

X_train = np.asarray(get_feature('train'))
y_train = np.asarray(get_label('train'))

X_test = get_feature('test')
y_test = get_label('test')

print(X_train.shape, len(y_train), len(X_test), len(y_test))

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


'''
X = np.array([[0 for x in range(256 * 256)]])
train_data_dir = "./Data/stage1_train/"
for name in os.listdir(train_data_dir):
		directory = train_data_dir + name + "/images/"
		directory = directory + os.listdir(directory)[0]
		image = cv2.imread(directory, 0)
		print(directory)
		print(len(image))
		

# print(X)

directory = './Data/stage1_train/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf/images/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf.png'

image = cv2.imread(directory)
cv2.imwrite('original.png', image)
# print(image)
print('------------------------------')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(image)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_kp.png', img)

print(len(get_feature()))
print(len(get_label()))
'''
