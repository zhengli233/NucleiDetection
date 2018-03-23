import cv2
import os
from matplotlib import pyplot as plt
from contour import get_contours
import numpy as np
'''
mask_dir = './Data/stage1_train/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf/masks/'
masks = os.listdir(mask_dir)
first_read = True
for mask in masks:
	mask = cv2.imread(mask_dir + mask)
	if first_read:
		combined_mask = mask
		first_read = False
	else:
		combined_mask |= mask

img_dir = './Data/stage1_train/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf/images/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf.png'
img = cv2.imread(img_dir)
mask_sample = mask_dir + '2ee4841c082b8e3610af6f75b166aa9b8bff8ba3e257fccf66633a37162487bc.png'
mask_sample = cv2.imread(mask_sample)
print(combined_mask.shape, img.shape)
count_combined = 0
count_mask = 0
for i in range(360):
	for j in range(360):
		for k in range(3):
			if combined_mask[i][j][k] != 0:
				count_combined += 1
			if mask_sample[i][j][k] != 0:
				count_mask += 1

print(count_combined, count_mask)
cv2.imwrite('combined_mask.png', combined_mask)


print('./KerasData/Image/' + os.path.splitext(os.listdir(img_dir)[0])[0])
print(os.path.splitext("./Data/stage1_train/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf/images/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf.png")[0])
'''

img_dir = './Data/stage1_train/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf/images/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf.png'
img = cv2.imread(img_dir)
mask_dir = './Data/stage1_train/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf/masks/dd404644bdf6d796aee4823214deb736de9d0bacea79fb91ede87ff7ffdef57d.png'
mask = cv2.imread(mask_dir)
'''
single = img & mask
print(single.shape)
plt.imshow(single)
plt.imshow(img)
plt.show()
'''
npz = []
npz.append(mask)
print(mask.shape)
np.savez('mask.npz', npz=npz)
file = np.load('mask.npz')
mask = file['npz']
print(mask.shape)
