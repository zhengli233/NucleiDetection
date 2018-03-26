from keras.models import model_from_yaml
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

yaml_file = open('combined_model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("combined_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
name = 'd3ce382f190ee24729bd2e80684c11bef72bc9c733cdbbc19a17d2c1b2e775f7'
img_dir = './Data/stage1_train/' + name + '/images/' + name + '.png'
img = cv2.imread(img_dir)
fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(img)
cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

y = loaded_model.predict(np.expand_dims(img, axis=0), steps=1)
cv2.normalize(y[0], y[0], alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
fig.add_subplot(1, 2, 2)
plt.imshow(y[0])
plt.show()