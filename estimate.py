from keras.models import model_from_yaml
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
from contour import get_contours
from Preprocessing import preprocessing
from sklearn.model_selection import train_test_split


def filter(threshold):
    for i, row in enumerate(predict_y):
        for j, chnl in enumerate(row):
            pixel = 0
            for k, channel in enumerate(chnl):
                pixel += channel
            if pixel > threshold * 3:
                predict_y[i][j] = [1, 1, 1]
            else:
                predict_y[i][j] = [0, 0, 0]



file_path = 'preprocessed_model'
yaml_file = open('' + file_path + '.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("" + file_path + ".h5")
print("Loaded model from" + file_path)


# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
data = np.load('preprocessed.npz')
data_x = data['X_train']
data_y = data['y_train']
# train test data split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=0)
accuracy = []
for img in range(test_x.shape[0]):
    width, height, channel = test_x[img].shape
    predict_y = loaded_model.predict(np.expand_dims(test_x[img], axis=0), steps=1)
    predict_y = cv2.resize(predict_y[0], (width, height))
    filter(0.2)
    predict_y = cv2.normalize(predict_y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    predict_y = np.array(predict_y, dtype=int)
    test_temp = np.array(255 * test_y[img], dtype=int)
    union = predict_y | test_temp
    intersection = predict_y & test_temp
    accuracy_temp = sum(sum(sum(intersection)))/sum(sum(sum(union)))
    accuracy.append(accuracy_temp)

print('accuracy is :',accuracy)
accuracy_avg = sum(accuracy)/len(accuracy)
print('average accuracy is:',accuracy_avg )
#
