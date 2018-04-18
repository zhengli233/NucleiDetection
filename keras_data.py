from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras import backend as K
import cv2
import numpy as np

img_width, img_height = 256, 256

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

'''
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(4096))
model.add(Activation('relu'))

model.add(Reshape((64, 64, 1)))
'''

model.add(Conv2DTranspose(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))

model.add(Conv2DTranspose(3, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))



model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


seed = 7


epochs = 3
data = np.load('/home/zhengli/ECE523/Project/model/preprocessed.npz')
x_train = data['X_train']
y_train = data['y_train']

model.fit(x=x_train, y=y_train, batch_size=32, epochs=epochs)


save_path = 'preprocessed_model'

model_yaml = model.to_yaml()
with open("/home/zhengli/ECE523/Project/model/" + save_path + ".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

model.save_weights("/home/zhengli/ECE523/Project/model/" + save_path + ".h5")
print("Saved model to " + save_path)
