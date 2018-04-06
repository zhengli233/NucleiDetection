from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras import backend as K
import cv2
import numpy as np

'''
class FixedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x = ((x/255.) - 0.5) * 2.
        return x
'''

img_width, img_height = 256, 256

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(256))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(4096))
model.add(Activation('relu'))

model.add(Reshape((64, 64, 1)))
model.add(Conv2DTranspose(64, (3, 3), data_format='channels_last'))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2DTranspose(1, (3, 3)))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

'''
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
'''
seed = 7


epochs = 10
data = np.load('combined.npz')
x_train = data['X_train']
y_train = data['y_train']

model.fit(x=x_train, y=y_train, batch_size=32, epochs=epochs)
'''
datagen = FixedImageDataGenerator(**data_gen_args)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
'''



model_yaml = model.to_yaml()
with open("combined_model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("combined_model.h5")
print("Saved model to disk")
