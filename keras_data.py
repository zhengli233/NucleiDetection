from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
import cv2

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
'''
img = load_img('./KerasData/Image/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

image_datagen.fit(x, augment=True, seed=seed)
mask_datagen.fit(x, augment=True, seed=seed)
'''
image_generator = image_datagen.flow_from_directory(
    './KerasData/Image',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    './KerasData/Mask',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)


'''
model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
'''