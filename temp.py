from keras.preprocessing.image import ImageDataGenerator

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_datagen.fit(images, augment=True, seed=seed)
image_generator = image_datagen.flow_from_directory(
        './Data/stage1_train/516a0e20327d6dfcedcf57e3056115e4fb29cdf4cb349003bdfc75c9b7f5c2cf',
        class_mode=None,
        seed=seed)
print(image_generator)