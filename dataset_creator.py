'''
Try to create a tf dataset for cells
'''

from __future__ import division
import tensorflow as tf
from object_detection.utils import dataset_util
import numpy as np
import cv2
import os
import imghdr
from contour import get_contours



flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def create_tf_example(example):
	# TODO(user): Populate the following variables from your example.
	img_directory = example + "/images/"
	img_directory += os.listdir(img_directory)[0]
	img = load_image(img_directory)
	height, width, channels = img.shape
	# print(img.shape)
	# height = None # Image height
	# width = None # Image width
	filename = img_directory # Filename of the image. Empty if image is not from file

	encoded_image_data = tf.compat.as_bytes(img.tostring()) # Encoded image bytes
	image_format =  imghdr.what(img_directory)# b'jpeg' or b'png'

	xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
	xmaxs = [] # List of normalized right x coordinates in bounding box
	             # (1 per box)
	ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
	ymaxs = [] # List of normalized bottom y coordinates in bounding box
	             # (1 per box)
	classes_text = [] # List of string class name of bounding box (1 per box)
	classes = [] # List of integer class id of bounding box (1 per box)

	masks_directory = example + "/masks/"
	masks = os.listdir(masks_directory)
	for mask in masks:
		mask_directory = masks_directory + mask
		x, y, w, h = get_contours(mask_directory)
		print(x)
		print(width)
		xmin = x / width
		xmax = (x + w) / width
		ymin = y / height
		ymax = (y + h) / height
		class_text = tf.compat.as_bytes(mask)
		classe = 1
		print(xmin)

		xmins.append(xmin)
		xmaxs.append(xmax)
		ymins.append(ymin)
		ymaxs.append(ymax)
		classes_text.append(class_text)
		classes.append(classe)

	tf_example = tf.train.Example(features=tf.train.Features(feature={
	    'image/height': dataset_util.int64_feature(height),
	    'image/width': dataset_util.int64_feature(width),
	    'image/filename': dataset_util.bytes_feature(filename),
	    'image/source_id': dataset_util.bytes_feature(filename),
	    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
	    'image/format': dataset_util.bytes_feature(image_format),
	    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
	    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
	    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
	    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
	    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
	    'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))
	return tf_example


# def main(_):
writer = tf.python_io.TFRecordWriter('./TFDataset/train.tfrecords')

  # TODO(user): Write code to read in your dataset to examples variable
examples = os.listdir('./Data/stage1_train')
for example in examples:
	tf_example = create_tf_example('./Data/stage1_train/' + example)
	writer.write(tf_example.SerializeToString())

writer.close()

'''
if __name__ == '__main__':
	tf.app.run()
'''