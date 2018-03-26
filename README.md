log

problems:
Even though the training accuracy is high, the prediction of the trained model is meaningless.
Maybe need to use gray scale images as the label.

Solution:
1. Data pre-process:
	Apply mask on the original image to get the image with one nuclei, denoted as image_x, in shape of (num_images, width, height, channels). Could do it as (num_images, width * height * channels)
	The contour of the nuclei, which can be got from get_contour(), will be the label of image_x, in the shape of (num_images, [x, y, w, h]).
2. Build CNN
3. Try to use YOLO
	https://arxiv.org/pdf/1506.02640.pdf

The problem is that I haven't found out how to use the trained weight to implement YOLO
So I still need time to learn YOLO.

current work
recognize images with one nuclei
