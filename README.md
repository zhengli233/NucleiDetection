log

problems:
1. images are in different size:
	train: 
	{(512, 640): 13, 
	 (256, 256): 334, 
	 (603, 1272): 6, 
	 (1024, 1024): 16, 
	 (520, 696): 92, 
	 (256, 320): 112, 
	 (260, 347): 5, 
	 (360, 360): 91, 
	 (1040, 1388): 1} 
	total = 670
	test: 
	{(520, 696): 4, 
	 (519, 161): 2, 
	 (256, 256): 24, 
	 (524, 348): 4, 
	 (512, 640): 8, 
	 (260, 347): 4, 
	 (520, 348): 4, 
	 (512, 680): 8, 
	 (390, 239): 1, 
	 (519, 253): 4, 
	 (519, 162): 2} 
	total = 65
2. even if only use 256 * 256 sized images, we can't simply do linear model on them because features are much more than the samples.
3. 41/65 test data has no masks!

Solution:
1. Data pre-process:
	Apply mask on the original image to get the image with one nuclei, denoted as image_x, in shape of (num_images, width, height, channels). Could do it as (num_images, width * height * channels)
	The contour of the nuclei, which can be got from get_contour(), will be the label of image_x, in the shape of (num_images, [x, y, w, h]).
2. Build CNN
3. Try to use YOLO
	https://arxiv.org/pdf/1506.02640.pdf

The problem is that I haven't found out how to use the trained weight to implement YOLO
So I still need time to learn YOLO.
