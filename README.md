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
train a object detection model using tensorflow
https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

Done work
create the dataset using the training data:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
basic idea:
use cv2 to find the bound of each cell
pass the image information and the bound to the creator, then the creator forms the dataset

Current work:
find out how to implement the R-CNN model