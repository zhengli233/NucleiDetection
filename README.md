How to use:
1. Run combined_mask.py to pre-process the image and mask data. Remember to change the file path to your local path.
2. Run keras_data.py to train the cnn model. Remember to change the save path to your local path.
3. Run predict.py to check the model. Again, change the file paths to your local path.

Potential improvement:
1. Separate the output to several single-nuclei maskes.
2. Mini rectangles sometimes occur in regular ones. 
3. The model has difficulty in detecting nuclei in purple-cell-and-white-background images.