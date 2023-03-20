# Data Setup

## Training data setup:
1. Download Data from [link](). We combine and mix randomly sampled 300 2D slices obtained from each of the two sets: a) ___wet b) ___cured
2. Use `padpatch.py` script to perform padding and patching of all images according to our paper to output 60000 images.
3. Use `sample.py` to split the dataset into traininng and validation sets. Set the train, validation and/or test variable in the script.
4. Use `datasplit.py` to create subsets (50000:10000, 25000:5000, and 12500:2500) of the dataset according to train:validation splits generated in the previous step and save data to respective folders.
5. Update the configuration files with the paths of these datasets and their corresponding textfiles containing the image IDs.
* Follow the same process for setting up the groundtruth files.



## Inference data setup:

1. Download Data from [link](). We use 2D slices obtained from four sets containing 1000 images each: a) ___wet b) ___cured c) ___wet d) ___cured
2. Use `padpatch.py` script for each set to perform padding and patching of all images. It will result in 100000 testing images.
3. Write the filenames in textfile using `sample.py`
5. Update the configuration files with the paths of these test datasets and their corresponding textfiles containing the image IDs.
* Follow the same process for setting up the groundtruth files.


Refer to MMSegmentation's documentation about setting up data [here](https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#
).
