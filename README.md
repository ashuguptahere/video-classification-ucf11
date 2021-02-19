# Video Classification UCF11
Video Classification on UCF11 dataset


# Preprocessing

First, download the dataset from UCF Repository [https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php] then run the "preprocessing.ipynb" file to preprocess UCF11 dataset.

In the preprocessing phase I used a different technique which extracted exactly 1,650 frames per category
meaning 1,650 x 11 = 18,150 frames or you can say images in whole dataset

You can download the dataset from my kaggle profile:
	[https://www.kaggle.com/oetjunhbxu/video-classification-ucf11]

# About Dataset

UCF11 folder: (where original dataset is located)

~Aroung 150 videos in each category

~1,600 total videos in UCF101 folder

dataset folder: (where only one videos is there for each category after combining)

~only 1 video in each category

~11 total videos in dataset folder after combining all the videos

training folder: (where training_set is located)

~1500 frames in each category

~1500 x 11 = 16,500 total frames

testing folder: (where testing_set is located)

~150 frames in each category

~150 x 11 = 1,650 total frames


frames in training_set = 16,500

frames validation set = 20% of training set (3,300 frames)

frames in testing_set = 1,650 (10% of training set)


# Model Analysis:
models used:

	ResNet50
	
	ResNet101

	ResNet152
	
	ResNet50V2
	
	ResNet101V2

	ResNet152V2
	
	MobileNet
	
	MobileNetV2

MobileNet and MobileNetV2 are worst model to perform Video Classification because they aren't made for heavy datasets infact they are made for mobile and embedded devices, hence named "mobile". Also MobileNets are giving good accuracies but have higher losses, that's why we discarded this model

ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2 and ResNet152V2 all of them are giving much impressive results than their counterparts (MobileNets) but took much time for training because of the fact that it contains more deeper and hidden layers.


# Required Parameters

dataset = "UCF11_updated_mpg/"

dataset2 = "dataset/"

train_path = "training_set/"

test_path = "testing_set/"

no_of_frames = 1650

epochs = 20

batch_size = 32

n_classes = 11

optimizer = "Adam"

loss_metric = "categorical_crossentropy"

last_layer_activation_function = "softmax"


input shape of ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, MobileNet and MobileNetV2 are all the same and that is: (224, 224, 3) = [image height, image width and number of channels]
