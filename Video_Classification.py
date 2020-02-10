# Importing Essential Libraries
import os
import cv2
import glob
import tensorflow
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Getting Labelled Data
names = os.listdir('dataset/')

# Extracting images from each video of every category
# for name in tqdm(names):
#     count = 0
#     a = glob.glob('dataset/'+name+'/*.mpg')
#     for i in range(len(a)):
#         cap = cv2.VideoCapture(a[i])
#         frameRate = cap.get(5)
#         while(cap.isOpened()):
#             frameId = cap.get(1)
#             ret, frame = cap.read()
#             if (ret != True):
#                 break
#             if (frameId % math.floor(frameRate) == 0):
#                 cv2.imwrite('dataset/'+name+'/'+'{}.jpg'.format(count), frame)
#                 count += 1
#         cap.release()

# Fetching images and Storing it into array
basketball = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'basketball/*.jpg'))]
biking = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'biking/*.jpg'))]
diving = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'diving/*.jpg'))]
golf_swing = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'golf_swing/*.jpg'))]
horse_riding = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'horse_riding/*.jpg'))]
soccer_juggling = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'soccer_juggling/*.jpg'))]
swing = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'swing/*.jpg'))]
tennis_swing = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'tennis_swing/*.jpg'))]
trampoline_jumping = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'trampoline_jumping/*.jpg'))]
volleyball_spiking = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'volleyball_spiking/*.jpg'))]
walking = [cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (320, 240)) for file in tqdm(glob.glob('dataset/'+'walking/*.jpg'))]

# Plotting Image
plt.imshow(volleyball_spiking[0])
plt.show()

# Reshaping the data
basketball = np.array(basketball).reshape(len(basketball), 76800)
biking = np.array(biking).reshape(len(biking), 76800)
diving = np.array(diving).reshape(len(diving), 76800)
golf_swing = np.array(golf_swing).reshape(len(golf_swing), 76800)
horse_riding = np.array(horse_riding).reshape(len(horse_riding), 76800)
soccer_juggling = np.array(soccer_juggling).reshape(len(soccer_juggling), 76800)
swing = np.array(swing).reshape(len(swing), 76800)
tennis_swing = np.array(tennis_swing).reshape(len(tennis_swing), 76800)
trampoline_jumping = np.array(trampoline_jumping).reshape(len(trampoline_jumping), 76800)
volleyball_spiking = np.array(volleyball_spiking).reshape(len(volleyball_spiking), 76800)
walking = np.array(walking).reshape(len(walking), 76800)

# Converting lists into dataframes
basketball = pd.DataFrame(basketball)
biking = pd.DataFrame(biking)
diving = pd.DataFrame(diving)
golf_swing = pd.DataFrame(golf_swing)
horse_riding = pd.DataFrame(horse_riding)
soccer_juggling = pd.DataFrame(soccer_juggling)
swing = pd.DataFrame(swing)
tennis_swing = pd.DataFrame(tennis_swing)
trampoline_jumping = pd.DataFrame(trampoline_jumping)
volleyball_spiking = pd.DataFrame(volleyball_spiking)
walking = pd.DataFrame(walking)

# Assigning Labels
basketball['label'] = np.zeros(len(basketball))
biking['label'] = np.ones(len(biking))
diving['label'] = 2 * np.ones(len(diving))
golf_swing['label'] = 3 * np.ones(len(golf_swing))
horse_riding['label'] = 4 * np.ones(len(horse_riding))
soccer_juggling['label'] = 5 * np.ones(len(soccer_juggling))
swing['label'] = 6 * np.ones(len(swing))
tennis_swing['label'] = 7 * np.ones(len(tennis_swing))
trampoline_jumping['label'] = 8 * np.ones(len(trampoline_jumping))
volleyball_spiking['label'] = 9 * np.ones(len(volleyball_spiking))
walking['label'] = 10 * np.ones(len(walking))

# Merging dataset
dataset = pd.concat((basketball, biking, diving, golf_swing, horse_riding, soccer_juggling, swing, tennis_swing, trampoline_jumping, volleyball_spiking, walking), axis = 0)

# Deleting the faltu variables 
del(basketball, biking, diving, golf_swing, horse_riding, soccer_juggling, swing, tennis_swing, trampoline_jumping, volleyball_spiking, walking)

# Dividing data into X and y
X = dataset.iloc[:, 0:11408].values
X = X / 255.0
y = dataset.iloc[:, -1].values

# Defining Model
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(256, activation = 'relu'))
model.add(tensorflow.keras.layers.Dense(128, activation = 'relu'))
model.add(tensorflow.keras.layers.Dense(11, activation = 'softmax'))

# Compiling the model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Saving the model history
history = model.fit(X, y, epochs = 50)

pd.DataFrame(history.history).plot()

model.save("UCF11.h5")

# Prediction
# model.predict()
# predicion khud kr lena namuno