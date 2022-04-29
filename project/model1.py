#%%
import random
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import *
# from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2

#%%
TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'
IMG_WIDTH = 50
IMG_HEIGHT = 50
# plt.figure(figsize=(50, 50))

#%%
TRAIN_0 = TRAIN_PATH + '/four'
# for i in range(5):
#     file = random.choice(os.listdir(TRAIN_0))
#     image_path= os.path.join(TRAIN_0, file)
#     img=mpimg.imread(image_path)
#     ax=plt.subplot(1,5,i+1)
#     ax.title.set_text(file)
#     plt.imshow(img)

#%%
# read the image and convert into numpy array of float 32
def create_dataset(image_dir):
    image_array = []
    class_name = []

    for dir in os.listdir(image_dir):
        if dir == '.DS_Store':
            continue
        for file in os.listdir(os.path.join(image_dir, dir)):
            image_path = os.path.join(image_dir, dir, file)
            image = mpimg.imread(image_path)
            image=np.array(image)
            image = image.astype(np.float32)
            image = np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))
            image = np.array(image)/255 # normalize the image
            # image = np.expand_dims(image, axis=2)
            image_array.append(image)
            class_name.append(dir)
    return image_array, class_name

X_train, y_train = create_dataset(TRAIN_PATH)
X_test, y_test = create_dataset(TEST_PATH)

#%%
target_dict = {k: v for v, k in enumerate(np.unique(y_train))}
print(target_dict)
target_val=  [target_dict[y_train[i]] for i in range(len(y_train))]

val_dict = {k: v for v, k in enumerate(np.unique(y_test))}
val_val = [val_dict[y_test[i]] for i in range(len(y_test))]

#%%
# model = Sequential()
# model.add(InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
# model.add(Conv2D(32, (3, 3), activation = 'relu'))
# model.add(Conv2D(128, (3, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))
# model.add(Dropout(0.25))

# # model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='softmax'))
model=Sequential()
model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
model.summary()
# %%
# train_data = image.ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)
# test_data = image.ImageDataGenerator(rescale=1./255)

# #%%
# train_gen = train_data.flow_from_directory(
#     TRAIN_PATH,
#     target_size = (IMG_WIDTH, IMG_HEIGHT),
#     batch_size = 32,
#     class_mode = 'categorical'
# )
# val_gen = test_data.flow_from_directory(
#     TEST_PATH,
#     target_size = (IMG_WIDTH, IMG_HEIGHT),
#     batch_size = 32,
#     class_mode = 'categorical'    
# )
# %%
history = model.fit(
    x=tf.cast(np.array(X_train), tf.float64),
    y=tf.cast(list(map(int,target_val)),tf.int32),
    epochs=10,
    batch_size=64,
    verbose=1,
    validation_data=(tf.cast(np.array(X_test), tf.float64), tf.cast(list(map(int,val_val)),tf.int32))
)
# history = model.fit(
#     train_gen,
#     steps_per_epoch=train_gen.samples//train_gen.batch_size,
#     epochs = 30,
#     batch_size = 32,
#     validation_data = val_gen,
#     validation_steps=val_gen.samples//val_gen.batch_size,
#     verbose = 1,
# )

model.save('model.h5')