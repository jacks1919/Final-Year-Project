#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! pip uninstall -y tensorboard tb-nightly
# ! pip install -U tb-nightly


# In[2]:


# ! pip install tensorflow --upgrade


# In[3]:


# ! pip install matplotlib


# In[4]:


# ! pip install opencv-python


# In[5]:


# Imports needed
import os, datetime
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# from google.colab import files

img_height = 48
img_width = 48
input_shape = (48, 48, 1)

NAME = "PedDetection-cnn(28)-relu-drop(0.2)-cnn(16)-drop(0.2)-dense(28)-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model = keras.Sequential(
    [
        layers.Conv2D(28, (3,3), padding="same", input_shape=input_shape), # input layer
        layers.MaxPooling2D(),        # downsamples by taking the max value from each window
        layers.Dropout(0.2),          # ignores a random 20% of neurons during training to avoid overfitting
        layers.Conv2D(16, (3,3), padding="same"),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),          # ignores random neurons
        layers.Dense(28, 'relu'),
        layers.Flatten(),             # Flattening the 2D arrays for fully connected layers
        layers.Dense(10, 'softmax'),  # gives multiple values that sum to 1 eg. 0.0, 0.1, 0.9
    ]
)


train = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/DATASET/TRAIN/",
    labels="inferred", # takes names from folders
    label_mode="int",  # categorical, binary
    color_mode="grayscale",
    batch_size=8,
    image_size=(img_height, img_width),  # reshape if not in this size
    validation_split=0.05,
    subset="training",
    seed=1, # omitted seed as model is only trained once
)

validate = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/DATASET/TRAIN/",
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=8,
    image_size=(img_height, img_width),
    validation_split=0.05,
    subset="validation",
    seed=1,
)


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
    metrics=["accuracy"],
)
model.fit(train, epochs=3, callbacks=[tensorboard]) #, validation_split=0.05


# In[6]:


model.evaluate(validate)


# In[7]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[8]:


prediction = model.predict(validate)


# In[17]:


import numpy as np

CATEGORIES = ["Pedestrian", "No pedestrian"]  # will use this to convert prediction num to string value

print(np.argmax(prediction[0]))

print(CATEGORIES[int(prediction[0][0])])


# In[10]:


model.save('C:/Users/Jack/OneDrive - GMIT/Project Y4/PedDetection-CNN-48.model')


# In[ ]:




