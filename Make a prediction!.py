#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

CATEGORIES = ["Pedestrian", "No pedestrian"]  # will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 48  
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.


# In[2]:


model = keras.models.load_model('C:/Users/Jack/OneDrive - GMIT/Project Y4/PedDetection-CNN-48.model')


# In[3]:


prediction = model.predict([prepare('D:/Prediction images/NoPed.png')])
#print(prediction)  # will be a list in a list.
print(np.argmax(prediction))
print(CATEGORIES[int(prediction[0][0])]) #currently not working


# In[4]:


# plt.imshow([prepare('D:/Prediction images/Test.png')])
# plt.show()


# In[ ]:




