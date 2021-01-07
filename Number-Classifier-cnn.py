# Imports needed
import os, datetime
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
%load_ext tensorboard


from google.colab import files

img_height = 28
img_width = 28
input_shape = (28, 28, 1)

NAME = "Number-Classifier-cnn(28)-drop(0.2)-cnn(16)-drop(0.2)-dense(28)-{}".format(int(time.time()))

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
    "/content/drive/My Drive/hand-written-digits/",
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
    "/content/drive/My Drive/hand-written-digits/",
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=8,
    image_size=(img_height, img_width),
    validation_split=0.05,
    subset="validation",
    seed=1,
)

print(train)

# train /= 255
# validate /= 255

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
    metrics=["accuracy"],
)
model.fit(train, epochs=7, callbacks=[tensorboard]) #, validation_split=0.05

model.evaluate(validate)

#model.evaluate(train, callbacks=[tensorboard])

#model.evaluate(validate, callbacks=[tensorboard])
#model.evaluate(validate)

%tensorboard --logdir logs

prediction = model.predict(validate)

import numpy as np
print(np.argmax(prediction[0]))

#plt.imshow(prediction[0])
#plt.show()

**Test with the full MNIST dataeset**

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() #tf.keras.datasets module provides datasets already-vectorized, in Numpy format

# comment out this to see the image below
# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
# Scaleing the data makes it easier for the model to learn.
x_train /= 255
x_test /= 255

model.evaluate(x_train, y_train)

predictions = model.predict([x_train])
print(np.argmax(predictions[3]))

# plt.imshow(x_train[3])
# plt.show()