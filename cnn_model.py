import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.utils import np_utils

np.random.seed(1234)

# load data from MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28).astype('float32')
X_test = X_test.reshape(-1, 28, 28).astype('float32')

# normalization 
X_train = X_train / 255
X_test = X_test / 255

# one-hot outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def cnn_model():
    # CNN sequential model
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=(28,28,1)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=100)
    performance = model.evaluate(X_test, y_test, verbose=0)
    return model

model = cnn_model()

# save model as JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# save weights as HDF5
model.save_weights("model.h5")
