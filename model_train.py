from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, Dropout

input_dataset_path = "./images/chest_xray/train/data.pickle"
image_size = 512

if __name__ == '__main__':
    # load dataset
    dataset = pickle.load(open(input_dataset_path, "rb"))

    # split dataset to X(features) and y(labels) arrays for training
    X = []
    y = []
    for feature, label in dataset:
        X.append(feature)
        y.append(label)

    # convert X into np array and reshape to appropriate shape
    X = np.array(X).reshape((-1, image_size, image_size, 1))
    # convert y into np array
    y = np.array(y)
    # normalize X values
    X = X/255.0
    # delete unneeded objects
    del dataset

    # debugging line: print data arrays
    print(X)
    print(X.shape)
    print(y)

    # create neural network (model)
    model = Sequential()
    # add first Conv layer
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # add second Conv layer
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # add Dense layer
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    # add output layer
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # compile model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    # train model using training data
    model.fit(X, y, batch_size=32,  epochs=10, validation_split=0.2)
    # save model
    model.save("./trained_model")
