import pickle
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.callbacks import TensorBoard

input_dataset_path = "./images/chest_xray/train/data.pickle"
image_size = 512
model_name = f"chest_xray_pneumonia-{int(time.time())}"

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
    # print(X)
    print(X.shape)
    # print(y)

    # create neural network (model)
    model = Sequential()
    # add first Conv layer
    model.add(Conv2D(32, (5, 5), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # add second Conv layer
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # add third Conv layer
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # add forth Conv layer
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # add fifth Conv layer
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # add sixth Conv layer
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # add Flatten layer to flatten data for Dense layers
    model.add(Flatten())
    # add first Dense layer
    model.add(Dense(128))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # add second Dense layer
    model.add(Dense(128))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # add third Dense layer
    model.add(Dense(64))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # add forth Dense layer
    model.add(Dense(64))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # add fifth Dense layer
    model.add(Dense(32))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # add sixth Dense layer
    model.add(Dense(32))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # add output layer
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # compile model
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    # create tensorboard object
    tensorboard = TensorBoard(log_dir=f"logs/{model_name}")
    # train model using training data
    model.fit(
        X, y,
        batch_size=32,
        epochs=10,
        validation_split=0.2,
        callbacks=[tensorboard],
    )
    # save model
    model.save("./trained_model")
