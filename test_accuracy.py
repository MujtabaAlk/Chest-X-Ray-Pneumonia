import pickle
import tensorflow as tf
import numpy as np

IMAGE_SIZE = 512

if __name__ == '__main__':
    # load testing data
    testing_data = pickle.load(open("./images/chest_xray/test/data.pickle", "rb"))
    # create X(features) and y(labels) arrays
    X = []
    y = []
    # load data to arrays
    for feature, label in testing_data:
        X.append(feature)
        y.append(label)

    # convert X to numpy and reshape to proper shape
    X = np.array(X).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))

    # load model
    model = tf.keras.models.load_model("./trained_model")

    # make predictions
    predictions = model.predict(X)

    # count accurate predictions
    count = 0
    count_Ones = 0
    for prediction, label in zip(predictions, y):
        if int(prediction[0]) == label:
            count += 1
        if int(prediction[0]) == 1:
            count_Ones += 1

    # print accuracy and arrays
    print(y)
    print(predictions)
    print("\n-----------------------------------\n")
    print(f"number of accurate predictions {count}")
    print(f"accuracy percentage = {(count / len(y)) * 100}")
    print(f"number of predictions of 1 = {count_Ones}")
