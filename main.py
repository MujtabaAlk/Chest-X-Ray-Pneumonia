import os
import errno
from pathlib import Path
from PIL import Image
import numpy as np
import pickle
import random
from tensorflow import keras


def crop_resize_images(input_dir_path: str) -> str:
    """
    This function takes in a path to a directory of images and rescales the images to size (512, 512) and stores them
    in a new directory and returns the path to the output directory
    :param input_dir_path: a string that represents the path to the directory containing the images
    :return: a string that represents the path to the directory containing the rescaled images
    """
    if "\\" in input_dir_path:
        input_dir_path = input_dir_path.replace("\\", "/")

    # add forward slash if it does not exist
    if not input_dir_path.endswith("/"):
        input_dir_path = input_dir_path + "/"

    # output directory name
    output_dir_path = input_dir_path[:-1] + "_Scaled/"

    # create output directory if it does not exist
    if not os.path.exists(output_dir_path):
        try:
            os.makedirs(output_dir_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # loop through files in directory
    for file_name in os.listdir(input_dir_path):
        # if file is an image
        if file_name.endswith(".jpeg"):
            image = Image.open(input_dir_path + file_name)

            # determine cropping points based on dimensions
            image_width = image.width
            image_height = image.height
            if image_width > image_height:
                dim_difference = image_width - image_height
                left_edge = dim_difference / 2
                top_edge = 0
                right_edge = image_width - (dim_difference / 2)
                bottom_edge = image_height
            elif image_height > image_width:
                dim_difference = image_height - image_width
                left_edge = 0
                top_edge = dim_difference / 2
                right_edge = image_width
                bottom_edge = image_height - (dim_difference / 2)
            else:
                left_edge = 0
                top_edge = 0
                right_edge = image_width
                bottom_edge = image_height

            # crop image and save to output directory
            cropped_image = image.crop((left_edge, top_edge, right_edge, bottom_edge))
            # resize image
            output_image = cropped_image.resize((512, 512), Image.LANCZOS)
            # create new image name
            new_image_name = output_dir_path + file_name
            # save output image
            output_image.save(new_image_name)
        
        else:
            continue

    return output_dir_path


def load_images(dir_path: str) -> []:
    """
    This function will load the images in the specified directory into an array as numpy arrays
    :param dir_path:
    :return:
    """

    if "\\" in dir_path:
        dir_path = dir_path.replace("\\", "/")

    # add forward slash if it does not exist
    if not dir_path.endswith("/"):
        dir_path = dir_path + "/"

    images = []

    # loop through files in directory
    for file_name in os.listdir(dir_path):
        # if file is an image
        if file_name.endswith(".jpeg"):
            # load image
            image = Image.open(dir_path + file_name)

            # if image is not in grey scale mode
            if not image.mode == "L":
                image = image.convert("L")

            # store in numpy array
            image_array = np.array(image)
            images.append(image_array)

    return images


normal_test_input_directory = "images/chest_xray/test/NORMAL"
pneumonia_test_input_directory = "images/chest_xray/test/PNEUMONIA"

if __name__ == "__main__":

    # # Rescale normal test images
    # normal_test_output_directory = crop_resize_images(normal_test_input_directory)
    # print(normal_test_output_directory)
    # # Rescale pneumonia test images
    # pneumonia_test_output_directory = crop_resize_images(pneumonia_test_input_directory)
    # print(pneumonia_test_output_directory)

    # load testing data into array
    normal_test_data = load_images("./images/chest_xray/test/NORMAL_Scaled")
    pneumonia_test_data = load_images("./images/chest_xray/test/PNEUMONIA_Scaled")

    # store whole testing dataset into one object
    testing_dataset = []
    for image in normal_test_data:
        testing_dataset.append([image, 0])
    for image in pneumonia_test_data:
        testing_dataset.append([image, 1])

    # shuffle testing dataset
    random.shuffle(testing_dataset)

    # store normal testing data
    pickle_out = open("./images/chest_xray/test/NORMAL.pickle", "wb")
    pickle.dump(normal_test_data, pickle_out)
    pickle_out.close()
    # store pneumonia testing data
    pickle_out = open("./images/chest_xray/test/PNEUMONIA.pickle", "wb")
    pickle.dump(pneumonia_test_data, pickle_out)
    pickle_out.close()
    # store full testing dataset
    pickle_out = open("./images/chest_xray/test/data.pickle", "wb")
    pickle.dump(testing_dataset, pickle_out)
    pickle_out.close()
