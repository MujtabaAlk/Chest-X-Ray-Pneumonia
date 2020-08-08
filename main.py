from pathlib import Path
from PIL import Image
import numpy as np
import pickle
import random
from tensorflow import keras


normal_test_input_directory = "images/chest_xray/test/NORMAL"
pneumonia_test_input_directory = "images/chest_xray/test/PNEUMONIA"
output_dataset_file_name = "./images/chest_xray/test/data.pickle"
image_extensions = [".jpeg", ".jpg", ".png"]


def rescale_load_images(input_dir_path: str) -> []:
    """
    This function takes in a path to a directory of images, it will then rescales the images to size (512, 512)
    , store them in a new directory and load the images into an array after being converted to numpy arrays
    :param input_dir_path: a string that represents the path to the directory containing the images
    :return: An array of numpy arrays representing the rescaled images
    """
    # change \ to / for better compatibility
    if "\\" in input_dir_path:
        input_dir_path = input_dir_path.replace("\\", "/")

    input_dir = Path(input_dir_path)

    # create output directory path
    output_dir = input_dir.parent / (input_dir.name + "_Scaled")

    # create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # store paths for all image files in input directory with file extensions in image_extensions
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(input_dir.glob("*"+extension))

    # array for storing image numpy arrays
    image_arrays = []

    # loop through images in directory
    for image_path in image_paths:
        # import image as Image object
        input_image = Image.open(image_path)
        # if image is not in grey scale mode convert to gray scale
        if not input_image.mode == "L":
            input_image = input_image.convert("L")
        # determine cropping points based on dimensions
        image_width = input_image.width
        image_height = input_image.height
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
        cropped_image = input_image.crop((left_edge, top_edge, right_edge, bottom_edge))
        # resize image
        output_image = cropped_image.resize((512, 512), Image.LANCZOS)
        # create new image name
        new_image_path = output_dir / image_path.name
        # save output image
        output_image.save(new_image_path)
        # convert image to numpy array and append to image_arrays
        image_arrays.append(np.array(output_image))

    print(str(output_dir))  # print where the rescaled images are stored for testing purposes
    return image_arrays


if __name__ == "__main__":

    # rescale and load testing data (images) into arrays
    normal_test_data = rescale_load_images(normal_test_input_directory)
    pneumonia_test_data = rescale_load_images(pneumonia_test_input_directory)

    # store whole testing dataset into one object
    testing_dataset = []
    for image in normal_test_data:
        testing_dataset.append([image, 0])
    for image in pneumonia_test_data:
        testing_dataset.append([image, 1])

    # shuffle testing dataset
    random.shuffle(testing_dataset)

    # store normal testing data
    pickle_out = open(normal_test_input_directory + ".pickle", "wb")
    pickle.dump(normal_test_data, pickle_out)
    pickle_out.close()
    # store pneumonia testing data
    pickle_out = open(pneumonia_test_input_directory + ".pickle", "wb")
    pickle.dump(pneumonia_test_data, pickle_out)
    pickle_out.close()
    # store full testing dataset
    pickle_out = open(output_dataset_file_name, "wb")
    pickle.dump(testing_dataset, pickle_out)
    pickle_out.close()
