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


def crop_resize_images(input_dir_path: str) -> str:
    """
    This function takes in a path to a directory of images and rescales the images to size (512, 512) and stores them
    in a new directory and returns the path to the output directory
    :param input_dir_path: a string that represents the path to the directory containing the images
    :return: a string that represents the path to the directory containing the rescaled images
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

    # loop through images in directory
    for image_path in image_paths:
        # import image as Image object
        image = Image.open(image_path)
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
        new_image_path = output_dir / image_path.name
        # save output image
        output_image.save(new_image_path)

    return str(output_dir)


def load_images(dir_path: str) -> []:
    """
    This function will load the images in the specified directory into an array as numpy arrays
    :param dir_path:
    :return:
    """

    # change \ to / for better compatibility
    if "\\" in dir_path:
        dir_path = dir_path.replace("\\", "/")

    input_dir = Path(dir_path)

    # store paths for all image files in input directory with file extensions in image_extensions
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(input_dir.glob("*" + extension))

    images = []
    # loop through images in directory
    for image_path in image_paths:
        # load image
        image = Image.open(image_path)

        # if image is not in grey scale mode
        if not image.mode == "L":
            image = image.convert("L")

        # store in numpy array
        image_array = np.array(image)
        images.append(image_array)

    return images


if __name__ == "__main__":

    # Rescale normal test images
    normal_test_output_directory = crop_resize_images(normal_test_input_directory)
    print(normal_test_output_directory)
    # Rescale pneumonia test images
    pneumonia_test_output_directory = crop_resize_images(pneumonia_test_input_directory)
    print(pneumonia_test_output_directory)

    # load testing data into array
    normal_test_data = load_images(normal_test_output_directory)
    pneumonia_test_data = load_images(pneumonia_test_output_directory)

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
