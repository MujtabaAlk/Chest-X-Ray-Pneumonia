from PIL import Image
import os
import errno


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


input_directory_name = "images/chest_xray/test/NORMAL"

if __name__ == "__main__":

    output_directory = crop_resize_images(input_directory_name)
    print(output_directory)
