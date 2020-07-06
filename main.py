from PIL import Image
import os
import errno

directory = "images/chest_xray/test/NORMAL"
outputDirectory = "images/chest_xray/test/NORMAL_Cropped"

# add forward slash if it does not exist
if not directory.endswith("/"):
    directory = directory + "/"

if not outputDirectory.endswith("/"):
    outputDirectory = outputDirectory + "/"

# create output directory if it does not exist
if not os.path.exists(outputDirectory):
    try:
        os.makedirs(outputDirectory)
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# command line print arrays/lists
ratios = []
differenceList = []
heights = []
widths = []

for filename in os.listdir(directory):
    if filename.endswith(".jpeg"):
        image = Image.open(directory + filename)
        print(filename + ":\t", image.height, image.width)

        # determine cropping points based on dimensions
        width = image.width
        height = image.height
        if width > height:
            difference = width - height
            left = difference / 2
            top = 0
            right = width - (difference / 2)
            bottom = height
        elif height > width:
            difference = height - width
            left = 0
            top = difference / 2
            right = width
            bottom = height - (difference / 2)
        else:
            left = 0
            top = 0
            right = width
            bottom = height

        # crop image and save to output directory
        croppedImage = image.crop((left, top, right, bottom))
        newName = outputDirectory + filename
        print(newName + ":\t", croppedImage.height, croppedImage.width, "\n")
        # croppedImage.save(newName)

        # add image information to output arrays/lists
        heights.append(croppedImage.height)
        widths.append(croppedImage.width)
        ratio = croppedImage.height / croppedImage.width
        ratios.append(ratio)
        differenceList.append(croppedImage.height - croppedImage.width)
    else:
        continue

print("Heights:\nMax: ", max(heights), "Min: ", min(heights))
print("Widths:\nMax: ", max(widths), "Min: ", min(widths))
print("Ratios:\nMax: ", max(ratios), "Min: ", min(ratios))
print("difference:\nMax: ", max(differenceList), "Min: ", min(differenceList))
