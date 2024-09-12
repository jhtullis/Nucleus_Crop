
# Blast Cropper #1: Henry Tullis
# main.py: User interface and scripting
# 6-28-24

# ---- Imports ----
# opencv:
#   For computer vision / image analysis
import cv2

# os.path:
#   Will allow us to work easily with filepaths / folders
from os import listdir, path
from os.path import isfile, join

# process_image:
#   process_image.py should be a file in the same folder as main.py
#   I will put the code in there that actually does the analysis
#   to keep this file from becoming too cluttered.
import process_image as myimg

# matplotlib:
#   Could be used to visualize images as part of the program,
#   may be used later, but not now
# from matplotlib import pyplot as plt - may be needed later

# ---- User Defined Variables ----

# filepaths:
#   Copy the filepath (with \\ as the delimiter for Windows) 
#   to your input files folder using the file explorer
#   and paste it inside the quotes for each directory.
indir = "my_input_images"
outdir_crop = "my_output_crop"
outdir_box = "my_output_box"

# create a list of all image file names in the input folder
#   *Right now, we are assuming that this folder only contains image files*
#   Help from: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
filepaths = [f for f in listdir(indir) if isfile(join(indir, f))]

# execution:
#   For every filepath, process the image.
#   Then save the results in the output directory.
for imgname in filepaths:
    # ---- processing the image ----
    # load in the image
    print("\n\nProcessing", join(indir,imgname))
    image = cv2.imread(join(indir,imgname))

    # print image info
    # print(myimg.print_data_info(image))

    # find the image mask - basically, the pixels that we think are stained
    print("...Masking")
    mask = myimg.default_mask(image)

    # find the rectangles that we want to crop out of the image
    print("...Identifying cells")
    rectangles, keep = myimg.identify_cell_box_v1(mask)

    # ---- saving the output images ----
    # get the image name without an extension so we can save it as we want
    short_name = path.splitext(imgname)[0]

    # crop the images and save in the output directory
    print("...Saving outputs")
    myimg.save_cropped_images(image, rectangles, keep, short_name, outdir_crop, image_extension=".JPG")

    # also color the rectangles on the main image and save it too
    myimg.save_rectangles_on_image(image, rectangles, keep, short_name, outdir_box, image_extension=".JPG")



