# Blast Cropper #1: Henry Tullis
# analyze_image.py: Crop individual images
# 6-28-24
import numpy as np
import cv2
from cullensun_cluster import agglomerative_cluster
from os import path
from os.path import isfile, join

def default_mask(image):
    """Returns the mask of an image to identify the cell nuclei.
    The best foolproof method I have found is to restrict green > 210"""

    # Set the range of colors that we will interpret as "stained nuclei"
    #   feel free to experiment with these values
    #   another thing I want to try is using HSV conversion.
    #   any pixel outside of this range will be excluded.
    #   However, there is simply not much green in the nuclei compared 
    #   to everything else, so filtering out parts with very much green
    # is very helpful
    # lower_bound_stain = np.array([0, 210, 0])

    # colors in cv2 follow the Blue, Green, Red order

    lower_bound_stain = np.array([0, 0, 0])
    #                             ^  ^  ^
    #                        min  B  G  R

    upper_bound_stain = np.array([255, 45, 255])
    #                             ^    ^    ^
    #                         max B    G    R:

    # compute and return the mask

    # first find the values in the image that are within this range and make them all
    #   255, and turn everything else to 0.
    image = cv2.inRange(image, lower_bound_stain, upper_bound_stain)

    # Then turn the image from greyscale to binary
    #   It is stored as a greyscale image but we need it in binary instead
    #   take anything greater than 127 and less than 255 as a 1
    #   we only to the thing in position 1 of the list that is returned by 
    #   cv2.threshold.
    return cv2.threshold(image, 127, 255, cv2.IMREAD_GRAYSCALE)[1]

def print_data_info(image, image_name = None, stats=True):
    """Prints info about the numpy array holding the image information,
    including the formatting, etc"""
    print("\nImage Name:\t", str(image_name))
    shape = image.shape
    print("|--x-y Shape:\t", image.shape[:2])
    # get number of channels
    channels = 1
    if len(shape) > 2:
        channels = image.shape[2]
    print("|--Channels:\t", channels)
    print("|--Type:\t", image.dtype)
    if stats:
        print("|--Stats")
        # s = scistat.describe(image, axis = None)
        print("   |--Min:\t", np.amin(image))
        print("   |--Max:\t", np.amax(image))
        print("   |--Mean:\t", np.mean(image))
    # print("|--Flags:", "\n" + str(image.flags))

def identify_cell_box_v1(image, join_dist=100, keep_size=260):
    """Find cells in the image (naive)

    Returns the x and y coordinates for the boxes around each detected
    cell and returns these as a n x 2 numpy array, where n is the number 
    of cells. Input is the mask of the image with 1s representing areas
    with the cell stain.
    
    This method is a little naiive. I hope to make a faster and better
    way to pick out the cells, separate them, and clear out the background.
    """

    # Find the boundaries around all of the white areas in the image
    #   Later I will try to process the image a little first so that 
    #   there's not a bunch of random broken parts and super complicated
    #   boundaries
    contours, heirarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Because there are all of these nasty contours, throw out the ones
    #   that are especially small / zero-ish area (I think)
    #   good_contours are the ones that are big enough to keep
    good_contours = list()

    #   ... check each contour c in the list of contours
    for c in contours:
        M = cv2.moments(c)
        if not (int(M['m10']) == 0 or int(M['m01']) == 0 or int(M['m00']) == 0):
            # add it if it is big enough
            good_contours.append(c)

    # Stick the contours together that are dist pixels apart or less (dist defaults to 100)
    #   This uses the cullensun_cluster.py agglomerative_cluster method
    better_contours = agglomerative_cluster(good_contours, threshold_distance=join_dist)

    # Find the location and size of the rectangle that fits around each contour
    #   store these in a list of lists, where each inner list is a rectangle
    rectangles = list()
    for c in better_contours:
        x1, y1, w1, h1 = cv2.boundingRect(c)
        rectangles.append([x1,y1,w1,h1])
    
    # I think the code below might be redundant
    # sort these based on how big the area of each rectangle is
    #   area of each rectangle
    areas = [rect[2] * rect[3] for rect in rectangles]
    # print(areas)

    # If either dimension is less than 260 px identify that rectangle as a dud
    #   That way platlets, cells cut off by the photo, and smears don't create
    #   as many problems, however, the rectangle can be displayed still to 
    #   make sure nothing important is thrown out
    keep = list()
    for rectangle in rectangles:
        # whether or not to keep the box
        keepit = True
        # see if either of the dimensions is smaller than the tolerance
        if rectangle[2] < keep_size or rectangle[3] < keep_size:
            keepit = False
        keep.append(keepit)

    # return this data
    return rectangles, keep

def save_cropped_images(image, rectangles, keep, imgname, outdir, image_extension=".JPG"):
    """Save a cropped version of the image for each of the rectangles if it's keep"""
    for i, mypair in enumerate(zip(keep, rectangles)):
        # unpack the keep data and the rectangle
        keep, r = mypair

        # if the image made the cut, write the cropped one
        if keep:
            # get the position and dimensions of the rectangle
            x1, y1, w1, h1 = r

            # crop the image using this rectangle
            cropped = image[y1:y1+h1, x1:x1+w1, :]

            # save the image as desired
            #   image name
            imgpath = join(outdir, imgname + "[{}]".format(i) + image_extension)
            #   write the file
            cv2.imwrite(imgpath, cropped)

def save_rectangles_on_image(image, rectangles, keep, imgname, outdir, image_extension=".JPG"):
    myimg = image.copy()
    """Save a copy of the image with the rectangles drawn on it"""
    # for each rectangle
    for keep, r in zip(keep, rectangles):
        # get the location and dimensions
        x1, y1, w1, h1 = r
        # draw it on image
        #   color (0, 255, 255) is yellow
        if keep:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        #   thickness in pixles
        thickness = 20
        cv2.rectangle(myimg,(x1,y1),(x1+w1,y1+h1),color,thickness)
    
    # save image
    #   image name
    imgpath = join(outdir, imgname + "r" + image_extension)
    #   write the file (we can just write over thie imag)
    cv2.imwrite(imgpath, myimg)
