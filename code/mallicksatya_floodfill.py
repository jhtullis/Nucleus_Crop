# floodfill for binarized b/w images
# adapted from code by Satya Mallick: https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

import cv2
import numpy as np

def floodfill_img(image):
    # # Read image
    # im_in = cv2.imread("bw_cells2.png", cv2.IMREAD_GRAYSCALE)
    
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    
    th, im_th = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)

    # th, im_th = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out
    
    # # Display images.
    # cv2.imshow("Thresholded Image", im_th)
    # cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv2.imshow("Foreground", im_out)
    # cv2.waitKey(0)