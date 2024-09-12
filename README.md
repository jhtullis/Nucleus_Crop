Code and documentation written by Jason Henry Tullis

Last updated 09-11-24

# Nucleus_Crop
## Crop images of cells with stained nuclei for applications in nuclear texture analysis

![IMG_2244r](https://github.com/user-attachments/assets/52a3ed76-a3ed-4baa-b059-0b3bacc09136)

## Outline and Explanation

The main concept of the code is to identify stained cells' nuclei. This is useful because the isolated image of the nucleus can in turn be analyzed. Different cell types use DNA differently and more or less of it can be condensed, resulting in different textures in the nuclei that provide an interesting source of data on the cell.

The cells are stained, so the nuclei have a characteristic dark purple color. The application the code is being used for currently is to then crop the images of the cells, producing output images highlighting each nucleus. I have provided two output files here, one of them with the zoomed-in images of the nuclei identified, and another providing an annotation of the original photo with boxes to show the workings of the algorithm.

In order to identify the nuclei, the pixels are first thresholded, creating a binary mask that gives a good idea of where the nuclei are. Ideally, a 1 represents a pixel composing part of the cell's nucleus, and a 0 represents a pixel in another part of the image, so the mask should look visually like a white sillohette of the nucleus. 

![next_steps](https://github.com/user-attachments/assets/1fb6cf26-e693-4714-b0e5-f51a80742043)

> A binary mask, silhouetting regions of interest in the color image above

Currently, a 1 represents a pixel which in the original image has a 'green' value greater than 210. There are better ways to create the binary mask I have considered, including using color spaces to increase the consistency and precision with which the nuclear coloration can be identified, and training a classifier using test images to identify the pixels that correspond to nuclei, so far I haven't had the time to implement these.

The binary mask generation process isn't perfect, so a single cell may result in many fragmented pieces of the cell (a broken silhouette). Depending on the image, the fragmentation can be pretty extreme.

I have found that this situation is not ideal, but preferrable to a weaker condition resulting in much of the image turning white (obscured and false silhouettes). The fragmented pieces of the nuclei can then be reassembled into more complete pieces through some post processing.

![Diseño sin título (3)](https://github.com/user-attachments/assets/dd65021e-6cda-4171-a75a-d8c066d05d8b)

> Currently, I am using agglomerative clustering (with `cullensun_cluster.py`) to group different pieces of the nucleus together and identify the larger area that the fragments correspond to. *First image:* Original photo, zoomed in to this cell. *Second image:* Binary mask *Third image:* Contiguous regions in the binary mask are boxed. Smaller regions can be noise or dark particles in the image and are excluded from the mask (shown in red), including in the full size original image, at the cost of missing cells that are broken up like this one. *Fourth image:* An appropriate agglomerative clustering algorithm can piece together these broken parts of the cell fairly reliably, at the cost of sometimes confusing two nearby cells for one.

Unfortunately, this method also sometimes leads to some nuclei being stuck together as one region of interest. In order to combat this, I have also tried using morphological image analysis techniques, including opening, which helps to erode some small white islands in the image. I found the agglomerative clustering on its own to be simpler and more foolproof at the moment, however, with a larger data set and greater priority, morphological image analysis may be worth experimenting with further to increase the speed, accuracy, and sensitivity of the nuclear detection.

I developed the workflow and much of the code for this application, using algorithms developed by other people when appropriate and helpful. The basic agglomerative clustering algorithm in cullensun_cluster.py was developed by Cullen Sun (https://cullensun.medium.com/agglomerative-clustering-for-opencv-contours-cd74719b678e), the floodfill algorithm in mallicksatya_floodfill.py (not currently in use) was also adapted from code by Satya Mallick (https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/).

## Contents
### `main.py`
Run the larger workflow to read in, crop, and save the photos

### `process_image.py`
Methods written by me that are useful for cropping, annotating and saving the images.

### `cullensun_cluster.py`
Written by Cullen Sun, this code provides the algorithm for agglomerative clustering that I use to combine nearby regions of interest in order to capture a complete cell (and at times, groups multiple nearby cells into one image).

### `mallicksatya_floodfill.py`
Originally written and explained by Satya Mallick and then modified slightly by me, this file contains a method for performing a floodfill in order to fill in holes. The current version of my code doesn't call this method, however, I've found the method extra useful in conjunction with the morphological image analysis operations dilation and erosion. I could utilize it in a future verison of the program.

### `my_input_images`
Three test images of cells with stained nuclei to demonstrate the operation of the program.

### `my_output_box`
The original test images with regions of interest selected by the program in green, and those identified but later rejected in red.

### `my_output_crop`
The output images of nuclei cropped by the program.

