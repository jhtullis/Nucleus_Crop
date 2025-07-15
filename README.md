# Nucleus_Crop
 Crop images of cells with stained nuclei for applications in nuclear texture analysis

![IMG_2244r](https://github.com/user-attachments/assets/52a3ed76-a3ed-4baa-b059-0b3bacc09136)

> This GitHub repo and [Blast_Classification](https://github.com/jhtullis/Blast_Classification) accompany "Distinguishing Reactive Lymphocytes from Blasts Using Fractal Chromatin Patterns" by R. Cordner et al, under review by the *International Journal of Laboratory Hematology* as of May 2025.


## Introduction and Motivations

Blasts and reactive lymphocytes (RLs) are different classes of white blood cells that are hard for even expert pathologists to differentiate under the microscope. They signal very different clinical outcomes - high numbers of blasts are indicative of different types blood cancer, while high reactive lymph counts signal potential viral infection.

Different white blood cell types use DNA differently and more or less of it can be condensed, resulting in different textures in the nuclei that provide a valuable source of data on the cell.

This GitHub page documents code used to crop visual microscope images to the nuclei of stained white blood cells (blasts and RLs). These images, after some additional processing to remove the background of each cropped nucleus (cell exterior and cytoplasm) were analyzed using [TWOMBLI](https://github.com/wershofe/TWOMBLI) to extract relatively interpretable numeric features quantifying the fractal and other structure observed in the nuclei images. This data was then used to train a [classifier](https://github.com/jhtullis/Blast_Classification) to differentiate between RLs and blasts.

Although this cropping process necessary to train and test this classifier can be done and was done manually for the majority of our research, the process is time consuming and rather boring. The dark purple color of the cells' nuclei in our images (Wright stain) made a cropping program seem achievable without the use of CNNs or other deep learning techniques.

## Details

In order to identify nuclei in an input image, the program starts by creating a binary mask - an array of ones and zeros with the same width and heighth as the original image. Visually, the mask looks approximately like a white sillohette of the nucleus on a black background. The intention is that white pixels (1) correspond to nuclei (dark purple stain in the center of cells) are, while black pixels (0) correspond to all other regions in the image.

![IMG_2244r](https://github.com/user-attachments/assets/52a3ed76-a3ed-4baa-b059-0b3bacc09136)
![next_steps](https://github.com/user-attachments/assets/1fb6cf26-e693-4714-b0e5-f51a80742043)

> *Above*: A color image of white blood cells. Nuclei (dark purple) are stained with Wright's stain. *Below*: A binary mask corresponding roughly with the purple-stained nuclei in the color image above.

The binary mask is currently created by thresholding the color channels of the image - each pixel has red, green and blue values that range from 0 to 255. A simple rule is applied - all pixels whose colors lie within certain ranges are assigned a 1 in the binary mask, while those pixels who have at least one color outside of it's designated range is assigned a zero. The specific ranges (or thresholds) for each color were chosen through trial and error for reliability and simplicity with the specific problem of identifying nuclei with Wright's stain. 

The binary mask generation process isn't perfect, so a single nucleus may have a mask with many fragmented pieces, giving it the appearence of having a fractured silhouette in the mask image. This is because not all pixels in the nucleus have green values high enough to clear the threshold - but lowering the threshold tends to introduce noise elsewhere in the image that further complicates the identification of the nucleus. Instead of introducing additional white pixels where they shouldn't be, the code in this repository works around the fragmentation of the nuclei by clustering these fragmented pieces together when they are sufficiently close.

![Diseño sin título (3)](https://github.com/user-attachments/assets/dd65021e-6cda-4171-a75a-d8c066d05d8b)

> *First image:* A photo including one or more white blood cells is taken. *Second image:* A binary mask is generated using color thresholding to identify the nuclei, but the nuclei are fragmented into many pieces. *Third image:* Contiguous regions in the binary mask are identified by recording the lower and upper limits in the x and y directions (shown using colored boxes, overlaid onto the color image). Smaller regions like those shown in red boxes are excluded unless they are adjacent to a larger region to ensure that unwanted features of the image, such as platlets, are excluded. *Fourth image:* An agglomerative clustering algorithm (code from Cullen Sun) can piece together these broken parts of the cell fairly reliably, at the cost of sometimes confusing two nearby cells for one.

The general principles used in this codebase are commonly in use in other applications, and there are many improvements that could be made. More sophisticated, reliable approaches to create a binary mask using thresholding could include transforming each pixel's red, green, and blue values to other values better describing things such as hue and saturation, a technique known as color spaces (see [this article](https://learnopencv.com/color-spaces-in-opencv-cpp-python/) by Vikas Gupta for a good introduction). Morphological image analysis techniques ([see this OpenCV tutorial](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)) might provide additional tools that could address limitations of the methodology presented here, especially for reversing mask fragmentation or removing small unwanted particles in the mask.

## Attribution

The basic agglomerative clustering algorithm in cullensun_cluster.py was developed by Cullen Sun (https://cullensun.medium.com/agglomerative-clustering-for-opencv-contours-cd74719b678e). The floodfill algorithm in mallicksatya_floodfill.py (not currently in use) was also adapted from code by Satya Mallick (https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/). Jason Henry Tullis developed the workflow and the remaining code for this application.

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

