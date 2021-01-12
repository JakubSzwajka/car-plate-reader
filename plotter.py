from local_utils import logger
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import settings

def plot_plate(LpImg):
    plt.imshow(LpImg)
    plt.axis(False)
    plt.show()

def plot_grey_scale_plates(plate_image, gray, blur, binary, thre_mor ):
    fig = plt.figure(figsize=(12,7))
    plt.rcParams.update({"font.size":18})
    grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig)
    plot_image = [plate_image, gray, blur, binary,thre_mor]
    plot_name = ["plate_image","gray","blur","binary","dilation"]
    
    for i in range(len(plot_image)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.title(plot_name[i])
        if i ==0:
            plt.imshow(plot_image[i])
        else:
            plt.imshow(plot_image[i],cmap="gray")

    plt.savefig(settings.PROCES_PHOTO_FOLDER + "threshding.png", dpi=300)
    plt.show()


def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def plot_letters_on_plate( test_roi , crop_characters):
    fig = plt.figure(figsize=(10,6))
    plt.axis(False)
    plt.imshow(test_roi)
    plt.savefig(settings.PROCES_PHOTO_FOLDER + 'grab_digit_contour.png',dpi=300)
    plt.show()

def plot_binary_plate(crop_characters):
    fig = plt.figure(figsize=(14,4))
    grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)

    for i in range(len(crop_characters)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.imshow(crop_characters[i],cmap="gray")
    
    plt.savefig(settings.PROCES_PHOTO_FOLDER + "segmented_leter.png",dpi=300)
    plt.show()