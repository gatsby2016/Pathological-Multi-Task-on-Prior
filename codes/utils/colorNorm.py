# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:25:35 2018
@author: yann
original link: https://blog.csdn.net/c_chuxin/article/details/83181299
reference link if you want to learn the color transfer method implemented
[HERE](https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/)
"""
# from skimage import io, data, color
import numpy as np
import cv2
import os
from PIL import Image


# import color_transfer
def image_stats(image):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space

    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def color_transfer(refpath, target):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    Parameters:
    -------
    source: NumPy array
        OpenCV image in BGR color space (the source image)
    target: NumPy array
        OpenCV image in BGR color space (the target image)

    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (uint8)
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.imread(refpath)
    target = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return Image.fromarray(cv2.cvtColor(transfer, cv2.COLOR_BGR2RGB))


def show_image(title, image, width=300):
    # resize the image to have a constant width, just to
    # make displaying the images take up less screen real
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # show the resized image
    cv2.imshow(title, resized)


###########################################################################
# colorReference = cv2.imread('forTransfer/2.png')
# target = cv2.imread('forTransfer/test.tif')
# transfer = color_transfer(colorReference, target)
#
# # check to see if the output image should be saved
# # show the images and wait for a key press
# show_image("colorReference", colorReference)
# show_image("target", target)
# show_image("Transfer", transfer)
# cv2.waitKey(0)
#


if __name__ == "__main__":
    path = '../data/train/benign/'
    savepth = '../data/train/benign_Tran/'
    files = os.listdir(path)
    for file in files:
        target = cv2.imread(path+file)
    # transfer the color distribution from the source image to the target image
        transfer = color_transfer('reference.png', target)
        cv2.imwrite(savepth+file, transfer)