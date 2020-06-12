import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def check_image(path):
    imglists = glob(path)

    for index in range(len(imglists)):
        maskname = imglists[index]
        imgname = maskname.replace('_anno', '')
        img = Image.open(imgname)
        mask = Image.open(maskname)
        mask = np.array(mask)
        mask[mask > 1] = 1
        if mask.sum() == 0:
            print(maskname)
            plt.subplot(121)
            plt.imshow(img)
            plt.subplot(122)
            plt.imshow(mask)
            plt.show()
            os.remove(maskname)
            os.remove(imgname)


if __name__ == "__main__":
    check_image(path='../data/mt2patchAugb/train/*_anno*.bmp')