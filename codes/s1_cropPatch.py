import os
import numbers
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt


def Crop5Patch(img, size):
    """Crop the given PIL Image into four corners and the central crop
    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        crop_height = int(size)
        crop_width = int(size)
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
        crop_height, crop_width = size

    image_width, image_height = img.size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = img.crop((0, 0, crop_width, crop_height))
    tr = img.crop((image_width - crop_width, 0, image_width, crop_height))
    bl = img.crop((0, image_height - crop_height, crop_width, image_height))
    br = img.crop((image_width - crop_width, image_height - crop_height,
                   image_width, image_height))

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    center = img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))

    return tl, tr, bl, br, center


def basic_offline_augment(img):
    from torchvision import transforms

    rotation90 = transforms.RandomRotation(degrees=[90, 90])
    rotation180 = transforms.RandomRotation(degrees=[180, 180])
    rotation270 = transforms.RandomRotation(degrees=[270, 270])
    r90img = rotation90(img)
    r180img = rotation180(img)
    r270img = rotation270(img)

    from torchvision.transforms import functional as F
    himg = F.hflip(img)
    vimg = F.vflip(img)

    return img, r90img, r180img, r270img, himg, vimg


if __name__ == "__main__":
    imgsize = 320
    path = '/home/cyyan/projects/MToPrior/data/mt2/'
    spc = 'train'
    namelists = glob(path + spc + '*.bmp')

    if not os.path.exists(path.replace('mt2', 'mt2patchAugb/'+spc)):
        os.mkdir(path.replace('mt2', 'mt2patchAugb/'+spc))

    basicAug = True
    for name in namelists:
        print(name)
        img = Image.open(name)
        imglists = Crop5Patch(img, imgsize)

        for ind in range(len(imglists)):
            pimg = imglists[ind]

            if basicAug:
                augimgs = basic_offline_augment(pimg)

                for j in range(len(augimgs)):
                    aug = augimgs[j]
                    # plt.subplot('23'+str(j+1))
                    # plt.imshow(aug)
                    aug.save(name.replace('mt2', 'mt2patchAugb/'+spc).replace('.bmp', '_'+str(ind)+'_'+str(j)+'.bmp'))
                # plt.show()
            else:
                pimg.save(name.replace('mt2', 'mt2patch/' + spc).replace('.bmp', '_' + str(ind) + '.bmp'))