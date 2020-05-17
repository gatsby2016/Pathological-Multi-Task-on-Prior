import torch.utils.data as data
import torch
from PIL import Image
import glob
import cv2
import numpy as np
from utils import myTransforms


class SCdataset(data.Dataset):
    def __init__(self, txt_path, spc='val'):
        fh = open(txt_path, 'r')
        lists = []
        for line in fh:
            line = line.rstrip()
            words = line.split(' ')
            lists.append((words[0], words[1], int(words[2])))

        self.lists = lists
        self.spc = spc

        self.basicpro = myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=1),
                                                   myTransforms.RandomVerticalFlip(p=1),
                                                   myTransforms.AutoRandomRotation()])  # above: randomly selecting one
        self.morphpro = myTransforms.RandomElastic(alpha=2, sigma=0.06)
        self.colorpro = myTransforms.Compose([myTransforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
                                              myTransforms.RandomChoice([myTransforms.ColorJitter(saturation=(0.8, 1.2),hue=0.2),
                                                                         myTransforms.HEDJitter(theta=0.03)])
                                              ])
        self.tensorrpro = myTransforms.Compose([myTransforms.ToTensor(),  # operated on image
                                                myTransforms.Normalize([0.786, 0.5087, 0.7840], [0.1534, 0.2053, 0.1132])
                                                ])

    def __getitem__(self, index):
        imagename, maskname, label = self.lists[index]
        
        img = Image.open(imagename)
        mask = Image.open(maskname)
        if self.spc is 'train':
            img, mask = self.morphpro(self.basicpro(img, mask))
            img = self.tensorrpro(self.colorpro(img))
        else:
            img = self.tensorrpro(img)
        # img = cv2.imread(imagename, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # img = np.float32(img/255)
        # mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        # if self.transform is not None:
        #     img = self.transform(img)
        # mask = cv2.resize(mask,(40,40))
        mask = np.array(mask)
        mask[mask>1] = 1
        mask = Image.fromarray(mask)
        return img, mask, label  # only feature to use 40*40

    def __len__(self):
        return len(self.lists)


def deTransform(mean, std, tensor):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor