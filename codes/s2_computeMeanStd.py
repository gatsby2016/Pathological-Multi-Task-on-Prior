# coding: utf-8
# scripts for computing mean and std
# from __future__ import print_function
import numpy as np
import random
import os
import cv2
import argparse
import glob

#%% define mean and std compution function
def ComputeMeanStd(imagename):
    img = cv2.imread(imagename)
    img = img[:,:, (2,1,0)] # BGR to RGB
    arr = np.array(img)/255.
    mean_vals = np.mean(arr, (0,1))
    std_vals = np.std(arr, (0,1)) # Compute the mean  and std along the specified axis 0,1.
    return mean_vals, std_vals

# define compution function for filelist
def ComputeMeanStd_List(List):
    Mean_val = []
    Std_val = []
    for name in List:
        # filename = os.path.join(path, name)
        M, S = ComputeMeanStd(name)
        Mean_val.append(M)
        Std_val.append(S)
    datamean = np.mean(Mean_val, 0)
    datastd = np.mean(Std_val, 0)
    return datamean, datastd


#### get args
def GetArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--part', type=int, default=1, choices=[1, 10, 100, 1000,10000],
                        help='1/Part of len(files), have 1, 10, 100, 1000, 10000 choices')
    parser.add_argument('-P', '--path', type=str,
                        default=['/home/cyyan/projects/CaSoGP/data/train/benign_Tran/',
                                 '/home/cyyan/projects/CaSoGP/data/train/malignant_Tran/'
                                 ], help='Img path1')
    parser.add_argument('-S', '--savepath', type=str, default='../result/MeanStd.npz',
                        help='Path to save the MeanStd value')
    # parser.add_argument('--multi_img', action='store_true', default=True,
    #                     help='use multi images or single image')

    args = parser.parse_args()
    return args


#%%
if __name__ == '__main__':
    args = GetArgs()

    Allmean = []
    Allstd = []
    for path in args.path:
        # filelist = os.listdir(path)
        filelist = glob.glob(path + '*.png')

        # files = []
        # [files.append(f[:-9]+'.png') for f in filelist]
        # filelist = files

        random.shuffle(filelist)
        nums = len(filelist)//args.part # randomly selected 1/part data for compution
        Singlemean, Singlestd = ComputeMeanStd_List(filelist[0:nums])
        Allmean = np.append(Allmean, Singlemean)
        Allstd = np.append(Allstd, Singlestd)

    NormMean = np.around(np.mean(Allmean.reshape(-1,3), axis=0), 4)
    NormStd  = np.around(np.mean(Allstd.reshape(-1,3), axis=0), 4) # mean of A part and B part ,and decimal is 3
    print("normMean = {}".format(NormMean))
    print("normStd = {}".format(NormStd))
    print('transforms.Normalize({}, {})'.format(NormMean, NormStd))

    np.savez(args.savepath, NormMean = NormMean, NormStd = NormStd)
    ######################################## load the npz data
    # npzfile = np.load('MeanStd.npz')
    # Mean, Std = npzfile['NormMean'], npzfile['NormStd']
#%%

# 2019.6.10
# normMean = [0.8106 0.5949 0.8088]
# normStd = [0.1635 0.2139 0.1225]
# transforms.Normalize([0.8106 0.5949 0.8088], [0.1635 0.2139 0.1225])