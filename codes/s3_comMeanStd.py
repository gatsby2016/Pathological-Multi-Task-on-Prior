import numpy as np
import cv2
import argparse


def ComputeMeanStd(imagename):
    img = cv2.imread(imagename)
    img = img[:,:, (2, 1, 0)]  # BGR to RGB
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

    parser.add_argument('--part', type=int, default=1, choices=[1, 10, 100, 1000, 10000],
                        help='1/Part of len(files), have 1, 10, 100, 1000, 10000 choices')
    parser.add_argument('-P', '--path', type=str,
                        default='../data/mt2patch/train.txt', help='Img txt path')
    parser.add_argument('-S', '--savepath', type=str, default='../results/mt2patchMeanStd.npz',
                        help='Path to save the MeanStd value')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    np.random.seed(2020)
    args = GetArgs()

    Allmean = []
    Allstd = []

    fh = open(args.path, 'r')
    lists = []
    for line in fh:
        line = line.rstrip()
        words = line.split(' ')
        lists.append(words[0])

    np.random.shuffle(lists)
    nums = len(lists)//args.part  # randomly selected 1/part data for compution
    Singlemean, Singlestd = ComputeMeanStd_List(lists[0:nums])
    Allmean = np.append(Allmean, Singlemean)
    Allstd = np.append(Allstd, Singlestd)

    NormMean = np.around(np.mean(Allmean.reshape(-1,3), axis=0), 4)
    NormStd  = np.around(np.mean(Allstd.reshape(-1,3), axis=0), 4) # mean of A part and B part ,and decimal is 3
    print("normMean = {}".format(NormMean))
    print("normStd = {}".format(NormStd))
    print('transforms.Normalize({}, {})'.format(NormMean, NormStd))

    np.savez(args.savepath, NormMean = NormMean, NormStd = NormStd)