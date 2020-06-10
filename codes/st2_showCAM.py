# -*- coding: utf-8 -*-
from PIL import Image
import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torchvision import transforms

from utils import config, mySegClsData
from utils.myMTNet import PSPNet
import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='myParams.yaml', help='config file')
    parser.add_argument('opts', help='see myParams.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def LoadNet(args, modelpath):
    model = PSPNet(layers=args.layers, seg_classes=args.seg_classes, cls_classes= args.cls_classes, zoom_factor=args.zoom_factor,
                   branchSeg = args.branch_S, ProbTo = args.SegTo, BatchNorm=nn.BatchNorm2d)
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['state_dict'])
    print(model)
    model.cuda().eval()
    return model


# hook the feature extractor 
features_blobs = [0]
def hook_feature(module, input, output):
#    features_blobs.append(output.data.cpu().numpy())
    features_blobs[0] = output.data.cpu().numpy()

    
# get the softmax weight after the average pooling process
def GetWeightSoftmax(network, prediction):
    params = list(network.parameters())
    weight_softmax = np.squeeze(params[174].data.cpu().numpy())
    weight = weight_softmax[prediction, :]
    return weight


# generate the class activation maps upsample to original size
def ReturnCAM(feature_conv, weight_softmax, input_h, input_w):
    _, _, nc, h, w = np.shape(feature_conv)
    cam = np.dot(weight_softmax, np.reshape(feature_conv, (nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam = cv2.resize(cam_img, (input_h, input_w))
    return output_cam


def deTransform(mean, std, tensor):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor


############################main function #####################################
def main():
    savepath = '../result/attentionmapProbTo/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    args = get_parser()


    ## image pre-processing
    mean = [0.8106, 0.5949, 0.8088]
    std = [0.1635, 0.2139, 0.1225]
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])
    val_data = mySegClsData.SCdataset(args.val_root, transform=train_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers)


    net = LoadNet(args, args.model_path)
    # net._modules.get(finalconv_name).register_forward_hook(hook_feature) # get feature maps
    net._modules['classifier']._modules.get('8').register_forward_hook(hook_feature)

    with torch.no_grad():
        for i, (input, mask, label) in tqdm.tqdm(enumerate(val_loader)):
            input = input.cuda()
            if not args.HyValidation:
                mask = mask.cuda().long()
            else:
                mask = mask.cuda()  # .long()
                input = torch.mul(input, mask.unsqueeze(dim=1).float())

            if args.branch_S:
                pre_seg, pre_classifier = net(input)
            else:
                pre_classifier = net(input)
                # pre_classifier = net(input, mask)

            prediction = torch.argmax(F.softmax(pre_classifier,dim=1), 1)
            prediction = prediction.cpu().squeeze().numpy()
            # print(prediction)

            weight = GetWeightSoftmax(net, prediction)
            CAMs = ReturnCAM(features_blobs, weight, 320,320)

            attentionmap = cv2.applyColorMap(CAMs, cv2.COLORMAP_JET)
            img = deTransform(mean, std, input).squeeze().cpu().transpose(0,2)
            img = img.transpose(0,1)
            # cv2.imwrite(savepath+str(i)+'.png',np.int32(img*255))

            if str(label[0].cpu().numpy()) == str(prediction):
                cv2.imwrite(savepath+str(i)+'_l'+str(label[0].cpu().numpy())+'_p'+str(prediction)+'.png', attentionmap*0.5 + np.int32(img*255)*0.5)
            else:
                cv2.imwrite(savepath + str(i) + '_l' + str(label[0].cpu().numpy()) + '_p' + str(prediction) + '__wrong.png',
                            attentionmap * 0.5 + np.int32(img * 255) * 0.5)


if __name__ == '__main__':
    main()