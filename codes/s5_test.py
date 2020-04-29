import os
import time
import argparse
import glob
from sklearn.metrics import roc_curve, auc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torchvision import transforms

from utils import config, segclsData
from utils.util import AverageMeter, intersectionAndUnionGPU #,poly_learning_rate
from utils.pspnet import PSPNet


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='Params.yaml', help='config file')
    parser.add_argument('opts', help='see Params.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def deTransform(mean, std, tensor):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor


def validate(val_loader, model, args):
    # print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    correct_meter = AverageMeter()
    TP = AverageMeter()
    TPFP =  AverageMeter()
    nums_meter = AverageMeter()

    real = np.array([])
    score = np.array([])

    model.eval()
    with torch.no_grad():
        for i, (input, mask, label) in enumerate(val_loader):
            input = input.cuda()
            label = label.cuda()
            if not args.HyValidation:
                mask = mask.cuda().long()
            else:
                mask = mask.cuda()  # .long()
                input = torch.mul(input, mask.unsqueeze(dim=1).float())

            if args.branch_S:
                pre_seg, pre_classifier = model(input)
                intersection, union, target = intersectionAndUnionGPU(pre_seg.max(1)[1], mask, args.seg_classes, args.ignore_label)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
                # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            else:
                pre_classifier = model(input)
                # pre_classifier = model(input, mask)

            prediction = torch.argmax(F.softmax(pre_classifier,dim=1), 1)
            correct_meter.update((prediction == label).sum().item())
            TP.update(((prediction == 1) & (label == 1)).sum().item())
            TPFP.update((prediction == 1).sum().item())

            nums_meter.update(input.size(0))

            score = np.concatenate((score, F.softmax(pre_classifier,dim=1)[:,1].cpu().numpy()), axis = 0)
            real = np.concatenate((real, label.cpu().numpy()),axis = 0)

    ######### AUC
    fpr,tpr, _ = roc_curve(real, score, pos_label = 1) ###计算真正率和假正率
    AUC = auc(fpr,tpr) ###计算auc的值

    Recall = TP.sum / (real == 1).sum()
    Precision = TP.sum / TPFP.sum
    SPC = (correct_meter.sum - TP.sum) / (real == 0).sum()
    Acc = correct_meter.sum / nums_meter.sum
    print('Classification Val Result Acc/AUC/Precision/Recall/SPC {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(Acc, AUC, Precision,  Recall, SPC))

    if args.branch_S:
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        for i in range(args.seg_classes):
            print('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        print('Segmentation Val Result mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'
              .format(np.mean(iou_class), np.mean(accuracy_class), (sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10))))

    # print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def main():
    args = get_parser()

    mean = [0.8106, 0.5949, 0.8088]
    std = [0.1635, 0.2139, 0.1225]

    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])

    val_data = segclsData.SCdataset(args.val_root, transform=train_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = PSPNet(layers=args.layers, seg_classes=args.seg_classes, cls_classes= args.cls_classes, zoom_factor=args.zoom_factor,
                   branchSeg = args.branch_S, ProbTo = args.SegTo, BatchNorm=nn.BatchNorm2d)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # print("=> loaded checkpoint '{}'".format(args.model_path))
    # print(model)
    model.cuda()

    validate(val_loader, model, args)

    # with torch.no_grad():
    #     print('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')
    #     for i, (input, target) in enumerate(test_loader):
    #         # intersection_meter = AverageMeter()
    #         # union_meter = AverageMeter()
    #         # target_meter = AverageMeter()
    #
    #         input = input.to(device)
    #         target = target.to(device).long()
    #         output = model(input)
    #         if args.zoom_factor != 8:
    #             output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
    #
    #         output = output.max(1)[1]
    #         result = np.uint8(255 * output.squeeze().numpy())
    #         mask = np.uint8(255 * target.squeeze().numpy())
    #         img = np.uint8(255 * deTransform(mean, std, input).squeeze().numpy())  # 将颜色反标准化
    #         img = img[::-1, :, :].transpose((1, 2, 0))  # 转为opencv读取的格式 H*W*C， RGB --> BGR
    #
    #         filename = str(i)
    #         cv2.imwrite('../result/testResult_semseg/'+filename+'.png', img)
    #         cv2.imwrite('../result/testResult_semseg/'+filename+'_gt.png', mask)
    #         cv2.imwrite('../result/testResult_semseg/'+filename+'_pre.png', result)
    #         print(filename)# break
    #         # cv2.waitKey(0)
    #     print('<<<<<<<<<<<<<<<<< End Testing <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    # model_path = '../model_ValidationFeature/epoch_39.pth'
    main()
    #
    # model_path = '../model_ValidationFeature/epoch_'
    # for epoch in range(1,100+1):
    #     print('Now testing epoch: ', epoch)
    #     main(model_path+str(epoch)+'.pth')
    #     when probto, best testA performance is 61, classifier accuracy is 0.9498 seg mIOU is 0.8347
        # when noProbTo, best testA performance is 57, classifier accuracy is 0.9642, seg mIOU is 0.8378