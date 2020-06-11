import os
import numpy as np
import argparse
import json
from tqdm import tqdm
from PIL import Image
import cv2

import torch
from torch import nn
import torch.nn.init as initer
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc

from utils import myConfig


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def poly_learning_rateV2(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES,power), 8)


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.modules.conv._ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.modules.batchnorm._BatchNorm)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def get_parser():
    parser = argparse.ArgumentParser(description='MT (Segmentation & Classification) on prior by PyTorch')
    parser.add_argument('--config', type=str, default='myParams.yaml', help='config file')
    parser.add_argument('opts', help='see myParams.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None, "Please provide config file for myParams."
    cfg = myConfig.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = myConfig.merge_cfg_from_list(cfg, args.opts)

    if not os.path.exists(os.path.join(cfg.resultpath, cfg.task)):
        os.mkdir(os.path.join(cfg.resultpath, cfg.task))
    with open(os.path.join(cfg.resultpath, cfg.task, cfg.configs), 'w') as f:
        json.dump(cfg, f, indent=2)

    return cfg


def de_transform(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor


def independent_evaluation(val_loader, model, args, GEN=False):
    print('>>>>>>>>>>>>>>>> Start Independent Evaluation or Inference >>>>>>>>>>>>>>>>')
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    correct_meter = AverageMeter()
    nums_meter = AverageMeter()
    TP = AverageMeter()
    TPFP = AverageMeter()

    real = np.array([])
    score = np.array([])

    model.eval()
    with torch.no_grad():
        for i, (input, mask, label) in tqdm(enumerate(val_loader)):
            input = input.cuda()
            label = label.cuda()
            mask = mask.cuda().long()
            # mask = mask.cuda()  # .long()
            # input = torch.mul(input, mask.unsqueeze(dim=1).float())

            if args.task == 'singleCls':
                pre_cls = model(input)
                # pre_classifier = model(input, mask.resize_((input.size(0), 40,40)))

            elif args.task == 'singleSeg':
                pre_seg = model(input)

            else:
                pre_seg, pre_cls = model(input)

            if args.task != 'singleCls':
                intersection, union, target = intersectionAndUnionGPU(pre_seg.max(1)[1], mask, args.segCls)
                intersection_meter.update(intersection.cpu().numpy())
                union_meter.update(union.cpu().numpy())
                target_meter.update(target.cpu().numpy())

                if GEN:
                    if not os.path.exists(os.path.join(args.resultpath, args.task, 'GenPred')):
                        os.mkdir(os.path.join(args.resultpath, args.task, 'GenPred'))
                    probs = F.softmax(pre_seg, dim=1)[:, 1, :, :]
                    img = de_transform(input, [0.786, 0.5087, 0.7840], [0.1534, 0.2053, 0.1132])
                    for one in range(probs.size(0)):
                        pro = np.uint8(255 * probs[one, ...].squeeze().cpu().numpy())
                        probmap = cv2.applyColorMap(pro, cv2.COLORMAP_JET)
                        gt = np.uint8(255 * mask[one, ...].squeeze().cpu().numpy())
                        im = np.uint8(255 * img[one, ...].squeeze().cpu().numpy()).transpose(1, 2, 0)
                        cv2.imwrite(os.path.join(args.resultpath, args.task,
                                                 'GenPred', str(i*args.batch_size+one)+'_pre.png'), probmap)
                        cv2.imwrite(os.path.join(args.resultpath, args.task,
                                                 'GenPred', str(i*args.batch_size+one)+'_gt.png'), gt)
                        cv2.imwrite(os.path.join(args.resultpath, args.task,
                                                 'GenPred', str(i*args.batch_size+one)+'_img.png'), im)

            if args.task != 'singleSeg':
                prediction = torch.argmax(F.softmax(pre_cls, dim=1), 1)
                correct_meter.update((prediction == label).sum().item())
                TP.update(((prediction == 1) & (label == 1)).sum().item())
                TPFP.update((prediction == 1).sum().item())

                nums_meter.update(input.size(0))
                score = np.concatenate((score, F.softmax(pre_cls, dim=1)[:, 1].cpu().numpy()), axis=0)
                real = np.concatenate((real, label.cpu().numpy()), axis=0)

    if args.task != 'singleSeg':
        fpr,tpr, _ = roc_curve(real, score, pos_label=1)
        AUC = auc(fpr, tpr)
        specific = (correct_meter.sum - TP.sum) / (real == 0).sum()
        recall = TP.sum / (real == 1).sum()
        precision = TP.sum / TPFP.sum

        Acc = correct_meter.sum / nums_meter.sum
        print(f'Classification Result Acc/AUC/Precision/Recall/SPC '
              f'{Acc:.4f}/{AUC:.4f}/{precision:.4f}/{recall:.4f}/{specific:.4f}.')

    if args.task != 'singleCls':
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        for i in range(args.segCls):
            print(f'Class_{i} Result: iou/accuracy {iou_class[i]:.4f}/{accuracy_class[i]:.4f}.')
        print('Segmentation Val Result mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'
              .format(np.mean(iou_class), np.mean(accuracy_class),
                      (sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10))))
    print('<<<<<<<<<<<<<<<<< End Independent Evaluation or Inference<<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    pass