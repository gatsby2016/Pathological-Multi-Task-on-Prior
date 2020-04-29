import os
import time
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
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


def main_worker(args):
    model = PSPNet(layers=args.layers, seg_classes=args.seg_classes, cls_classes= args.cls_classes, zoom_factor=args.zoom_factor,
                   branchSeg = args.branch_S, ProbTo = args.SegTo, BatchNorm=nn.BatchNorm2d)

    # modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    # modules_new = [model.ppm, model.cls, model.aux]
    # params_list = []
    # for module in modules_ori:
    #     params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    # for module in modules_new:
    #     params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))

    print("Classes: {}".format(args.cls_classes))
    print(model)
    model.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    # net.load_state_dict(torch.load('/media/yann/FILE/Kidney/Yan/Ablation_Study/ResNet40_First/model/resnet_params_0.pkl'))
    if args.weight:
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint['state_dict'])

    if args.resume:
        # checkpoint = torch.load(args.resume)
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    mean = [0.8106, 0.5949, 0.8088]
    std = [0.1635, 0.2139, 0.1225]
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])

    train_data = segclsData.SCdataset(args.train_root, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_data = segclsData.SCdataset(args.val_root, transform=train_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        epoch_log = epoch + 1
        train(train_loader, model, criterion, optimizer, epoch, args)
        filename = args.save_path + '/epoch_' + str(epoch_log) + '.pth'
        torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
        # validate2(val_loader, model, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    correct_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (input, mask, label) in enumerate(train_loader):
        input = input.cuda()
        label = label.cuda()
        if not args.HyValidation:
            mask = mask.cuda().long()
        else:
            mask = mask.cuda()#.long()
            input = torch.mul(input, mask.unsqueeze(dim=1).float())

        if args.branch_S:
            pre_seg, pre_classifier = model(input)
            # classifier_loss = criterion(pre_classifier, label)
            # seg_loss = criterion(pre_seg, mask)
            #
            # classifier_loss, seg_loss = torch.mean(classifier_loss), torch.mean(seg_loss)
            # loss = classifier_loss + args.seg_weight * seg_loss
### 2019.10.16 comment 4 lines above , and add 2 lines below.
            seg_loss = criterion(pre_seg, mask)
            loss = torch.mean(seg_loss)

        else:
            pre_classifier = model(input)
            # pre_classifier = model(input, mask) #### marking on 0622 recording, exp3
            classifier_loss = criterion(pre_classifier, label)
            loss = torch.mean(classifier_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0)

        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate the remaining time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        # current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        # for index in range(0, args.index_split):
        #     optimizer.param_groups[index]['lr'] = current_lr
        # for index in range(args.index_split, len(optimizer.param_groups)):
        #     optimizer.param_groups[index]['lr'] = current_lr * 10

        print('********Epoch********[{}/{}][{}/{}] ********Batch******** {batch_time.val:.3f} ({batch_time.avg:.3f})' 
              '********Remain******** {remain_time} ********_Loss_******** {loss_meter.val:.4f}.'
            .format(epoch+1, args.epochs, i + 1, len(train_loader), batch_time=batch_time,
                      remain_time=remain_time, loss_meter=loss_meter))

        prediction = torch.argmax(F.softmax(pre_classifier,dim=1), 1)
        correct_meter.update((prediction == label).sum().item())
        Acc = correct_meter.sum / (correct_meter.count * n )
        # print('Classification Result at epoch [{}/{}]: Acc/Acc/Acc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs, Acc, Acc, Acc))

        if args.branch_S:
            intersection, union, target = intersectionAndUnionGPU(pre_seg.max(1)[1], mask, args.seg_classes, args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
            print('Segmentation Result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))


def validate(val_loader, model, args):
    print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    correct_meter = AverageMeter()
    nums_meter = AverageMeter()

    model.eval()
    for i, (input, mask, label) in enumerate(val_loader):
        input = input.cuda()
        label = label.cuda()
        mask = mask.cuda().long()

        if args.branch_S:
            pre_seg, pre_classifier = model(input)
            intersection, union, target = intersectionAndUnionGPU(pre_seg.max(1)[1], mask, args.seg_classes, args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        else:
            pre_classifier = model(input)

        prediction = torch.argmax(F.softmax(pre_classifier,dim=1), 1)
        correct_meter.update((prediction == label).sum().item())
        nums_meter.update(input.size(0))


    Acc = correct_meter.sum / nums_meter.sum
    print('Classification Val Result Acc/Acc/{:.4f}/{:.4f}.'.format(Acc, Acc))

    if args.branch_S:
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        for i in range(args.seg_classes):
            print('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        print('Segmentation Val Result mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'
              .format(np.mean(iou_class), np.mean(accuracy_class), (sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10))))

    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def validate2(val_loader, model, args):
    # print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    correct_meter = AverageMeter()
    nums_meter = AverageMeter()

    real = np.array([])
    score = np.array([])

    model.eval()
    with torch.no_grad():
        for i, (input, mask, label) in enumerate(val_loader):
            input = input.cuda()
            label = label.cuda()
            mask = mask.cuda().long()
            # mask = mask.cuda()  # .long()
            # input = torch.mul(input, mask.unsqueeze(dim=1).float())

            if args.branch_S:
                pre_seg, pre_classifier = model(input)
                intersection, union, target = intersectionAndUnionGPU(pre_seg.max(1)[1], mask, args.seg_classes, args.ignore_label)
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
                # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            else:
                pre_classifier = model(input)
                # pre_classifier = model(input, mask.resize_((input.size(0), 40,40)))

            prediction = torch.argmax(F.softmax(pre_classifier,dim=1), 1)
            correct_meter.update((prediction == label).sum().item())
            nums_meter.update(input.size(0))

            score = np.concatenate((score, F.softmax(pre_classifier,dim=1)[:,1].cpu().numpy()), axis = 0)
            real = np.concatenate((real, label.cpu().numpy()),axis = 0)

    ######### AUC
    fpr,tpr, _ = roc_curve(real, score, pos_label = 1) ###计算真正率和假正率
    AUC = auc(fpr,tpr) ###计算auc的值

    Acc = correct_meter.sum / nums_meter.sum
    print('Classification Val Result Acc/AUC/{:.4f}/{:.4f}.'.format(Acc, AUC))

    if args.branch_S:
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        for i in range(args.seg_classes):
            print('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        print('Segmentation Val Result mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'
              .format(np.mean(iou_class), np.mean(accuracy_class), (sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10))))

    # print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main_worker(get_parser())
