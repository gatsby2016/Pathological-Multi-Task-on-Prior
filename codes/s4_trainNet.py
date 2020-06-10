import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_curve, auc

from utils import mySegClsData
from utils.myMTNet import MTNet
from utils.myUtil import AverageMeter, intersectionAndUnionGPU, poly_learning_rateV2, get_parser


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    trainset = mySegClsData.SCdataset(args.train_root, spc='train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valset = mySegClsData.SCdataset(args.val_root, spc='val')
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print(f"Classification Classes: {args.clsCls}; Seg Classes: {args.segCls}")
    net = MTNet(segCla=args.segCls, clsCla=args.clsCls, task=args.task)

    if len(args.gpu) > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net.cuda()

    if args.restore:
        net.load_state_dict(torch.load(args.restore))
        print('####################Loading model...', args.restore)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.decay)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr)

    if not os.path.exists(os.path.join(args.modelckpts, args.task)):
        os.mkdir(os.path.join(args.modelckpts, args.task))

    trainwriter = SummaryWriter(log_dir='{}/{}'.format(os.path.join(args.resultpath, args.task), 'train'))
    valwriter = SummaryWriter(log_dir='{}/{}'.format(os.path.join(args.resultpath, args.task), 'val'))

    for epoch in range(args.epochs):
        poly_learning_rateV2(optimizer, epoch, args.epochs, args.base_lr, power=args.power)
        print('Current LR:', optimizer.param_groups[0]['lr'])

        train(trainloader, net, optimizer, criterion, epoch, trainwriter, args)
        validate(valloader, net, criterion, epoch, valwriter, args)

    trainwriter.close()
    valwriter.close()


def train(loader, model, optimizer, criterion, epoch, writer, args):
    assert args.task in ['singleCls', 'singleSeg', 'MTwoP', 'MTonP'],\
        'Now Task only supports singleCls, singleSeg, MTwoP and MTonP'
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    correct_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(loader)
    for i, (input, mask, label) in enumerate(loader):
        input = input.cuda()
        label = label.cuda().long()
        if not args.HyValidation:
            mask = mask.cuda().long()
        else:
            mask = mask.cuda()#.long()
            input = torch.mul(input, mask.unsqueeze(dim=1).float())

        if args.task == 'singleCls':
            pre_cls = model(input)
            # pre_classifier = model(input, mask) #### marking on 0622 recording, exp3
            cls_loss = criterion(pre_cls, label)
            loss = torch.mean(cls_loss)

        elif args.task == 'singleSeg':
            pre_seg = model(input)
            seg_loss = criterion(pre_seg, mask)
            loss = torch.mean(seg_loss)

        else:
            pre_seg, pre_cls = model(input)
            seg_loss = criterion(pre_seg, mask)
            cls_loss = criterion(pre_cls, label)
            cls_loss, seg_loss = torch.mean(cls_loss), torch.mean(seg_loss)
            loss = cls_loss + args.lossAlpha * seg_loss

            cls_loss_meter.update(cls_loss.item(), input.size(0))
            seg_loss_meter.update(seg_loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.task != 'singleSeg':
            prediction = torch.argmax(F.softmax(pre_cls, dim=1), 1)
            correct_meter.update((prediction == label).sum().item())

        if args.task != 'singleCls':
            intersection, union, target = intersectionAndUnionGPU(pre_seg.max(1)[1], mask, args.segCls)
            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())
            target_meter.update(target.cpu().numpy())

        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate the remaining time
        current_iter = epoch * len(loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        print(f'********Epoch********[{epoch+1}/{args.epochs}][{i + 1}/{len(loader)}]'
              f' ********Batch******** {batch_time.val:.3f} ({batch_time.avg:.3f})'
              f'********Remain******** {remain_time} ********_Loss_******** {loss_meter.val:.4f}.')

    writer.add_scalar('loss', loss_meter.sum / loss_meter.count, epoch)
    if args.task == 'MTwoP' or 'MTonP':
        writer.add_scalar('Only Seg loss', seg_loss_meter.sum / seg_loss_meter.count, epoch)
        writer.add_scalar('Only Cls loss', cls_loss_meter.sum / cls_loss_meter.count, epoch)

    if args.task != 'singleSeg':
        Acc = correct_meter.sum / ((correct_meter.count - 1) * args.batch_size + input.size(0))
        print(f'Classification Result at epoch [{epoch + 1}/{args.epochs}]: Acc {Acc:.4f}.')
        writer.add_scalar('Classification Accuracy', Acc, epoch)

    if args.task != 'singleCls':
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        print('Segmentation Result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'
              .format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
        writer.add_scalar('Segmentation mIoU', mIoU, epoch)
        writer.add_scalar('Segmentation mAcc', mAcc, epoch)
        writer.add_scalar('Segmentation allAcc', allAcc, epoch)

    # writer.add_images('Random Training Images', input)
    # writer.add_images('Random Training Masks ', mask.unsqueeze(1))
    # writer.add_graph(model, input)
    torch.save(model.state_dict(), os.path.join(args.modelckpts, args.task, str(epoch) + '.pkl'))
    # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)


def validate(val_loader, model, criterion, epoch, writer, args):
    print('>>>>>>>>>>>>>>>> Start Validation Evaluation >>>>>>>>>>>>>>>>')

    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()

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
            if args.task == 'singleCls':
                pre_cls = model(input)
                # pre_classifier = model(input, mask.resize_((input.size(0), 40,40)))
                cls_loss = criterion(pre_cls, label)
                loss = torch.mean(cls_loss)

            elif args.task == 'singleSeg':
                pre_seg = model(input)
                seg_loss = criterion(pre_seg, mask)
                loss = torch.mean(seg_loss)

            else:
                pre_seg, pre_cls = model(input)
                seg_loss = criterion(pre_seg, mask)
                cls_loss = criterion(pre_cls, label)
                cls_loss, seg_loss = torch.mean(cls_loss), torch.mean(seg_loss)
                loss = cls_loss + args.lossAlpha * seg_loss
                cls_loss_meter.update(cls_loss.item(), input.size(0))
                seg_loss_meter.update(seg_loss.item(), input.size(0))

            loss_meter.update(loss.item(), input.size(0))

            if args.task != 'singleCls':
                intersection, union, target = intersectionAndUnionGPU(pre_seg.max(1)[1], mask, args.segCls)
                intersection_meter.update(intersection.cpu().numpy())
                union_meter.update(union.cpu().numpy())
                target_meter.update(target.cpu().numpy())

            if args.task != 'singleSeg':
                prediction = torch.argmax(F.softmax(pre_cls, dim=1), 1)
                correct_meter.update((prediction == label).sum().item())
                nums_meter.update(input.size(0))
                score = np.concatenate((score, F.softmax(pre_cls, dim=1)[:, 1].cpu().numpy()), axis=0)
                real = np.concatenate((real, label.cpu().numpy()), axis=0)

    writer.add_scalar('loss', loss_meter.sum / loss_meter.count, epoch)
    if args.task == 'MTwoP' or 'MTonP':
        writer.add_scalar('Only Seg loss', seg_loss_meter.sum / seg_loss_meter.count, epoch)
        writer.add_scalar('Only Cls loss', cls_loss_meter.sum / cls_loss_meter.count, epoch)

    if args.task != 'singleSeg':
        fpr,tpr, _ = roc_curve(real, score, pos_label=1)  ###计算真正率和假正率
        AUC = auc(fpr, tpr) ###计算auc的值
        Acc = correct_meter.sum / nums_meter.sum
        print('Classification Val Result Acc/AUC/{:.4f}/{:.4f}.'.format(Acc, AUC))
        writer.add_scalar('Classification Accuracy', Acc, epoch)
        writer.add_scalar('Classification AUC Result', AUC, epoch)

    if args.task != 'singleCls':
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        for i in range(args.segCls):
            print(f'Class_{i} Result: iou/accuracy {iou_class[i]:.4f}/{accuracy_class[i]:.4f}.')
        print('Segmentation Val Result mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'
              .format(np.mean(iou_class), np.mean(accuracy_class), (sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10))))
        writer.add_scalar('Segmentation mIoU', np.mean(iou_class), epoch)
        writer.add_scalar('Segmentation mAcc', np.mean(accuracy_class), epoch)
        writer.add_scalar('Segmentation allAcc', (sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)), epoch)

    print('<<<<<<<<<<<<<<<<< End Validation Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    arg = get_parser()
    main(arg)
