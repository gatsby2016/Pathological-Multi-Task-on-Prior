import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc

from utils import mySegClsData
from utils.myMTNet import MTNet
from utils.myUtil import AverageMeter, intersectionAndUnionGPU, get_parser


def independent_evaluation(val_loader, model, args):
    print('>>>>>>>>>>>>>>>> Start Independent Evaluation >>>>>>>>>>>>>>>>')

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
        for i, (input, mask, label) in enumerate(val_loader):
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

            if args.task != 'singleSeg':
                prediction = torch.argmax(F.softmax(pre_cls, dim=1), 1)
                correct_meter.update((prediction == label).sum().item())
                TP.update(((prediction == 1) & (label == 1)).sum().item())
                TPFP.update((prediction == 1).sum().item())

                nums_meter.update(input.size(0))
                score = np.concatenate((score, F.softmax(pre_cls, dim=1)[:, 1].cpu().numpy()), axis=0)
                real = np.concatenate((real, label.cpu().numpy()), axis=0)

    if args.task != 'singleSeg':
        fpr,tpr, _ = roc_curve(real, score, pos_label=1)  ###计算真正率和假正率
        AUC = auc(fpr, tpr) ###计算auc的值
        specific = (correct_meter.sum - TP.sum) / (real == 0).sum()
        recall = TP.sum / (real == 1).sum()
        precision = TP.sum / TPFP.sum

        Acc = correct_meter.sum / nums_meter.sum
        print(f'Classification Result Acc/AUC/Precision/Recall/SPC {Acc:.4f}/{AUC:.4f}/{precision:.4f}/{recall:.4f}/{specific:.4f}.')

    if args.task != 'singleCls':
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        for i in range(args.segCls):
            print(f'Class_{i} Result: iou/accuracy {iou_class[i]:.4f}/{accuracy_class[i]:.4f}.')
        print('Segmentation Val Result mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'
              .format(np.mean(iou_class), np.mean(accuracy_class), (sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10))))

    print('<<<<<<<<<<<<<<<<< End Independent Evaluation <<<<<<<<<<<<<<<<<')


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    testloader = DataLoader(mySegClsData.SCdataset(args.val_root, spc='val'),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print(f"Classification Classes: {args.clsCls}; Seg Classes: {args.segCls}")
    net = MTNet(segCla=args.segCls, clsCla=args.clsCls, task=args.task)

    if len(args.gpu) > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net.cuda()

    if args.restore:
        net.load_state_dict(torch.load(os.path.join(args.modelckpts, args.task, args.restore)))
        print('####################Loading model...', args.restore)

    independent_evaluation(testloader, net, args)


if __name__ == '__main__':
    arg = get_parser()
    main(arg)