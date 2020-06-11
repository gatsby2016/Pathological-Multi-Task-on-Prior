import os

import torch
import torch.nn.parallel
from torch.utils.data import DataLoader

from utils import mySegClsData
from utils.myMTNet import MTNet
from utils.myUtil import get_parser, independent_evaluation


def independent_main(args, gen=False):
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

    independent_evaluation(testloader, net, args, GEN=gen)


if __name__ == '__main__':
    arg = get_parser()
    independent_main(arg, gen=False)
    # independent_main(arg, gen=True)