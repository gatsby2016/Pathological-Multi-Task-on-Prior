import torch
from torch import nn
import torch.nn.functional as F

import utils.resnet as models

BatchNorm = nn.BatchNorm2d


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class MTNet(nn.Module):
    def __init__(self, segCla=2, clsCla=2, task='singleCls',
                 bins=(1, 2, 3, 6), BatchNorm=nn.BatchNorm2d, dropout=0.1, pretrained=False):
        super(MTNet, self).__init__()

        assert task in ['singleCls', 'singleSeg', 'MTwoP', 'MTonP'], \
            'Now Task only supports singleCls, singleSeg, MTwoP and MTonP'
        assert 2048 % len(bins) == 0

        self.task = task
        models.BatchNorm = BatchNorm

        resnet = models.resnet50(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        if self.task != 'singleSeg':
            fea_dim = 2048
            self.classifier = nn.Sequential(
                nn.Conv2d(fea_dim, fea_dim//2, kernel_size=3, stride=2, padding=1, bias=False),
                BatchNorm(fea_dim//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(fea_dim//2, fea_dim//4, kernel_size=3, stride=2, padding=1, bias=False),
                BatchNorm(fea_dim//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(fea_dim//4, fea_dim//4, kernel_size=3, stride=2, padding=1, bias=False),
                BatchNorm(fea_dim//4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1))
            self.fc = nn.Linear(fea_dim//4, clsCla)

        if self.task != 'singleCls':
            fea_dim = 2048
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2
            self.cls = nn.Sequential(
                nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
                BatchNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(512, segCla, kernel_size=1))

    def forward(self, x):  # def forward(self, x, mask):
        x_size = x.size()
        assert (x_size[2]) % 8 == 0 and (x_size[3]) % 8 == 0

        x = self.layer3(self.layer2(self.layer1(self.layer0(x))))
        x_tmp = self.layer4(x)  # x_tmp size is: [batch, channel, H/8, W/8]

        if self.task != 'singleCls':
            h = int((x_size[2]) / 8 * 8)
            w = int((x_size[3]) / 8 * 8)

            x_seg_tmp = self.cls(self.ppm(x_tmp))  # low resolution segmentation
            x_seg = F.interpolate(x_seg_tmp, size=(h, w), mode='bilinear', align_corners=True)

        if self.task == 'MTonP':
            x_soft = F.softmax(x_seg_tmp, dim=1)
            prob = x_soft[:, 1, :, :].unsqueeze(dim=1)  # prob size is: [batch, H/8, W/8]
            # prob = mask.unsqueeze(dim = 1).float()
            x_tmp = torch.mul(x_tmp, prob)
            # x_tmp = [x_tmp[:,channel,:,:].mul(prob) for channel in range(2048)]

        if self.task != 'singleSeg':
            x_cls = self.classifier(x_tmp)
            x_cls = self.fc(x_cls.view(x_cls.size(0), -1))

        if self.task == 'singleSeg':
            return x_seg
        elif self.task == 'singleCls':
            return x_cls
        else:
            return x_seg, x_cls


if __name__ == '__main__':
    input = torch.rand(4, 3, 473, 473).cuda()
    model = MTNet(bins=(1, 2, 3, 6), dropout=0.1, pretrained=True).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())
