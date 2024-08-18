import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 不要在每个卷积层后加bn，而是要把卷积后的结果加在一起后再bn，不然很难收敛，不要开bias
class SkeletonStrengtheningBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkeletonStrengtheningBlock1, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        # self.use_avg = False
        # if in_channels == out_channels:
        #     self.avg = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        #     self.use_avg = True
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        square = self.conv3x3(x)
        ver = self.conv3x1(x)
        hor = self.conv1x3(x)
        # if self.use_avg:
        #     avg = self.avg(x)
        #     out = self.bn(square+ver+hor+avg)
        out = self.bn(square+ver+hor)
        out = self.relu(out)
        return out


class SkeletonStrengtheningBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkeletonStrengtheningBlock2, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        self.avg = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        square = self.conv3x3(x)
        ver = self.conv3x1(x)
        hor = self.conv1x3(x)
        avg = self.avg(x)
        out = self.bn(square+ver+hor+avg)
        out = self.relu(out)
        return out


class SkeletonStrengtheningBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkeletonStrengtheningBlock3, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        self.avg = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        square = self.conv3x3(x)
        ver = self.conv3x1(x)
        hor = self.conv1x3(x)
        avg = self.avg(x)
        out = self.bn(square+ver+hor+avg)
        out = self.conv1x1(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class SkeletonStrengtheningBlock4(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkeletonStrengtheningBlock4, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        self.avg = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        square = self.conv3x3(x)
        ver = self.conv3x1(x)
        hor = self.conv1x3(x)
        avg = self.avg(x)
        if square.shape[1] == x.shape[1]:
            out = self.bn(x+square+ver+hor+avg)
        else:
            out = self.bn(square+ver+hor+avg)
        out = self.relu(out)
        return out


class SkeletonStrengtheningBlock5(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkeletonStrengtheningBlock5, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        if in_channels == out_channels:
            self.conv1x1 = nn.Identity()
        else:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=0, bias=False)
        self.avg = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        square = self.conv3x3(x)
        ver = self.conv3x1(x)
        hor = self.conv1x3(x)
        avg = self.avg(x)
        center = self.conv1x1(x)
        out = self.bn(square+ver+hor+avg+center)
        out = self.relu(out)
        return out


class SkeletonStrengtheningBlock6(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkeletonStrengtheningBlock6, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.conv3x1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.conv1x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        self.avg = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        square = self.conv3x3(x)
        ver = self.conv3x1(x)
        hor = self.conv1x3(x)
        avg = self.avg(x)
        out = self.bn1(square+ver+hor+avg)
        out = self.relu(out)
        out = self.conv1x1(out)
        out = self.bn2(out)
        out = self.relu(out) + x
        return out


class SkeletonStrengtheningBlock7(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkeletonStrengtheningBlock7, self).__init__()
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        self.avg = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ver = self.conv3x1(x)
        hor = self.conv1x3(x)
        avg = self.avg(x)
        out = self.bn1(ver+hor+avg)
        out = self.relu(out)
        return out


class SkeletonStrengtheningBlock8(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkeletonStrengtheningBlock8, self).__init__()
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), stride=stride, padding=(2, 0), bias=False)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=stride, padding=(0, 2), bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(2, 2), dilation=2, bias=False)
        self.avg = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.basic = BasicBlock(out_channels, out_channels)

    def forward(self, x):
        ver = self.conv3x1(x)
        hor = self.conv1x3(x)
        squre = self.conv3x3(x)
        avg = self.avg(x)
        out = self.bn1(ver+hor+avg+squre)
        out = self.relu(out)
        out = self.basic(out)
        return out


class SkeletonStrengtheningBlock9(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkeletonStrengtheningBlock9, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False)
        self.conv1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False)
        self.avg = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        square = self.bn1(self.conv3x3(x))
        ver = self.bn2(self.conv3x1(x))
        hor = self.bn3(self.conv1x3(x))
        avg = self.bn4(self.avg(x))
        out = square + ver + hor + avg
        out = self.relu(out)
        return out


class SkeletonStrengtheningBlock10(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SkeletonStrengtheningBlock10, self).__init__()
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=2, padding=2, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        out = self.bn1(x3+x5)
        out = self.relu(out)
        return out


# Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SSAttention1(nn.Module):
    def __init__(self, kernel_size=7):
        super(SSAttention1, self).__init__()
        # self.max = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        # self.avg = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.square = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.ver = nn.Conv2d(2, 1, (1, kernel_size), padding=(0, kernel_size//2), bias=False)
        self.hor = nn.Conv2d(2, 1, (kernel_size, 1), padding=(kernel_size//2, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        sqr = self.square(x)
        ver = self.ver(x)
        hor = self.hor(x)
        return self.sigmoid(sqr+ver+hor)
