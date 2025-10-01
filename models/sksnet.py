import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import SE, ConvModule
from torchvision import models

# article
# SKS-Net: Skeleton-strengthening network for surface defect segmentation
# fuse bn and conv
def _fuse_bn_kernel(conv_kernel, bn):
    """
    融合 Conv+BN 得到新的 (kernel, bias)
    conv_kernel: [out_c, in_c, kH, kW]
    bn: nn.BatchNorm2d
    """
    gamma = bn.weight
    beta  = bn.bias
    var   = bn.running_var
    mean  = bn.running_mean
    eps   = bn.eps

    std = (var + eps).rsqrt()
    fused_kernel = conv_kernel * (gamma * std).reshape(-1, 1, 1, 1)
    fused_bias   = beta - gamma * mean * std
    return fused_kernel, fused_bias

def _fuse_bn_tensor(conv_w, conv_b, bn):
    """返回融合后的 (kernel, bias)"""
    gamma = bn.weight
    beta  = bn.bias
    mean  = bn.running_mean
    var   = bn.running_var
    eps   = bn.eps
    std   = (var + eps).rsqrt()
    fused_w = conv_w * (gamma * std).reshape(-1, 1, 1, 1)
    fused_b = beta - gamma * mean * std if conv_b is not None else beta
    return fused_w, fused_b

# conv1x3 or conv3x1 --> 3x3
def _pad_to_3x3(kernel):
    """
    把 3x1 或 1x3 的核补零成 3x3
    kernel: [out_c, in_c, kH, kW]
    """
    out_c, in_c, kH, kW = kernel.shape
    assert (kH == 3 and kW == 1) or (kH == 1 and kW == 3)
    pad = (1, 1) if kW == 1 else (0, 0, 1, 1)  # (left, right, top, bottom)
    return F.pad(kernel, pad, mode='constant', value=0)

# conv3x3 dilation=2 --> conv 5x5
def _dilated2five(kernel_dil3):
    """
    把 dilation=2 的 3×3 核转换成等效 5×5 稀疏核
    kernel_dil3: [out_c, in_c, 3, 3]
    return:      [out_c, in_c, 5, 5]  (只有 3×3 的 9 个位置非 0)
    """
    out_c, in_c, kH, kW = kernel_dil3.shape
    assert kH == 3 and kW == 3
    k5 = torch.zeros(out_c, in_c, 5, 5, dtype=kernel_dil3.dtype, device=kernel_dil3.device)
    k5[:, :, ::2, ::2] = kernel_dil3  # 对应 dilation=2 的采样位置
    return k5


class SkeletonStrengtheningBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.stride = stride

        if deploy:
            # 推理阶段仅一个 3x3 Conv + BN
            self.rep_conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.rep_bn   = nn.BatchNorm2d(out_channels)
        else:
            # 训练阶段三条分支
            self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.conv3x1 = nn.Conv2d(in_channels, out_channels, (3, 1), stride=stride, padding=(1, 0), bias=False)
            self.conv1x3 = nn.Conv2d(in_channels, out_channels, (1, 3), stride=stride, padding=(0, 1), bias=False)
            self.bn3x3   = nn.BatchNorm2d(out_channels)
            self.bn3x1   = nn.BatchNorm2d(out_channels)
            self.bn1x3   = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.relu(self.rep_bn(self.rep_conv(x)))

        # 训练阶段三路相加
        out = self.bn3x3(self.conv3x3(x)) + \
              self.bn3x1(self.conv3x1(x)) + \
              self.bn1x3(self.conv1x3(x))
        return self.relu(out)

    def switch_to_deploy(self):
        """调用后永久转为单路 3x3 结构"""
        if self.deploy:
            return
        # 1. 把 3x1、1x3 补零成 3x3
        k3x3 = self.conv3x3.weight.data
        k3x1 = _pad_to_3x3(self.conv3x1.weight.data)
        k1x3 = _pad_to_3x3(self.conv1x3.weight.data)

        # 2. 分别融合各自 BN
        k3x3, b3x3 = _fuse_bn_kernel(k3x3, self.bn3x3)
        k3x1, b3x1 = _fuse_bn_kernel(k3x1, self.bn3x1)
        k1x3, b1x3 = _fuse_bn_kernel(k1x3, self.bn1x3)

        # 3. 相加得到等效核与偏置
        rep_kernel = k3x3 + k3x1 + k1x3
        rep_bias   = b3x3 + b3x1 + b1x3

        # 4. 新建单路 Conv-BN 并赋值
        self.rep_conv = nn.Conv2d(k3x3.shape[1], k3x3.shape[0], 3,
                                  stride=self.stride, padding=1, bias=False)
        self.rep_bn   = nn.BatchNorm2d(k3x3.shape[0])
        with torch.no_grad():
            self.rep_conv.weight.copy_(rep_kernel)
            self.rep_bn.weight.fill_(1)
            self.rep_bn.bias.copy_(rep_bias)
            self.rep_bn.running_mean.zero_()
            self.rep_bn.running_var.fill_(1)

        # 5. 删除训练分支
        for name in ['conv3x3', 'conv3x1', 'conv1x3', 'bn3x3', 'bn3x1', 'bn1x3']:
            delattr(self, name)
        self.deploy = True

    
class SkeletonStrengtheningBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.stride = stride

        if deploy:
            self.rep_conv = nn.Conv2d(in_channels, out_channels, 5, stride=stride, padding=2, bias=False)
            self.rep_bn   = nn.BatchNorm2d(out_channels)
        else:
            # 分支1：普通 3×3
            self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn3     = nn.BatchNorm2d(out_channels)
            # 分支2：dilation=2 的 3×3（感受野 5×5）
            self.conv5x5 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, dilation=2, padding=2, bias=False)
            self.bn5     = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.relu(self.rep_bn(self.rep_conv(x)))
        x3 = self.bn3(self.conv3x3(x))
        x5 = self.bn5(self.conv5x5(x))
        return self.relu(x3 + x5)

    def switch_to_deploy(self):
        if self.deploy:
            return
        # 1. 获取融合后的 3×3 核与偏置
        w3, b3 = _fuse_bn_tensor(self.conv3x3.weight, None, self.bn3)
        w5, b5 = _fuse_bn_tensor(self.conv5x5.weight, None, self.bn5)

        # 2. 把 dilation=2 的 3×3 展开成 5×5 稀疏核
        w5 = _dilated2five(w5)

        # 3. 把普通 3×3 补零成 5×5
        w3 = F.pad(w3, [1, 1, 1, 1])  # 上下左右各补 1

        # 4. 相加得到等效 5×5 核与偏置
        rep_w = w3 + w5
        rep_b = b3 + b5

        # 5. 构建单路 5×5 Conv-BN 并赋值
        self.rep_conv = nn.Conv2d(rep_w.shape[1], rep_w.shape[0], 5,
                                  stride=self.stride, padding=2, bias=False)
        self.rep_bn   = nn.BatchNorm2d(rep_w.shape[0])
        with torch.no_grad():
            self.rep_conv.weight.copy_(rep_w)
            self.rep_bn.weight.fill_(1)
            self.rep_bn.bias.copy_(rep_b)
            self.rep_bn.running_mean.zero_()
            self.rep_bn.running_var.fill_(1)

        # 6. 删除训练分支
        for name in ['conv3x3', 'conv5x5', 'bn3', 'bn5']:
            delattr(self, name)
        self.deploy = True
    
class FeatureFuseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = ConvModule(in_channels, out_channels)
        self.conv2 = ConvModule(out_channels, out_channels)
        self.se = SE(out_channels)

    def forward(self, low, high):
        high = self.up(high)
        out = self.conv1(torch.cat((high, low), dim=1))
        out = self.conv2(out)
        out = self.se(out)
        return out

class FCFDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, Block=SkeletonStrengtheningBlock):
        super(FCFDecoder, self).__init__()
        layers = []
        if num_layers > 1:
            middle_channels = out_channels // 2
        else:
            middle_channels = out_channels
        self.FA = SkeletonStrengtheningBlock2(in_channels, in_channels)
        self.CA = SE(in_channels)
        layers.append(Block(in_channels, middle_channels))
        for i in range(1, num_layers-1):
            layers.append(Block(middle_channels, middle_channels))
        layers.append(Block(middle_channels, out_channels))
        layers.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2))
        self.layers = nn.Sequential(*layers)

    def forward(self, low_x, high_x):
        high_x = self.FA(high_x)
        high_x = self.CA(high_x)
        x = high_x * low_x
        x = self.layers(x)
        return x

class SKSNet(nn.Module):
    def __init__(self, in_channel, num_classes, Block=SkeletonStrengtheningBlock, Encoder='vgg'):
        super(SKSNet, self).__init__()
        if Encoder == 'vgg':
            vgg = models.vgg19_bn()
            vgg.features[0] = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
            features = list(vgg.features.children())
            self.enc1 = nn.Sequential(*features[0:7])
            self.enc2 = nn.Sequential(*features[7:14])
            self.enc3 = nn.Sequential(*features[14:27])
            self.enc4 = nn.Sequential(*features[27:40])
            self.enc5 = nn.Sequential(*features[40:])

            self.dec5 = nn.Sequential(
                *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] + [Block(512, 512)] * 4)
            )

        self.dec4 = FCFDecoder(512, 256, 4, Block)
        self.dec3 = FCFDecoder(256, 128, 4, Block)
        self.dec2 = FCFDecoder(128, 64, 2, Block)
        self.dec1 = FCFDecoder(64, 64, 2, Block)

        self.dco5 = nn.Conv2d(512, 1, kernel_size=1)
        self.dco4 = nn.Conv2d(256, 1, kernel_size=1)
        self.dco3 = nn.Conv2d(128, 1, kernel_size=1)
        self.dco2 = nn.Conv2d(64, 1, kernel_size=1)
        self.dco1 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(F.interpolate(enc4, dec5.size()[2:], mode='bilinear'), dec5)
        dec3 = self.dec3(F.interpolate(enc3, dec4.size()[2:], mode='bilinear'), dec4)
        dec2 = self.dec2(F.interpolate(enc2, dec3.size()[2:], mode='bilinear'), dec3)
        dec1 = self.dec1(F.interpolate(enc1, dec2.size()[2:], mode='bilinear'), dec2)
        dco1 = F.interpolate(self.dco1(dec1), x.shape[-2:], mode='bilinear')

        return dco1
    
def fuse_model(model: nn.Module):
    """
    递归遍历 model，对所有
        SkeletonStrengtheningBlock 和 SkeletonStrengtheningBlock2
    执行 switch_to_deploy()（原地修改，不返回新模型）
    """
    for name, module in model.named_children():
        # 如果是目标 block，直接融合
        if isinstance(module, (SkeletonStrengtheningBlock, SkeletonStrengtheningBlock2)):
            module.switch_to_deploy()
        else:
            # 继续向下递归
            fuse_model(module)

if __name__ == '__main__':
    model = SKSNet(3, 4)
    x = torch.rand(4, 3, 224, 224)
    model.eval()
    # before fusion
    t1 = model(x)
    fuse_model(model)
    # after fusion
    t2 = model(x)
    print((t1 - t2).mean()) # tensor(0.0006, grad_fn=<MeanBackward0>)
    