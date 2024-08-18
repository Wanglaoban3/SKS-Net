import torch
import torch.nn.functional as F
from torch import nn
from .common import BasicBlock


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, Block=BasicBlock):
        super(_EncoderBlock, self).__init__()
        layers = [
            Block(in_channels, out_channels),
            Block(out_channels, out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, Block=BasicBlock, Attention=None):
        super(_DecoderBlock, self).__init__()
        layers = []
        layers.append(Block(in_channels, middle_channels))
        if Attention:
            layers.append(Attention(middle_channels))
        layers.append(Block(middle_channels, middle_channels))
        layers.append(nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, in_channel, num_classes, Block=BasicBlock, Attention=None):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(in_channel, 64, Block=Block)
        self.enc2 = _EncoderBlock(64, 128, Block=Block)
        self.enc3 = _EncoderBlock(128, 256, Block=Block)
        self.enc4 = _EncoderBlock(256, 512, dropout=True, Block=Block)
        self.center = _DecoderBlock(512, 1024, 512, Block, Attention)
        self.dec4 = _DecoderBlock(1024, 512, 256, Block, Attention)
        self.dec3 = _DecoderBlock(512, 256, 128, Block, Attention)
        self.dec2 = _DecoderBlock(256, 128, 64, Block, Attention)
        self.dec1 = nn.Sequential(
            Block(128, 64),
            Block(64, 64)
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        # initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        return final
