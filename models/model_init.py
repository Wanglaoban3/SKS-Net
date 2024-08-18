import torch.nn as nn

def initialize_weights(model):
    if isinstance(model, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d, nn.Sigmoid, nn.Upsample, nn.PReLU)):
        return
    for module in model.children():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
        else:
            initialize_weights(module)
