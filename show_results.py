import argparse
import os
import torch
import torchvision.transforms
from dataset import data
from loss import MultiScaleLoss

from utils import transform, AverageMeter, Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='seg_former',
                    choices=['dlasdd', 'seg_net', 'ss_net', 'ss_net2', 'ss_net3', 'deeplabv3', 'psp_net', 'u_net',
                             'edr_net', 'resunet_pp', 'pga_net', 'seg_former', 'a_net'])
parser.add_argument('--block', type=str, default='basic')
parser.add_argument('--attention', type=str, default='')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--in_channel', type=int, default=3)
parser.add_argument('--benchmark', type=str, default='RSDD2',
                    choices=['KolektorSDD1', 'KolektorSDD2', 'CrackForest', 'RSDD1', 'Magnetic', 'RSDD2'])
parser.add_argument('--checkpoint', type=str, default='save/RSDD2_seg_former_basic__epoch120_miou40.51_f1score57.66_precision55.42_recall60.08_dice57.66.pth')
parser.add_argument('--save_path', type=str, default='./data_show')
parser.add_argument('--data_root', type=str, default='C:/wrd/RSDDs/Type-II RSDDs dataset')
args = parser.parse_args()

# global config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# select block
from models import *

if args.block == 'basic':
    Block = BasicBlock
elif args.block == 'ssblock1':
    Block = SkeletonStrengtheningBlock1
elif args.block == 'ssblock2':
    Block = SkeletonStrengtheningBlock2
elif args.block == 'ssblock3':
    Block = SkeletonStrengtheningBlock3
elif args.block == 'ssblock4':
    Block = SkeletonStrengtheningBlock4
elif args.block == 'ssblock5':
    Block = SkeletonStrengtheningBlock5
elif args.block == 'ssblock9':
    Block = SkeletonStrengtheningBlock9
else:
    raise AttributeError("输入正确的模块名")

# select attention block
if args.attention == 'SE':
    Attention = SE
elif args.attention == 'CBAM':
    Attention = CBAM
elif args.attention == 'CBAM':
    Attention = CBAM
else:
    Attention = None

# select model
if args.model_type == 'dlasdd':
    model = dlasdd.SegNetwork(args.in_channel, 1024, args.num_classes)
elif args.model_type == 'psp_net':
    model = PSPNet(args.in_channel, args.num_classes)
elif args.model_type == 'seg_net':
    model = SegNet(args.in_channel, args.num_classes, Block, Attention)
elif args.model_type == 'u_net':
    model = UNet(args.in_channel, args.num_classes, Block, Attention)
elif args.model_type == 'deeplabv3':
    model = DeepLabV3(args.in_channel, args.num_classes)
elif args.model_type == 'ss_net':
    model = SSNet(args.in_channel, args.num_classes, Block, Attention, Encoder='vgg')
elif args.model_type == 'ss_net2':
    model = SSNet2(args.in_channel, args.num_classes, Block, Attention, Encoder='vgg')
elif args.model_type == 'ss_net3':
    model = SSNet3(args.in_channel, args.num_classes, Block, Attention, Encoder='vgg')
elif args.model_type == 'edr_net':
    model = EDRNet(args.in_channel, args.num_classes)
elif args.model_type == 'resunet_pp':
    model = ResUNetplusplus(args.in_channel, args.num_classes)
elif args.model_type == 'pga_net':
    model = PGANet(args.in_channel, args.num_classes)
elif args.model_type == 'seg_former':
    model = create_segformer_b0(args.num_classes)
elif args.model_type == 'a_net':
    model = ANet(args.num_classes)

model.load_state_dict(torch.load(args.checkpoint))
model.to(device)
model.eval()

benchmark = args.benchmark
root_path = args.data_root
batch_size = 1
train_ann = 'train.txt'
test_ann = 'test.txt'
padding_resize = False
if benchmark == 'KolektorSDD1':
    inputWH = (704, 256)
    mean = [186.5734, 186.5734, 186.5734]
    std = [23.3513, 23.3513, 23.3513]
elif benchmark == 'KolektorSDD2':
    inputWH = (646, 230)
    mean = [45.2396, 43.6739, 44.8110]
    std = [8.8128, 9.0113, 10.2784]
elif benchmark == 'CrackForest':
    inputWH = (320, 480)
    mean = [138.9284, 133.7921, 129.6794]
    std = [18.7216, 18.2210, 20.5493]
elif benchmark == 'RSDD1':
    inputWH = (1280, 160)
    mean = [101.7831, 101.7831, 101.7831]
    std = [73.3089, 73.3089, 73.3089]
elif benchmark == 'RSDD2':
    inputWH = (1248, 64)
    mean = [171.0793, 171.0793, 171.0793]
    std = [57.4940, 57.4940, 57.4940]
elif benchmark == 'Magnetic':
    inputWH = (448, 448)
    mean = [75.5886, 75.5886, 75.5886]
    std = [63.4075, 63.4075, 63.4075]
    padding_resize = True
else:
    raise Exception("请输入正确的benchmark")


# val transform template
valtrans = transform.Compose([
    transform.Resize(inputWH, padding_resize),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)
])

# val data
val_dataset = data.SemData(benchmark, split='val', data_root=root_path, data_list=test_ann, transform=valtrans)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size)

average_meter = AverageMeter(val_dataloader.dataset)
weight = torch.tensor([0.1, 0.9]).to(device)
scale = torch.tensor([1, 0.3, 0.2, 0., 0.]).to(device)
criterion = MultiScaleLoss(weight, scale)
save_path = os.path.join(args.save_path, args.benchmark)
os.makedirs(save_path, exist_ok=True)

for i, (x, y) in enumerate(val_dataloader):
    x = x.to(device)
    y = y.to(device)
    out = model(x)
    if isinstance(out, tuple):
        pred_mask = out[0].cpu().detach()
    else:
        pred_mask = out.cpu().detach()
    loss = criterion(out, y)
    pred_mask = torch.max(pred_mask, dim=1)[1]
    TP, TN, FP, FN = Evaluator.classify_prediction(pred_mask, y.cpu())
    average_meter.update(TP, TN, FP, FN, loss.cpu().detach().clone())
    miou, precision, recall, f1, dice = average_meter.evaluation()
    average_meter.reset()
    for j in range(len(mean)):
        x[:, j, :, :] = x[:, j, :, :] * std[j] + mean[j]
    x = torchvision.transforms.ToPILImage()(x.to(torch.uint8).squeeze())
    x.save(os.path.join(save_path, f'{i}_ori.png'))
    pred_mask = torchvision.transforms.ToPILImage()(pred_mask.to(torch.uint8)*255)
    true_mask = torchvision.transforms.ToPILImage()(y.to(torch.uint8)*255)
    pred_mask.save(os.path.join(save_path, f'{i}_{args.model_type}_{round(miou.item(), 3)}.png'))
    true_mask.save(os.path.join(save_path, f'{i}_label.png'))
