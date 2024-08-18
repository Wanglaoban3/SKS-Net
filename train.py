'''
train scipts

author: zacario li
date: 2020-10-09
'''
import os
import torch.optim
import torch.utils.data
from loss import MultiScaleLoss
from dataset import data
from utils import transform
import argparse
from utils.metrics import TotalDiceIou
from utils.logger import Logger
import time


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr

def prepare_dataset(rootpath, batch_size, benchmark):
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
    elif benchmark == 'NEUSEG':
        inputWH = (224, 224)
        mean = [110.313, 110.313, 110.313]
        std = [51.9945, 51.9945, 51.9945]
    else:
        raise Exception("请输入正确的benchmark")

    # train transform template
    trans = transform.Compose([
        transform.Resize(inputWH, padding_resize),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    # val transform template
    valtrans = transform.Compose([
        transform.Resize(inputWH, padding_resize),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    # training data
    train_dataset = data.SemData(benchmark, split='train', data_root=rootpath, data_list=train_ann, transform=trans)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    pin_memory=True,
                                                    drop_last=False)

    # val data
    val_dataset = data.SemData(benchmark, split='val', data_root=rootpath, data_list=test_ann, transform=valtrans)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    pin_memory=True)
    
    return train_dataloader, val_dataloader


def sub_sn_train(model, training, optimizer, criterion, dataloader, currentepoch, maxIter, base_lr):
    if training:
        model.train()
        # scheduler.step()
    else:
        model.eval()
    num_classes = 2
    if dataloader.dataset.benchmark == 'NEUSEG':
        num_classes = 4
    average_meter = TotalDiceIou(num_classes)
    total_loss = 0

    for i, (x, y) in enumerate(dataloader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        # loss_size = out.shape[2:]
        # y = F.interpolate(y.unsqueeze(1).float(), loss_size, mode='nearest').squeeze(1).long()
        # out[1] = nn.Upsample(scale_factor=8, mode='bilinear')(out[1])
        loss = criterion(out, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 调整学习率
            curiter = currentepoch * len(dataloader) + i + 1
            newlr = poly_learning_rate(base_lr, curiter, maxIter)
            optimizer.param_groups[0]['lr'] = newlr
        total_loss += loss.item()
        # 3. Evaluate prediction
        if isinstance(out, tuple):
            pred_mask = out[0].cpu().detach()
        else:
            pred_mask = out.cpu().detach()
        pred_mask = torch.max(pred_mask, dim=1)[1]
        average_meter.update(pred_mask.numpy(), y.squeeze().cpu().numpy())

    # Write evaluation results
    avg_loss = total_loss / len(dataloader)
    miou = average_meter.get_mIoU()
    dice = average_meter.get_mdice()
    pix_acc = average_meter.get_pixes_accuracy()
    Mode = 'Train' if training else 'Validation'
    print(f"Mode: {Mode}\tepoch: {currentepoch}\tloss: {round(avg_loss, 5)}\tmiou: {round(miou, 4)}\t"
          f"dice: {round(dice, 4)}\tpix acc: {round(pix_acc, 5)}")
    return avg_loss, miou, dice, pix_acc


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='ss_net', choices=['dlasdd', 'seg_net', 'ss_net', 'ss_net2', 'ss_net3', 'deeplabv3', 'psp_net', 'u_net', 'edr_net', 'resunet_pp', 'pga_net', 'seg_former', 'a_net'])
    parser.add_argument('--block', type=str, default='basic')
    parser.add_argument('--attention', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--benchmark', type=str, default='KolektorSDD1', choices=['KolektorSDD1', 'KolektorSDD2', 'CrackForest', 'RSDD1', 'Magnetic', 'RSDD2', 'NEUSEG'])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--save_path', type=str, default='save')
    parser.add_argument('--data_root', type=str, default='C:/wrd/KolektorSDD')
    parser.add_argument('--batch_size', type=int, default=6)
    args = parser.parse_args()

    # global config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Logger.initialize(args, training=True)


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
    else:
        raise AttributeError("输入正确的模型名")
    initialize_weights(model)
    Logger.log_params(model)

    # test code
    x = torch.rand(2, 3, 704, 256)
    model(x)

    # load weight
    epoch_start = 0
    if args.checkpoint != '':
        wt = torch.load(args.checkpoint)
        model.load_state_dict(wt)
    if epoch_start >= args.epoch:
        print(f"current ckpt's epoch[{epoch_start}] is smaller and equal than global_epoch[{args.global_epoch}]")
        print(f'stop training...')

    model.to(device)
    # build dataset
    train_loader, val_loader = prepare_dataset(args.data_root, args.batch_size, args.benchmark)

    # criterion = diceloss.DiceLoss()
    if args.benchmark == 'KolektorSDD1':
        weight = torch.tensor([0.01, 0.99]).to(device)
    elif args.benchmark == 'NEUSEG':
        weight = torch.tensor([0.25, 0.25, 0.25, 0.25]).to(device)
    else:
        weight = torch.tensor([0.1, 0.9]).to(device)
    scale = torch.tensor([1, 0.3, 0.2, 0., 0.]).to(device)
    criterion = MultiScaleLoss(weight, scale)
    #criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.01, 0.99]).to(device))
    '''
    wt = torch.tensor([0.1, 10000000])
    wt = wt.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=wt)
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    last_best_name = ''
    maxIter = args.epoch * len(train_loader)
    # lambda_lr = lambda epoch: 0.1 if epoch < 30 else (0.01 if epoch < 50 else 0.001)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    # starting traing
    for epoch in range(epoch_start, args.epoch):
        trn_loss, trn_miou, trn_dice, trn_pa = sub_sn_train(model, True, optimizer, criterion, train_loader, epoch, maxIter, args.lr)
        with torch.no_grad():
            val_loss, val_miou, val_dice, val_pa = sub_sn_train(model, False, optimizer, criterion, val_loader, epoch, maxIter, args.lr)
        # Save the best model
        if val_miou > best_val_miou:
            if epoch > 0:
                # 删除上一次保存的最佳权重
                os.remove(last_best_name)
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)
            filename = f'{args.save_path}/{args.benchmark}_{args.model_type}_{args.block}_{args.attention}_epoch{str(epoch)}' \
                       f'_miou{str(round(best_val_miou, 4))}' \
                       f'_pixacc{str(round(val_pa, 4))}' \
                       f'_dice{str(round(val_dice, 4))}.pth'
            last_best_name = filename
            torch.save(model.state_dict(), filename)
        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/pixacc', {'trn_pixacc': trn_pa, 'val_pixacc': val_pa}, epoch)
        Logger.tbd_writer.add_scalars('data/dice', {'trn_dice': trn_dice, 'val_dice': val_dice}, epoch)
        Logger.tbd_writer.add_scalars('data/lr', {'lr': optimizer.param_groups[0]['lr']}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
