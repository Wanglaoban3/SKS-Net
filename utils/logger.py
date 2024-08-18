r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        # self.class_ids_interest = dataset.class_ids
        # self.class_ids_interest = torch.tensor(self.class_ids_interest)

        # if self.benchmark == 'pascal':
        #     self.nclass = 20
        # elif self.benchmark == 'coco':
        #     self.nclass = 80
        # elif self.benchmark == 'fss':
        #     self.nclass = 1000
        # elif self.benchmark=='industrial':
        #     self.nclass=20
        # else:
        #     self.nclass = 1

        self.TP_buf = torch.zeros([1]).float()
        self.TN_buf = torch.zeros([1]).float()
        self.FP_buf = torch.zeros([1]).float()
        self.FN_buf = torch.zeros([1]).float()
        self.loss_buf = []

    def reset(self):
        self.TP_buf = torch.zeros([1]).float()
        self.TN_buf = torch.zeros([1]).float()
        self.FP_buf = torch.zeros([1]).float()
        self.FN_buf = torch.zeros([1]).float()
        self.loss_buf = []

    def update(self, TP, TN, FP, FN, loss):
        # self.intersection_buf.index_add_(1, class_id, inter_b.float())
        # self.union_buf.index_add_(1, class_id, union_b.float())
        # self.TP_buf.index_add_(1, torch.tensor([0]), TP.float())
        # self.TN_buf.index_add_(1, torch.tensor([0]), TN.float())
        # self.FP_buf.index_add_(1, torch.tensor([0]), FP.float())
        # self.FN_buf.index_add_(1, torch.tensor([0]), FN.float())
        self.TP_buf += TP.sum()
        self.TN_buf += TN.sum()
        self.FP_buf += FP.sum()
        self.FN_buf += FN.sum()
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    # def compute_iou(self):
    #     eps = 1e-8
    #     iou = (self.intersection_buf + eps) / (self.union_buf + eps) * 100
    #     fb_iou = iou.mean()
    #     return iou[1], fb_iou

    def evaluation(self):
        eps = 1e-8
        miou = (self.TP_buf + eps) / (self.FP_buf + self.TP_buf + self.FN_buf + eps)
        precision = (self.TP_buf + eps) / (self.TP_buf + self.FP_buf + eps)
        recall = (self.TP_buf + eps) / (self.TP_buf + self.FN_buf + eps)
        f1 = (2 * precision * recall + eps) / (precision + recall + eps)
        dice = (2 * self.TP_buf) / (2 * self.TP_buf + self.FN_buf + self.FP_buf)
        return miou*100, precision*100, recall*100, f1*100, dice * 100

    def write_result(self, split, epoch):
        iou, precision, recall, f1, dice = self.evaluation()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'mIoU: %5.2f   ' % iou
        msg += 'precision: %5.2f   ' % precision
        msg += 'recall: %5.2f   ' % recall
        msg += 'f1: %5.2f   ' % f1
        msg += 'dice: %5.2f   ' % dice
        # msg += 'FB-IoU: %5.2f   ' % fb_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            # iou, fb_iou = self.compute_iou()
            iou, precision, recall, f1, dice = self.evaluation()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f   ' % iou
            msg += 'precision: %5.2f   ' % precision
            msg += 'recall: %5.2f   ' % recall
            msg += 'f1: %5.2f   ' % f1
            msg += 'dice: %5.2f   ' % dice
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', args.benchmark, f'{args.model_type}_{args.block}_{args.attention}{logpath}.log')
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Few-shot Seg. with IndustrialNet ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in HSNet
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))

