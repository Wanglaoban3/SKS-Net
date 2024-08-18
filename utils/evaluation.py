r""" Evaluate mask prediction """
import torch


class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred_mask, gt_mask):
        # compute intersection and union of each episode in a batch
        TPs, TNs, FPs, FNs = [], [], [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            TP = ((_pred_mask == 1) & (_gt_mask == 1)).sum()  # 真正例
            TN = ((_pred_mask == 0) & (_gt_mask == 0)).sum()  # 真负例
            FP = ((_pred_mask == 1) & (_gt_mask == 0)).sum()  # 假正例
            FN = ((_pred_mask == 0) & (_gt_mask == 1)).sum()  # 假负例
            # _area_inter = torch.stack((TN, TP), dim=0)
            # _area_union = torch.stack((TN+FN+FP, TP+FN+FP), dim=0)
            TPs.append(TP)
            TNs.append(TN)
            FPs.append(FP)
            FNs.append(FN)
            # area_union.append(_area_union)
        # area_inter = torch.stack(area_inter).t()
        # area_union = torch.stack(area_union).t()
        TPs = torch.stack(TPs)
        TNs = torch.stack(TNs)
        FPs = torch.stack(FPs)
        FNs = torch.stack(FNs)
        return TPs, TNs, FPs, FNs
