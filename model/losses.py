import torch
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_util import compute_tensor_iu
from einops import rearrange
from collections import defaultdict
import cv2

def get_iou_hook(values):
    return 'iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)  #single object. the iou between predicted mask and GT

def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i']+1)/(values['hide_iou/sec_u']+1)

iou_hooks_so = [
    get_iou_hook,
]

iou_hooks_mo = [
    get_iou_hook,
    get_sec_iou_hook,
]


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.bce = BootstrappedCE()
        self.sml1 = nn.SmoothL1Loss()

    def compute(self, data, it):

        losses = defaultdict(int)

        loss_seg = F.binary_cross_entropy_with_logits(data['logits'], data['cls_gt'].float())


        vision_flow_map_gt_fore = data['cls_gt'][:, :, 2::4, 2::4]
        vision_flow_map_gt_back = 1 - data['cls_gt'][:, :, 2::4, 2::4]
        vision_flow_map_gt = torch.cat((vision_flow_map_gt_fore, vision_flow_map_gt_back), dim=1)
        vision_flow_map_gt = rearrange(vision_flow_map_gt, "n c h w -> n (h w) c").float()
        vision_flow_map_gt = torch.matmul(vision_flow_map_gt, vision_flow_map_gt.permute(0, 2, 1))

        loss_vision_map = F.binary_cross_entropy_with_logits(data['vision_map'], data['cls_gt'][:, :, 2::4, 2::4].float())
        loss_flow_map = F.binary_cross_entropy_with_logits(data['flow_map'], data['cls_gt'][:, :, 2::4, 2::4].float())
        loss_vision_flow_map = F.binary_cross_entropy_with_logits(data["vision_flow_map"], vision_flow_map_gt)
        losses["vision_contrast_loss"] = loss_vision_map
        losses["flow_contrast_loss"] = loss_flow_map
        losses["vision_flow_contrast_loss"] = loss_vision_flow_map


        losses["total_loss"] = loss_seg + loss_vision_map + loss_flow_map + loss_vision_flow_map


        new_total_i, new_total_u = compute_tensor_iu(data["mask"] > 0.5, data['gt'] > 0.5)

        losses['hide_iou/i'] = new_total_i
        losses['hide_iou/u'] = new_total_u


        return losses
