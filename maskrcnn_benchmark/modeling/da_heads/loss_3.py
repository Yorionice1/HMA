"""
This file contains specific functions for computing losses on the da_heads
file
"""

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import consistency_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.poolers import Pooler
from ..utils import cat
import pdb
class DALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        scales = [0.25,0.125,0.0625]
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_0 = Pooler(
            output_size=(resolution, resolution),
            scales=[scales[0]],
            sampling_ratio=sampling_ratio,
        )
        pooler_1 = Pooler(
            output_size=(resolution, resolution),
            scales=[scales[1]],
            sampling_ratio=sampling_ratio,
        )
        pooler_2 = Pooler(
            output_size=(resolution, resolution),
            scales=[scales[2]],
            sampling_ratio=sampling_ratio,
        )
        self.pooler_0 = pooler_0
        self.pooler_1 = pooler_1
        self.pooler_2 = pooler_2
        self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)
        self.retina = cfg.MODEL.RETINANET_ON
        
    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets:
            is_source = targets_per_image.get_field('is_source')
            mask_per_image = is_source.new_ones(1, dtype=torch.bool) if is_source.any() else is_source.new_zeros(1, dtype=torch.bool)
            masks.append(mask_per_image)
        return masks

    def __call__(self, proposals,da_img, da_ins, da_img_consist, da_ins_consist, da_ins_labels, targets):
        """
        Arguments:
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
            da_ins_loss (Tensor)
            da_consist_loss (Tensor)
        """

        masks = self.prepare_masks(targets)
        masks = torch.cat(masks, dim=0)
        da_img_flattened = []
        da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        _, _, H, W = da_img[0].shape
        up_sample = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)
        upsampled_loss = []
        for i, feat in enumerate(da_img):
            feat = da_img[i]
            feat = up_sample(feat)
            da_img_label_per_level = torch.zeros_like(feat, dtype=torch.float32)
            da_img_label_per_level[masks, :] = 1
            lv_loss = F.binary_cross_entropy_with_logits\
                (feat, da_img_label_per_level, reduction='none')
            upsampled_loss.append(lv_loss)
        da_img_loss = torch.stack(upsampled_loss)
        da_img_loss = da_img_loss.mean()
        
        if self.retina:
            return da_img_loss, 0, 0
        # da_img_loss, _ = torch.median(da_img_loss, dim=0)
        # da_img_loss, _ = torch.max(da_img_loss, dim=0)
        # da_img_loss, _ = torch.min(da_img_loss, dim=0)\
        # for da_img_per_level in da_img:
        #     N, A, H, W = da_img_per_level.shape
        #     da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
        #     da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
        #     da_img_label_per_level[masks, :] = 1

        #     da_img_per_level = da_img_per_level.reshape(N, -1)
        #     da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
            
        #     da_img_flattened.append(da_img_per_level)
        #     da_img_labels_flattened.append(da_img_label_per_level)
            
        # da_img_flattened = torch.cat(da_img_flattened, dim=0)
        # da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)
        
        # da_img_loss = F.binary_cross_entropy_with_logits(
        #     da_img_flattened, da_img_labels_flattened
        # )
        img_consist_fea_0 = self.avgpool(self.pooler_0([da_img_consist[0]], proposals))
        img_consist_fea_1 = self.avgpool(self.pooler_1([da_img_consist[1]], proposals))
        img_consist_fea_2 = self.avgpool(self.pooler_2([da_img_consist[2]], proposals))
        da_consist_loss_0 = F.l1_loss(img_consist_fea_0.view(img_consist_fea_0.size(0), -1), da_ins_consist)
        da_consist_loss_1 = F.l1_loss(img_consist_fea_1.view(img_consist_fea_1.size(0), -1), da_ins_consist)
        da_consist_loss_2 = F.l1_loss(img_consist_fea_2.view(img_consist_fea_2.size(0), -1), da_ins_consist)
        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        )
        da_consist_loss = (da_consist_loss_0+da_consist_loss_1+da_consist_loss_2)/3
        # da_ins_loss = 0
        # da_consist_loss = 0
        return da_img_loss, da_ins_loss, da_consist_loss
        # return 0.0,0.0,0.0

def make_da_heads_loss_evaluator(cfg):
    loss_evaluator = DALossComputation(cfg)
    return loss_evaluator
