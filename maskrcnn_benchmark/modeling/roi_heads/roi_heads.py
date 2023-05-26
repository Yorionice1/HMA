# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head

import pdb
class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box, da_ins_feas, da_ins_labels,da_proposals = self.box(features, proposals, targets)
        losses.update(loss_box)
       
        return x, detections, losses, da_ins_feas, da_ins_labels,da_proposals

    
class CombinedROIHeads_meta(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads_meta, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None,cls_pred=None,box_pred =None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box= self.box(features, proposals, targets,cls_pred,box_pred)
        losses.update(loss_box)
       
        return x, detections, losses
    def subsample_bbox(self,proposals,targets):
        proposals = self.box.subsample_bbox(proposals,targets)
        return proposals
    
    def subsample_bbox_da(self,proposals,targets):
       
        da_proposals = self.box.subsample_bbox_da(proposals, targets)
        return da_proposals
    def da_ins_labels(self,class_logits,box_regression):
        da_ins_labels = self.box.da_ins_labels(class_logits,box_regression)
        return da_ins_labels


def build_roi_heads(cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg)))

    # combine individual heads in a single module
    if roi_heads:
        if cfg.MODEL.META_ARCHITECTURE.split('_')[-1]=='meta':
            roi_heads = CombinedROIHeads_meta(cfg, roi_heads)
        else:
            roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
