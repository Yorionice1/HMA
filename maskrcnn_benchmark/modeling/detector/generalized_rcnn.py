# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
# from ..da_heads.da_heads import build_da_heads
from ..da_heads.da_heads_3 import build_da_heads
import pdb
from torch.autograd import Variable
class ReptileModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        # grad_list = []
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size())).cuda()
            if p.requires_grad:
                p.grad.data.zero_()  # not sure this is required
                p.grad.data.add_(p.data - target_p.data)
                # grad_list.append(p.data - target_p.data)
            else:
                p.grad.data.zero_()  # not sure this is required
                p.grad.data.add_(p.data - target_p.data)
        # return grad_list

    def point_grad_to_para(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        index = 0
        grad_list = []
        for p in self.parameters():
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size())).cuda()
            if p.requires_grad:
                p.grad.data.zero_()  # not sure this is required
                p.grad.data.add_(p.data - target[index])
                grad_list.append(p.data - target[index])
                index=index+1
        return grad_list
    def meta_align_grad(self, grad):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        index = 0
        for p in self.parameters():

            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size())).cuda()
            if p.requires_grad:
                p.grad.data.zero_()
                if grad[index] is None:
                   p.grad.data.add_(Variable(torch.zeros(p.size())).cuda())
                else:
                   p.grad.data.add_(grad[index])
                index+=1
                # p.grad.data.zero_()  # not sure this is required
                # p.grad.data.add_(p.data - target_p.data)
    def meta_backward(self, weight):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        index = 0
        for p in self.parameters():

            if p.requires_grad:
                p.data = weight[index]
                index+=1
                # p.grad.data.zero_()  # not sure this is required
                # p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters())

class GeneralizedRCNN(ReptileModel):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        ReptileModel.__init__(self)
        self.retina = cfg.MODEL.RETINANET_ON
        if self.retina:
            self.backbone,self.fpn = build_backbone(cfg)
            # self.backbone = build_backbone(cfg)
        else:
            self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.da_heads = build_da_heads(cfg)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        if self.retina:
            # features = self.backbone(images.tensors)
            # proposals, proposal_losses = self.rpn(images, features, targets)
            features_backbone = self.backbone(images.tensors)
            features = self.fpn(features_backbone)
            proposals, proposal_losses = self.rpn(images, features, targets)
            if self.da_heads and self.training:
                da_losses = self.da_heads(features, None, None,None,targets)
            else:
                da_losses={}
            x = features
            result = proposals
            detector_losses = {}

        else:
            features = self.backbone(images.tensors)
            proposals, proposal_losses = self.rpn(images, [features[-1]], targets)
        
            if self.roi_heads:
                x, result, detector_losses, da_ins_feas, da_ins_labels,da_proposals = self.roi_heads([features[-1]], proposals, targets)
                if self.da_heads and self.training:
                    da_losses = self.da_heads(features, da_ins_feas, da_ins_labels,da_proposals,targets)
                    # da_losses = self.da_heads([features[-1]], da_ins_feas, da_ins_labels, targets)
                else:
                    da_losses={}


            # else:
            #     # RPN-only models don't have roi_heads
            #     x = features
            #     result = proposals
            #     detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(da_losses)
            return losses

        return result

    def clone(self,net):
        clone = net
        clone.load_state_dict(self.state_dict())
        clone.cuda()
        return clone
