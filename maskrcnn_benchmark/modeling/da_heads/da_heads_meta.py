# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import GradientScalarLayer

from .loss_3 import make_da_heads_loss_evaluator
import pdb
class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, is_retina):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()

        self.da_img_conv1_layers = []
        self.da_img_conv2_layers = []
        if is_retina:
            in_channels_block = [256,512,1024]
        else:
            in_channels_block = [128,256,512]
        for idx in range(3):
            conv1_block = "da_img_conv1_level{}".format(idx)
            conv2_block = "da_img_conv2_level{}".format(idx)
            conv1_block_module = nn.Conv2d(in_channels_block[idx], 512, kernel_size=1, stride=1)
            conv2_block_module = nn.Conv2d(512, 1, kernel_size=1, stride=1)
            for module in [conv1_block_module, conv2_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                torch.nn.init.normal_(module.weight, std=0.001)
                torch.nn.init.constant_(module.bias, 0)
            self.add_module(conv1_block, conv1_block_module)
            self.add_module(conv2_block, conv2_block_module)
            self.da_img_conv1_layers.append(conv1_block)
            self.da_img_conv2_layers.append(conv2_block)


    def forward(self, x):
        img_features = []
       
        
        for feature, conv1_block, conv2_block in zip(
                    x, self.da_img_conv1_layers, self.da_img_conv2_layers
            ):
                inner_lateral = getattr(self, conv1_block)(feature)
                last_inner = F.relu(inner_lateral)
                img_features.append(getattr(self, conv2_block)(last_inner))
        
        return img_features


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
       
        
            x = F.relu(self.fc1_da(x))
            x = F.dropout(x, p=0.5, training=self.training)

            x = F.relu(self.fc2_da(x))
            x = F.dropout(x, p=0.5, training=self.training)

            x = self.fc3_da(x)
       
            return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(DomainAdaptationModule, self).__init__()

        self.cfg = cfg.clone()
        
        self.loss_evaluator = make_da_heads_loss_evaluator(cfg)

        self.img_weight = cfg.MODEL.DA_HEADS.DA_IMG_LOSS_WEIGHT
        self.ins_weight = cfg.MODEL.DA_HEADS.DA_INS_LOSS_WEIGHT
        self.cst_weight = cfg.MODEL.DA_HEADS.DA_CST_LOSS_WEIGHT


    def forward(self, da_proposals,da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, targets):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        
        if self.training:
                da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
                    da_proposals,da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, targets
                )
                losses = {}
                if (self.img_weight > 0) and (da_img_loss != 0):
                    losses["loss_da_image"] = self.img_weight * da_img_loss
                if (self.ins_weight > 0) and (da_ins_loss != 0):
                    losses["loss_da_instance"] = self.ins_weight * da_ins_loss
                if (self.cst_weight > 0) and (da_consistency_loss != 0):
                    losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss
                return losses
        return {}

def build_da_heads(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return DomainAdaptationModule(cfg)
    return []
