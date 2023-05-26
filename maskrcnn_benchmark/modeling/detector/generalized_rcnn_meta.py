# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.layers import GradientScalarLayer

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
# from ..da_heads.da_heads import build_da_heads
from ..da_heads.da_heads_meta import build_da_heads
import pdb
from torch.autograd import Variable
from torchvision import models
from maskrcnn_benchmark.modeling.backbone import resnet
import torch.nn.functional as F
from ..backbone.common import (conv1x1_block, conv3x3_block, conv7x7_block)
from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)
from maskrcnn_benchmark.modeling.backbone import resnet_meta
from collections import OrderedDict


def find_keys_with_chars(params, chars):
    if params is None:
        return None
    param = OrderedDict()
    for key in params.keys():
        if chars in key:
            child_key = key.split(chars)[1][1:]
            # print(chars)
            param.update({child_key:params[key]})
    return param

    # return [k for k in d.keys() if chars in k]

        
class ReptileModel(MetaModule):

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

class GeneralizedRCNN_meta(ReptileModel):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN_meta, self).__init__()
        ReptileModel.__init__(self)
        self.cfg = cfg.clone()

        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.da_heads = build_da_heads(cfg)
        self.backbone_mode = 'maml'
        self.head_mode ='maml'
        if self.backbone_mode =='maml':
            self.backbone = resnet_meta.resnet50(mode=self.backbone_mode)
        else:
            self.backbone = build_backbone(cfg)
        #backbone-----------------------------------------------------------------------------------------
        # self.backbone = models.vgg16_bn(pretrained=True)
        #rpn-----------------------------------------------------------------------------------------------
        # self.rpn_conv = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        # self.rpn_cls_logits = nn.Conv2d(1024, 15, kernel_size=1, stride=1)
        # self.rpn_bbox_pred = nn.Conv2d(1024, 15 * 4, kernel_size=1, stride=1)
        self.rpn_conv = conv3x3_block(1024, 1024,stride=1,use_bn=False,mode=self.head_mode)
        self.rpn_cls_logits = conv1x1_block(1024, 15,  stride=1,use_bn=False,activation=None,mode=self.head_mode)
        self.rpn_bbox_pred = conv1x1_block(1024, 15*4,  stride=1,use_bn=False,activation=None,mode=self.head_mode)
        # for l in [self.rpn_conv, self.rpn_cls_logits, self.rpn_bbox_pred]:
        #     torch.nn.init.normal_(l.weight, std=0.01)
        #     torch.nn.init.constant_(l.bias, 0)

        #roi head----------------------------------------------------------------------------------------------------------
        self.pooling = Pooler(output_size=(14, 14),scales=(1.0 / 16,),sampling_ratio=0,)
        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        self.roi_head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION
        )

        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

       
        if self.head_mode =='maml':
            num_inputs = 2048
            self.cls_score = MetaLinear(in_features=num_inputs,out_features=num_classes)
            num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
            self.bbox_pred =  MetaLinear(in_features=num_inputs,out_features=num_bbox_reg_classes * 4)
        else:
            num_inputs = 2048
            self.cls_score = nn.Linear(num_inputs, num_classes)
            num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
            self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        # nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        # nn.init.constant_(self.cls_score.bias, 0)

        # nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        # nn.init.constant_(self.bbox_pred.bias, 0)

        #da_heads---------------------------------------------------------------------------------


        self.grl_img = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        self.grl_img_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        #da img head
        in_channels_block = [256,512,1024]
        self.da_img_conv1_level_0 = conv1x1_block(in_channels_block[0], 512,  stride=1,use_bn=False,activation=None,mode=self.head_mode)
        self.da_img_conv2_level_0 = conv1x1_block(512, 1,  stride=1,use_bn=False,activation=None,mode=self.head_mode)

        self.da_img_conv1_level_1 = conv1x1_block(in_channels_block[1], 512,  stride=1,use_bn=False,activation=None,mode=self.head_mode)
        self.da_img_conv2_level_1 = conv1x1_block(512, 1,  stride=1,use_bn=False,activation=None,mode=self.head_mode)

        self.da_img_conv1_level_2 = conv1x1_block(in_channels_block[2], 512,  stride=1,use_bn=False,activation=None,mode=self.head_mode)
        self.da_img_conv2_level_2 = conv1x1_block(512, 1,  stride=1,use_bn=False,activation=None,mode=self.head_mode)

        # for module in [self.da_img_conv1_level_0, self.da_img_conv2_level_0,self.da_img_conv1_level_1, self.da_img_conv2_level_1,self.da_img_conv1_level_2, self.da_img_conv2_level_2]:
        #         # Caffe2 implementation uses XavierFill, which in fact
        #         # corresponds to kaiming_uniform_ in PyTorch
        #         torch.nn.init.normal_(module.weight, std=0.001)
        #         torch.nn.init.constant_(module.bias, 0)


        #da_ins_head
        if self.head_mode =='maml':
            self.fc1_da = MetaLinear(in_features=2048,out_features=1024)
            self.fc2_da = MetaLinear(in_features=1024,out_features=1024)
            self.fc3_da = MetaLinear(in_features=1024,out_features=1)
        else:
            self.fc1_da = nn.Linear(2048, 1024)
            self.fc2_da = nn.Linear(1024, 1024)
            self.fc3_da = nn.Linear(1024, 1)
        # for l in [self.fc1_da, self.fc2_da]:
        #     nn.init.normal_(l.weight, std=0.01)
        #     nn.init.constant_(l.bias, 0)
        # nn.init.normal_(self.fc3_da.weight, std=0.05)
        # nn.init.constant_(self.fc3_da.bias, 0)


       
    def forward(self, images,  targets=None,params=None):
                     
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        # for key, value in dict(self.backbone_meta.named_parameters()).items():
        #     pdb.set_trace()
        #backbone:
        if self.backbone_mode == 'maml':
            features = self.backbone(images.tensors,params=params)
        else:
            features = self.backbone(images.tensors)
        
        
        logits = []
        bbox_reg = []
        rpn_features = [features[-1]]
        
        for feature in rpn_features:
            if self.head_mode =='maml':
                t = self.rpn_conv(feature,params=find_keys_with_chars(params, 'rpn_conv'))
                logits.append(self.rpn_cls_logits(t,params=find_keys_with_chars(params, 'rpn_cls_logits')))
                bbox_reg.append(self.rpn_bbox_pred(t,params=find_keys_with_chars(params, 'rpn_bbox_pred')))
            else:
                t = self.rpn_conv(feature)
                logits.append(self.rpn_cls_logits(t))
                bbox_reg.append(self.rpn_bbox_pred(t))
        
        
        proposals, proposal_losses = self.rpn(images, features[-1], logits,bbox_reg, targets)
        if self.roi_heads:
            #rcnn:
            proposals = self.roi_heads.subsample_bbox(proposals,targets)
            pooled_features = self.pooling([features[-1]],proposals)
            #extrector
            if (self.head_mode =='maml') and (self.backbone_mode =='maml'):
                pooled_features = self.backbone.features.stage4(pooled_features,params=find_keys_with_chars(params, 'stage4'))
                # pooled_features = self.roi_head(pooled_features)
            else:
                # pooled_features = self.backbone.features.stage4(pooled_features)
                pooled_features = self.roi_head(pooled_features)
            #predict:
            pooled_features = self.avgpool(pooled_features)
            pooled_features = pooled_features.view(pooled_features.size(0), -1)
            # if params is not None:
            #     pdb.set_trace()
            if self.head_mode =='maml':
                cls_logit = self.cls_score(pooled_features,params=find_keys_with_chars(params, 'cls_score'))
                bbox_pred = self.bbox_pred(pooled_features,params=find_keys_with_chars(params, 'bbox_pred'))
            else:
                cls_logit = self.cls_score(pooled_features)
                bbox_pred = self.bbox_pred(pooled_features)

            x, result, detector_losses = self.roi_heads([features[-1]], proposals, targets,cls_logit,bbox_pred)
            if self.da_heads and self.training:
                 # da features:
                da_proposals = self.roi_heads.subsample_bbox_da(proposals,targets)
                da_pooled_features = self.pooling([features[-1]],da_proposals)
                da_ins_feas =self.backbone.features.stage4(da_pooled_features,params=find_keys_with_chars(params, 'stage4'))
        

                da_pred_feas = self.avgpool(da_ins_feas)
                da_pred_feas = da_pred_feas.view(da_pred_feas.size(0), -1)

                da_cls_logit = self.cls_score(da_pred_feas,params=find_keys_with_chars(params, 'cls_score'))
                da_bbox_pred = self.bbox_pred(da_pred_feas,params=find_keys_with_chars(params, 'bbox_pred'))

                da_ins_labels = self.roi_heads.da_ins_labels(da_cls_logit,da_bbox_pred)


                
                da_ins_feature = self.avgpool(da_ins_feas)
                da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

                img_grl_fea = [self.grl_img(fea) for fea in features]
                ins_grl_fea = self.grl_ins(da_ins_feature)
                img_grl_consist_fea = [self.grl_img_consist(fea) for fea in features]
                ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)
                #-----------------------------------da img------------------------------------------------------------
                da_img_features_0 = F.relu(self.da_img_conv1_level_0(img_grl_fea[0],params=find_keys_with_chars(params, 'da_img_conv1_level_0')))
                da_img_features_0 = self.da_img_conv2_level_0(da_img_features_0,params=find_keys_with_chars(params, 'da_img_conv2_level_0'))

                da_img_features_1 = F.relu(self.da_img_conv1_level_1(img_grl_fea[1],params=find_keys_with_chars(params, 'da_img_conv1_level_1')))
                da_img_features_1 = self.da_img_conv2_level_1(da_img_features_1,params=find_keys_with_chars(params, 'da_img_conv2_level_1'))

                da_img_features_2 = F.relu(self.da_img_conv1_level_2(img_grl_fea[2],params=find_keys_with_chars(params, 'da_img_conv1_level_2')))
                da_img_features_2 = self.da_img_conv2_level_2(da_img_features_2,params=find_keys_with_chars(params, 'da_img_conv2_level_2'))

                da_img_features = [da_img_features_0,da_img_features_1,da_img_features_2]
                 #-----------------------------------da ins------------------------------------------------------------
                da_ins_features = F.relu(self.fc1_da(ins_grl_fea,params=find_keys_with_chars(params, 'fc1_da')))
                
                da_ins_features = F.dropout(da_ins_features, p=0.5, training=self.training)

                da_ins_features = F.relu(self.fc2_da(da_ins_features,params=find_keys_with_chars(params, 'fc2_da')))
                da_ins_features = F.dropout(da_ins_features, p=0.5, training=self.training)

                #self.fc2_da(rpn_features[0].mean(3).mean(2),params=self.get_subdict(params, 'fc2_da'))

                da_ins_features = self.fc3_da(da_ins_features,params=find_keys_with_chars(params, 'fc3_da'))
                 #-----------------------------------da consis------------------------------------------------------------
                da_img_consis_0 = F.relu(self.da_img_conv1_level_0(img_grl_consist_fea[0],params=find_keys_with_chars(params, 'da_img_conv1_level_0')))
                da_img_consis_0 = self.da_img_conv2_level_0(da_img_consis_0,params=find_keys_with_chars(params, 'da_img_conv2_level_0'))

                da_img_consis_1 = F.relu(self.da_img_conv1_level_1(img_grl_consist_fea[1],params=find_keys_with_chars(params, 'da_img_conv1_level_1')))
                da_img_consis_1 = self.da_img_conv2_level_1(da_img_consis_1,params=find_keys_with_chars(params, 'da_img_conv2_level_1'))

                da_img_consis_2 = F.relu(self.da_img_conv1_level_2(img_grl_consist_fea[2], params=find_keys_with_chars(params, 'da_img_conv1_level_2')))
                da_img_consis_2 = self.da_img_conv2_level_2(da_img_consis_2,params=find_keys_with_chars(params, 'da_img_conv2_level_2'))

                da_img_consist_features = [da_img_consis_0,da_img_consis_1,da_img_consis_2]

                da_ins_consist_features = F.relu(self.fc1_da(ins_grl_consist_fea,params=find_keys_with_chars(params, 'fc1_da')))
                da_ins_consist_features = F.dropout(da_ins_consist_features, p=0.5, training=self.training)

                da_ins_consist_features = F.relu(self.fc2_da(da_ins_consist_features,params=find_keys_with_chars(params, 'fc2_da')))
                da_ins_consist_features = F.dropout(da_ins_consist_features, p=0.5, training=self.training)

                da_ins_consist_features = self.fc3_da(da_ins_consist_features, params=find_keys_with_chars(params, 'fc3_da'))

                da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
                da_ins_consist_features = da_ins_consist_features.sigmoid()
                #----------------------------------------da losses-----------------------------------------------------------
                da_losses = self.da_heads(da_proposals,da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, targets)
            else:
                da_losses={}

        # pdb.set_trace()
        
        # if self.roi_heads:
        #     x, result, detector_losses, da_ins_feas, da_ins_labels,da_proposals = self.roi_heads([features_4], proposals, targets)
        #     if self.da_heads and self.training:
        #         da_losses = self.da_heads(features, da_ins_feas, da_ins_labels,da_proposals,targets)
        #         # da_losses = self.da_heads([features[-1]], da_ins_feas, da_ins_labels, targets)
        #     else:
        #         da_losses={}



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
    

