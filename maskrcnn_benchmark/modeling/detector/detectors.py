# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .generalized_rcnn_meta import GeneralizedRCNN_meta


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,"GeneralizedRCNN_meta": GeneralizedRCNN_meta}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
