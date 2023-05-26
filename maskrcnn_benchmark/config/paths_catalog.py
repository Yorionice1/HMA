# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/test/data/datasets"
    DATASETS = {
         "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },
        "foggy_cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "foggy_cityscapes/images",
            "ann_file": "foggy_cityscapes/annotations/foggy_instancesonly_filtered_gtFine_train.json"
        },
        "foggy_cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "foggy_cityscapes/images",
            "ann_file": "foggy_cityscapes/annotations/foggy_instancesonly_filtered_gtFine_val.json"
        },
        "kitti_car_only_train_cocostyle": {
            "img_dir": "kitti/training/image_2",
            "ann_file": "kitti/annotations/caronly_training.json"
        },
        # "kitti_car_only_val": {
        #     "img_dir": "cityscapes/training/image_2",
        #     "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        # },
        "cityscapes_car_only_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/caronly_filtered_gtFine_train.json"
        },
        "cityscapes_car_only_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/caronly_filtered_gtFine_val.json"
        },
        "sim10k_train_cocostyle": {
            "img_dir": "sim10k/JPEGImages",
            "ann_file": "sim10k/annotations/car_instances.json"
        },
        # "cityscapes_car_only_val": {
        #     "img_dir": "cityscapes/images",
        #     "ann_file": "cityscapes/annotations/caronly_filtered_gtFine_val.json"
        # },
        "city_kitti_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/city2kitti_gtFine_train.json"
        },
        "city_kitti_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/city2kitti_gtFine_val.json"
        },
        'kitti_city_cocostyle': {
            "img_dir": 'kitti/training/image_2',
            "ann_file": 'kitti/annotations/kitti2city_training.json'
        },

        "city_bdd_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/city2bdd_gtFine_val.json"
        },
        "city_bdd_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/city2bdd_gtFine_train.json"
        },
        "city_bdd_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/city2bdd_gtFine_test.json"
        },

        "bdd_city_train_cocostyle": {
            "img_dir": "BDD/bdd100k/images/100k/train",
            "ann_file": "BDD/bdd100k/bdd100k_labels_7_cate_daytime_train.json"
        },
        "bdd_city_val_cocostyle": {
            "img_dir": "BDD/bdd100k/images/100k/val",
            "ann_file": "BDD/bdd100k/bdd100k_labels_7_cate_daytime_val.json"
        },
        "voc_train_cocostyle": {
            "img_dir": "VOCdevkit/VOC2012/JPEGImages",
            "ann_file": "VOCdevkit/VOC2012/train_watercolor.json"
        },
        "watercolor_train_cocostyle": {
            "img_dir": "watercolor/JPEGImages",
            "ann_file": "watercolor/train.json"
        },
        # "watercolor_train": {
        #     "img_dir": "watercolor/JPEGImages",
        #     "ann_file": "watercolor/train.json"
        # },
        "watercolor_val_cocostyle": {
            "img_dir": "watercolor/JPEGImages",
            "ann_file": "watercolor/test.json"
        },

        "voc_2012_train": {
            "data_dir": "VOCdevkit/VOC2012",
            "split": "trainval"
        },

        # "voc_2012_6_cate_train": {
        #     "data_dir": "VOCdevkit/VOC2012",
        #     "split": "bicycle_trainval,bird_trainval,car_trainval,cat_trainval,dog_trainval,person_trainval"
        # },

        "voc_2007_train": {
            "data_dir": "VOCdevkit/VOC2007",
            "split": "trainval"
        },

        # "voc_2007_6_cate_train": {
        #     "data_dir": "VOCdevkit/VOC2007",
        #     "split": "bicycle_trainval,bird_trainval,car_trainval,cat_trainval,dog_trainval,person_trainval"
        # },

        "watercolor_train_voc": {
            "data_dir": "watercolor",
            "split": "train"
        },
        "watercolor_val_voc": {
            "data_dir": "watercolor",
            "split": "test"
        },

        "clipart_train_voc": {
            "data_dir": "clipart",
            "split": "train"
        },

        "clipart_test_voc": {
            "data_dir": "clipart",
            "split": "test"
        },




        
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url