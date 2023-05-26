# Hierarchical Meta Alignment for Cross-Domain Object Detection 

## Installation

Please follow the instruction in [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) to install and use Domain-Adaptive-Faster-RCNN-PyTorch.
## Example Usage
An example of Domain Adaptive Faster R-CNN with FPN adapting from **Cityscapes** dataset to **Foggy Cityscapes** dataset is provided:
1. Follow the example in [Detectron-DA-Faster-RCNN](https://github.com/krumo/Detectron-DA-Faster-RCNN) to download dataset and generate coco style annoation files
2. Symlink the path to the Cityscapes and Foggy Cityscapes dataset to `datasets/` as follows:
    ```bash
    # symlink the dataset
    cd ~/github/Domain-Adaptive-Faster-RCNN-PyTorch
    ln -s /<path_to_cityscapes_dataset>/ datasets/cityscapes
    ln -s /<path_to_foggy_cityscapes_dataset>/ datasets/foggy_cityscapes
    ```
3. Train the Domain Adaptive Faster R-CNN:
    ```
    python tools/train_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes.yaml"
    ```
4. Test the trained model:
    ```
    python tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes.yaml" MODEL.WEIGHT <path_to_store_weight>/model_final.pth
    ```
### Pretrained Model
[Pretrained model](https://polybox.ethz.ch/index.php/s/OgkNFJHVkEscTO0) with image+instance+consistency domain adaptation on Resnet-50 bakcbone for Cityscapes->Foggy Cityscapes task is provided. For those who might be interested, the corresponding training log could be checked at [here](logs/city2foggy_r50_consistency_log.txt). The following results are all tested with Resnet-50 backbone.


