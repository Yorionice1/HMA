# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from torch.autograd import grad
import os
import pdb
import copy


def test(cfg, model, distributed = False):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        save_path = './experiments/city2foggy/metares50_MDAF/'
        # save_path = './experiments/VOC2watercolor/daf_baseline/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save(save_path+"model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter-1:
            checkpointer.save(save_path+"model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def do_da_train(
    model,
    source_data_loader,
    target_data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        images = (source_images+target_images).to(device)
        targets = [target.to(device) for target in list(source_targets+target_targets)]
    
        
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        # save_path = './experiments/VOC2clipart/3dc_consist_baseline/'
        save_path = './experiments/city2foggy/metares50_MDAF/'
        # save_path = './experiments/VOC2watercolor/daf_baseline/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save(save_path+"model_{:07d}".format(iteration), **arguments)
            # test(cfg,model)
        if iteration == max_iter-1:
            checkpointer.save(save_path+"model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time /(max_iter)
        )
    )


def do_da_train_draw(
    model,
    source_data_loader,
    target_data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    rpnc_rpnr_sum = 0

    rpnc_rcnnc_sum = 0

    rpnc_rcnnr_sum = 0

    rpnc_da_sum = 0

    rpnr_rcnnc_sum = 0

    rpnr_rcnnr_sum = 0

    rpnr_da_sum = 0
        
    rcnnc_rcnnr_sum = 0

    rcnnc_da_sum = 0

    rcnnr_da_sum = 0


    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        images = (source_images+target_images).to(device)
        targets = [target.to(device) for target in list(source_targets+target_targets)]
    
        
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)




        rpn_loss_cls = loss_dict['loss_objectness']
        rpn_loss_box = loss_dict['loss_rpn_box_reg']
        RCNN_loss_cls = loss_dict['loss_classifier']
        RCNN_loss_bbox = loss_dict['loss_box_reg']
        



        loss_det = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

        loss_da = loss_dict['loss_da_image'] + loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']

        sim = 0
        count = 0
        optimizer.zero_grad()
        loss_det.mean().backward(retain_graph=True)
        det_grad={}
        for key, value in model.named_parameters():
                if value.requires_grad:
                  temp = {key:copy.deepcopy(value.grad)}
                  det_grad.update(temp)   
                else:
                  pass

        optimizer.zero_grad()
        loss_da.mean().backward(retain_graph=True)
        da_grad={}
        for key, value in model.named_parameters():
                if value.requires_grad:
                  temp = {key:copy.deepcopy(value.grad)}
                  da_grad.update(temp)   
                else:
                  pass

        optimizer.zero_grad()
        rpn_loss_cls.mean().backward(retain_graph=True)
        rpn_cls_grad={}
        for key, value in model.named_parameters():
                if value.requires_grad:
                  temp = {key:copy.deepcopy(value.grad)}
                  rpn_cls_grad.update(temp)   
                else:
                  pass
        optimizer.zero_grad()
        rpn_loss_box.mean().backward(retain_graph=True)
        rpn_box_grad={}
        for key, value in model.named_parameters():
                if value.requires_grad:
                  temp = {key:copy.deepcopy(value.grad)}
                  rpn_box_grad.update(temp)   
                else:
                  pass
        optimizer.zero_grad()
        RCNN_loss_cls.mean().backward(retain_graph=True)
        RCNN_cls_grad={}
        for key, value in model.named_parameters():
                if value.requires_grad:
                  temp = {key:copy.deepcopy(value.grad)}
                  RCNN_cls_grad.update(temp)   
                else:
                  pass
        optimizer.zero_grad()
        RCNN_loss_bbox.mean().backward(retain_graph=True)
        RCNN_box_grad={}
        for key, value in model.named_parameters():
                if value.requires_grad:
                  temp = {key:copy.deepcopy(value.grad)}
                  RCNN_box_grad.update(temp)   
                else:
                  pass

        rpnc_rpnr_sim = 0
        rpnc_rpnr_count = 1

        rpnc_rcnnc_sim = 0
        rpnc_rcnnc_count = 1

        rpnc_rcnnr_sim = 0
        rpnc_rcnnr_count = 1

        rpnc_da_sim = 0
        rpnc_da_count = 1

        rpnr_rcnnc_sim = 0
        rpnr_rcnnc_count = 1

        rpnr_rcnnr_sim = 0
        rpnr_rcnnr_count = 1

        rpnr_da_sim = 0
        rpnr_da_count = 1
        
        rcnnc_rcnnr_sim = 0
        rcnnc_rcnnr_count = 1

        rcnnc_da_sim = 0
        rcnnc_da_count = 1

        rcnnr_da_sim = 0
        rcnnr_da_count = 1



        
        
       
        for key in rpn_cls_grad.keys():
            if rpn_cls_grad[key] is not None:
                rpn_grad_1 = rpn_cls_grad[key].reshape(-1)
                rpn_grad_2 = rpn_box_grad[key].reshape(-1)
                RCNN_grad_1 = RCNN_cls_grad[key].reshape(-1)
                RCNN_grad_2 = RCNN_box_grad[key].reshape(-1)
                da_grad_1 = da_grad[key].reshape(-1)
                #rpn_cls_reg:
                if rpn_grad_1.mean()!=0 and rpn_grad_2.mean()!=0:
                    sim = torch.nn.functional.cosine_similarity(rpn_grad_1, rpn_grad_2,dim = 0)
                    rpnc_rpnr_sim = rpnc_rpnr_sim + sim
                    rpnc_rpnr_count =rpnc_rpnr_count +1
                
                if rpn_grad_1.mean()!=0 and RCNN_grad_1.mean()!=0:
                    sim = torch.nn.functional.cosine_similarity(rpn_grad_1, RCNN_grad_1,dim = 0)
                    rpnc_rcnnc_sim = rpnc_rcnnc_sim + sim
                    rpnc_rcnnc_count = rpnc_rcnnc_count +1

                    
                if rpn_grad_1.mean()!=0 and RCNN_grad_2.mean()!=0:
                    sim = torch.nn.functional.cosine_similarity(rpn_grad_1, RCNN_grad_2,dim = 0)
                    rpnc_rcnnr_sim = rpnc_rcnnr_sim + sim
                    rpnc_rcnnr_count = rpnc_rcnnr_count +1


                if rpn_grad_1.mean()!=0 and da_grad_1.mean()!=0:
                    sim = torch.nn.functional.cosine_similarity(rpn_grad_1, da_grad_1,dim = 0)
                    rpnc_da_sim = rpnc_da_sim + sim
                    rpnc_da_count = rpnc_da_count +1

                if rpn_grad_2.mean()!=0 and RCNN_grad_1.mean()!=0:
                    sim = torch.nn.functional.cosine_similarity(rpn_grad_2, RCNN_grad_1,dim = 0)
                    rpnr_rcnnc_sim = rpnr_rcnnc_sim + sim
                    rpnr_rcnnc_count = rpnr_rcnnc_count +1

                if rpn_grad_2.mean()!=0 and RCNN_grad_2.mean()!=0:
                    sim = torch.nn.functional.cosine_similarity(rpn_grad_2, RCNN_grad_2,dim = 0)
                    rpnr_rcnnr_sim = rpnr_rcnnr_sim + sim
                    rpnr_rcnnr_count = rpnr_rcnnr_count +1

                if rpn_grad_2.mean()!=0 and da_grad_1.mean()!=0:
                    sim = torch.nn.functional.cosine_similarity(rpn_grad_2, da_grad_1,dim = 0)
                    rpnr_da_sim = rpnr_da_sim + sim
                    rpnr_da_count = rpnr_da_count +1

                if RCNN_grad_1.mean()!=0 and RCNN_grad_2.mean()!=0:
                    sim = torch.nn.functional.cosine_similarity(RCNN_grad_1, RCNN_grad_2,dim = 0)
                    rcnnc_rcnnr_sim = rcnnc_rcnnr_sim + sim
                    rcnnc_rcnnr_count = rcnnc_rcnnr_count +1

                if RCNN_grad_1.mean()!=0 and da_grad_1.mean()!=0:
                    sim = torch.nn.functional.cosine_similarity(RCNN_grad_1, da_grad_1,dim = 0)
                    rcnnc_da_sim = rcnnc_da_sim + sim
                    rcnnc_da_count = rcnnc_da_count +1
                if RCNN_grad_2.mean()!=0 and da_grad_1.mean()!=0:
                    sim = torch.nn.functional.cosine_similarity(RCNN_grad_2, da_grad_1,dim = 0)
                    rcnnr_da_sim = rcnnr_da_sim + sim
                    rcnnr_da_count = rcnnr_da_count +1

        rpnc_rpnr_mean = rpnc_rpnr_sim/rpnc_rpnr_count

        rpnc_rcnnc_mean = rpnc_rcnnc_sim/rpnc_rcnnc_count

        rpnc_rcnnr_mean = rpnc_rcnnr_sim/rpnc_rcnnr_count

        rpnc_da_mean = rpnc_da_sim/rpnc_da_count

        rpnr_rcnnc_mean = rpnr_rcnnc_sim/rpnr_rcnnc_count

        rpnr_rcnnr_mean = rpnr_rcnnr_sim/rpnr_rcnnr_count

        rpnr_da_mean = rpnr_da_sim/rpnr_da_count
        
        rcnnc_rcnnr_mean = rcnnc_rcnnr_sim/rcnnc_rcnnr_count

        rcnnc_da_mean = rcnnc_da_sim/rcnnc_da_count

        rcnnr_da_mean = rcnnr_da_sim/rcnnr_da_count



        rpnc_rpnr_sum = rpnc_rpnr_sum + rpnc_rpnr_mean

        rpnc_rcnnc_sum = rpnc_rcnnc_sum + rpnc_rcnnc_mean

        rpnc_rcnnr_sum = rpnc_rcnnr_sum + rpnc_rcnnr_mean

        rpnc_da_sum = rpnc_da_sum  + rpnc_da_mean

        rpnr_rcnnc_sum = rpnr_rcnnc_sum + rpnr_rcnnc_mean

        rpnr_rcnnr_sum = rpnr_rcnnr_sum + rpnr_rcnnr_mean

        rpnr_da_sum = rpnr_da_sum + rpnr_da_mean
            
        rcnnc_rcnnr_sum = rcnnc_rcnnr_sum + rcnnc_rcnnr_mean

        rcnnc_da_sum = rcnnc_da_sum + rcnnc_da_mean

        rcnnr_da_sum = rcnnr_da_sum + rcnnr_da_mean
        
        # rpnc_rpnr_sim,rpnc_rcnnc_sim,rpnc_rcnnr_sim,rpnc_da_sim,rpnr_rcnnc_sim,rpnr_rcnnr_sim,rpnr_da_sim,rcnnc_rcnnr_sim,rcnnc_da_sim,rcnnr_da_sim
       
        
        # print("[rpnc_rpnr_count: %.4f][rpnc_rcnnc_count: %.4f][rpnc_rcnnr_count: %.4f] [rpnc_da_count: %.4f] [rpnr_rcnnc_count: %.4f] [rpnr_rcnnr_count: %.4f] [rpnr_da_count: %.4f] [rcnnc_rcnnr_count: %.4f] [rcnnc_da_count: %.4f] [rcnnr_da_count:%.4f]" % (rpnc_rpnr_mean,rpnc_rcnnc_mean,rpnc_rcnnr_mean,rpnc_da_mean,rpnr_rcnnc_mean,rpnr_rcnnr_mean,rpnr_da_mean,rcnnc_rcnnr_mean,rcnnc_da_mean,rcnnr_da_mean))

        if iteration%5000 ==0:
            print('-----------------------------------------------------------------------------------------------------------------------------------')
            print("[rpnc_rpnr_count: %.4f][rpnc_rcnnc_count: %.4f][rpnc_rcnnr_count: %.4f] [rpnc_da_count: %.4f] [rpnr_rcnnc_count: %.4f] [rpnr_rcnnr_count: %.4f] [rpnr_da_count: %.4f] [rcnnc_rcnnr_count: %.4f] [rcnnc_da_count: %.4f] [rcnnr_da_count:%.4f]" % (rpnc_rpnr_sum/iteration,rpnc_rcnnc_sum/iteration,rpnc_rcnnr_sum/iteration,rpnc_da_sum/iteration,rpnr_rcnnc_sum/iteration,rpnr_rcnnr_sum/iteration,rpnr_da_sum/iteration,rcnnc_rcnnr_sum/iteration,rcnnc_da_sum/iteration,rcnnr_da_sum/iteration))

        losses.backward()
        optimizer.step()
        
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # if iteration % 20 == 0 or iteration == max_iter:
        #     logger.info(
        #         meters.delimiter.join(
        #             [
        #                 "eta: {eta}",
        #                 "iter: {iter}",
        #                 "{meters}",
        #                 "lr: {lr:.6f}",
        #                 "max mem: {memory:.0f}",
        #             ]
        #         ).format(
        #             eta=eta_string,
        #             iter=iteration,
        #             meters=str(meters),
        #             lr=optimizer.param_groups[0]["lr"],
        #             memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
        #         )
        #     )

        # save_path = './experiments/VOC2clipart/3dc_consist_baseline/'
        save_path = './experiments/city2foggy/metares50_MDAF/'
        # save_path = './experiments/VOC2watercolor/daf_baseline/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save(save_path+"model_{:07d}".format(iteration), **arguments)
            # test(cfg,model)
        if iteration == max_iter-1:
            checkpointer.save(save_path+"model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time /(max_iter)
        )
    )

def do_da_train_meta_old(
            model_1,
            model_2,
            meta_model,
            source_data_loader,
            target_data_loader,
            optimizer_1,
            optimizer_2,
            meta_optimizer,
            scheduler_1,
            scheduler_2,
            meta_scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            cfg,
        ):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model_1.train()
    model_2.train()
    meta_model.train()
    start_training_time = time.time()
    end = time.time()
    key = 0
    inner_lr = cfg.SOLVER.INNER_LR
    inner_inner_lr = cfg.SOLVER.INNER2_LR
    base_lr = cfg.SOLVER.BASE_LR
    model_1 = meta_model.clone(model_1)
    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration
        key = key+1
        images = (source_images+target_images).to(device)
        targets = [target.to(device) for target in list(source_targets+target_targets)]
        model_2 = model_1.clone(model_2)
        loss_dict = model_2(images, targets)
        if key == 1:
            losses = loss_dict['loss_objectness']
        elif key == 2:
            losses = loss_dict['loss_rpn_box_reg']
        elif key == 3:
            losses = loss_dict['loss_classifier']
        else:
            losses = loss_dict['loss_box_reg']
        #inner inner backward:

        net2_paras = list(filter(lambda p: p.requires_grad,model_2.parameters()))
        net2_grad = grad(losses, filter(lambda p: p.requires_grad,model_2.parameters()), create_graph=True, allow_unused=True)
        net2_paras = list(map(lambda aa: aa[1] - inner_inner_lr * aa[0] if aa[0] is not None else aa[1], zip(net2_grad, net2_paras)))
        model_2.meta_backward(net2_paras)
        loss_dict = model_2(images, targets)
        if key == 1:
            losses = loss_dict['loss_objectness'] + loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        elif key == 2:
            losses = loss_dict['loss_rpn_box_reg']+ loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        elif key == 3:
            losses = loss_dict['loss_classifier']+ loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        else:
            losses = loss_dict['loss_box_reg']+ loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        # losses = sum(loss for loss in loss_dict.values())
        #inner backward:
        net1_paras = list(filter(lambda p: p.requires_grad,model_1.parameters()))
        net2_grad = grad(losses, filter(lambda p: p.requires_grad,model_2.parameters()), create_graph=True, allow_unused=True)
        net1_paras = list(map(lambda aa: aa[1] - inner_lr * aa[0] if aa[0] is not None else aa[1], zip(net2_grad, net1_paras)))
        model_1.meta_backward(net1_paras)

        #reptile backward:
        if key ==4:
            meta_model.point_grad_to(model_1)
            meta_optimizer.step()
            model_1 = meta_model.clone(model_1)
            key = 0


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        
        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()
        
        meta_scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 2000 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=meta_optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        save_path = './experiments/reptile_MAML_42.5/inner_inner_lr_{}_inner_lr_{}_lr_{}/'.format(inner_inner_lr,inner_lr,base_lr)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save(save_path+"model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter-1:
            checkpointer.save(save_path+"model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def do_da_train_meta(
            model_1,
            model_2,
            meta_model,
            source_data_loader,
            target_data_loader,
            optimizer_1,
            optimizer_2,
            meta_optimizer,
            scheduler_1,
            scheduler_2,
            meta_scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            cfg,
        ):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model_1.train()
    model_2.train()
    meta_model.train()
    start_training_time = time.time()
    end = time.time()
    inner_lr = cfg.SOLVER.INNER_LR
    inner_inner_lr = cfg.SOLVER.INNER2_LR
    base_lr = cfg.SOLVER.BASE_LR
    model_1 = meta_model.clone(model_1)
    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        #data:
        data_time = time.time() - end
        arguments["iteration"] = iteration
        images = (source_images+target_images).to(device)
        targets = [target.to(device) for target in list(source_targets+target_targets)]
        model_1 = meta_model.clone(model_1)
        # rpn classification:
        model_2 = model_1.clone(model_2)
        loss_dict = model_2(images, targets) 
        losses = loss_dict['loss_objectness']
        optimizer_2.zero_grad()
        losses.backward()
        optimizer_2.step()
        
        loss_dict = model_2(images, targets)
        losses = loss_dict['loss_objectness'] + loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        optimizer_2.zero_grad()
        losses.backward()
        optimizer_2.step()

        #inner reptile update:
        optimizer_1.zero_grad()
        model_1.point_grad_to(model_2)
        optimizer_1.step()

        # rpn regression:
        model_2 = model_1.clone(model_2)
        loss_dict = model_2(images, targets) 
        losses = loss_dict['loss_rpn_box_reg']
        optimizer_2.zero_grad()
        losses.backward()
        optimizer_2.step()
        
        loss_dict = model_2(images, targets)
        losses = loss_dict['loss_rpn_box_reg'] + loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        optimizer_2.zero_grad()
        losses.backward()
        optimizer_2.step()

        #inner reptile update:
        optimizer_1.zero_grad()
        model_1.point_grad_to(model_2)
        optimizer_1.step()

        # rcnn classification:
        model_2 = model_1.clone(model_2)
        loss_dict = model_2(images, targets) 
        losses = loss_dict['loss_classifier']
        optimizer_2.zero_grad()
        losses.backward()
        optimizer_2.step()
        
        loss_dict = model_2(images, targets)
        losses = loss_dict['loss_classifier'] + loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        optimizer_2.zero_grad()
        losses.backward()
        optimizer_2.step()

        #inner reptile update:
        optimizer_1.zero_grad()
        model_1.point_grad_to(model_2)
        optimizer_1.step()


        # rcnn regression:
        model_2 = model_1.clone(model_2)
        loss_dict = model_2(images, targets) 
        losses = loss_dict['loss_box_reg']
        optimizer_2.zero_grad()
        losses.backward()
        optimizer_2.step()
        
        loss_dict = model_2(images, targets)
        losses = loss_dict['loss_box_reg'] + loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        optimizer_2.zero_grad()
        losses.backward()
        optimizer_2.step()

        #inner reptile update:
        optimizer_1.zero_grad()
        model_1.point_grad_to(model_2)
        optimizer_1.step()


        #meta reptile update:
        meta_optimizer.zero_grad()
        meta_model.point_grad_to(model_1)
        meta_optimizer.step()
        

        # net1_paras = list(filter(lambda p: p.requires_grad,model_1.parameters()))
        # net2_grad = grad(losses, filter(lambda p: p.requires_grad,model_2.parameters()), create_graph=True, allow_unused=True)
        # net1_paras = list(map(lambda aa: aa[1] - inner_lr * aa[0] if aa[0] is not None else aa[1], zip(net2_grad, net1_paras)))



        # elif key == 2:
        #     losses = loss_dict['loss_rpn_box_reg']
        # elif key == 3:
        #     losses = loss_dict['loss_classifier']
        # else:
        #     losses = loss_dict['loss_box_reg']
        #inner inner backward:
        # pdb.set_trace()
        # net2_paras = list(filter(lambda p: p.requires_grad,model_2.parameters()))
        # net2_grad = grad(losses, filter(lambda p: p.requires_grad,model_2.parameters()), create_graph=True, allow_unused=True)
        # net2_paras = list(map(lambda aa: aa[1] - inner_inner_lr * aa[0] if aa[0] is not None else aa[1], zip(net2_grad, net2_paras)))
        # model_2.meta_backward(net2_paras)
        # loss_dict = model_2(images, targets)
        # if key == 1:
        #     losses = loss_dict['loss_objectness'] + loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        # elif key == 2:
        #     losses = loss_dict['loss_rpn_box_reg']+ loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        # elif key == 3:
        #     losses = loss_dict['loss_classifier']+ loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        # else:
        #     losses = loss_dict['loss_box_reg']+ loss_dict['loss_da_instance'] + loss_dict['loss_da_consistency']+loss_dict['loss_da_image']
        # # losses = sum(loss for loss in loss_dict.values())
        # #inner backward:
        # net1_paras = list(filter(lambda p: p.requires_grad,model_1.parameters()))
        # net2_grad = grad(losses, filter(lambda p: p.requires_grad,model_2.parameters()), create_graph=True, allow_unused=True)
        # net1_paras = list(map(lambda aa: aa[1] - inner_lr * aa[0] if aa[0] is not None else aa[1], zip(net2_grad, net1_paras)))
        # model_1.meta_backward(net1_paras)

        #reptile backward:
        # if key ==4:
        #     meta_model.point_grad_to(model_1)
        #     meta_optimizer.step()
        #     model_1 = meta_model.clone(model_1)
        #     key = 0


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        
        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()
        
        meta_scheduler.step()
        scheduler_1.step()
        scheduler_2.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 1000 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=meta_optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        save_path = './experiments/reptile_reptile_42.5/inner_inner_lr_{}_inner_lr_{}_lr_{}/'.format(inner_inner_lr,inner_lr,base_lr)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save(save_path+"model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter-1:
            checkpointer.save(save_path+"model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )