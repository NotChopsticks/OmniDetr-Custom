# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache-2.0 License.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Licensed under the Apache-2.0 License.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher, data_prefetcher_semi
from util.misc import NestedTensor
import torch.nn as nn
from collections import OrderedDict
import torchvision
from util import box_ops
import sys
import scipy.optimize
import math
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.filtering import unified_filter_pseudo_labels

import pprint
import concurrent.futures


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_semi(model_student: torch.nn.Module, model_teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader_label: Iterable, data_loader_unlabel: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, threshold: float = 0.7, ema_keep_rate: float = 0.9996, passed_num_path: str = '', train_setting: str = '', pixels: int = 800, w_p: float = 0.5, w_t: float = 0.5):
    attribute_errors = 0
    model_student.train()
    model_teacher.eval()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = '\nEpoch: [{}]'.format(epoch)
    print_freq = 50

    prefetcher_label = data_prefetcher_semi(data_loader_label, device, prefetch=True)
    prefetcher_unlabel = data_prefetcher_semi(data_loader_unlabel, device, prefetch=True)
    # print("prefetcher_label properties:\n", dir(prefetcher_label))
    # print("prefetcher_unlabel properties:\n", dir(prefetcher_unlabel))
    # for i in range(1300):
    #     print("SKIPPING", i)
    samples_label_q, targets_label_q, records_label_q, samples_label_k, targets_label_k, records_label_k, indicator_label, labeltype_label = prefetcher_label.next()
    samples_unlabel_q, targets_unlabel_q, records_unlabel_q, samples_unlabel_k, targets_unlabel_k, records_unlabel_k, indicator_unlabel, labeltype_unlabel = prefetcher_unlabel.next()
    # print("targets_label_q object type:\n", type(targets_label_q))  # list
    # print("targets_label_q properties:\n", dir(targets_label_q))
    # wait = input("Press Any Key to continue.")
    error_count = 0
    unlabeled_only_iterations = 0
    skip_iteration = 0
    end_reached = False
    iteration_count = 0
    temp_sample_storage = []
    temp_target_storage = []
    temp_indicator_storage = []
    batch_counter = 0
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    # for _ in metric_logger.log_every(range(len(data_loader_label)), print_freq, header):
    # iteration_count = 0
    num_unlabeled_samples = len(data_loader_unlabel)
    num_extra_samples = (num_unlabeled_samples/2) % 3
    num_unlabeled_only = int((num_unlabeled_samples/2)/3)  # Number of iterations where you'll only have unlabeled data assuming they're taken 3 at a time
    total_iterations = int((num_unlabeled_samples/2) + num_unlabeled_only)-3  # The minus 3 is simply to provide a buffer to try and avoid hanging. Will/needs to be changed in the future

    total_iterations = len(data_loader_unlabel) - 3
    print("====== Starting Epoch with", total_iterations, "Planned Epochs ======")
    for _ in metric_logger.log_every(range(total_iterations), print_freq, header):
        print("=== Iteration:", iteration_count, "===")
        if end_reached:
            print("=== ENDING LOOP: ALL SAMPLES PROCESSED ===")
            continue
        # if skip_iteration > 0:
        #     skip_iteration -= 1
        #     continue
        # iteration_count += 1
        # if iteration_count == print_freq:
            # print("")
            # iteration_count = 0
        label_present = False
        unlabeled_present = False

        if samples_label_q is not None and samples_label_k is not None:
            tensor_label_q = samples_label_q.tensors
            tensor_label_k = samples_label_k.tensors
            label_present = True

        # try:
        if samples_unlabel_q is not None and samples_unlabel_k is not None:
            tensor_unlabel_q = samples_unlabel_q.tensors
            tensor_unlabel_k = samples_unlabel_k.tensors
            unlabeled_present = True

        if not label_present and not unlabeled_present:
            print("IT REACHED SOMEWHERE IT SHOULDN'T HAVE BEEN ABLE TO REACH: BOTH SAMPLES WERE NONE!!!")
            error_count += 1
            print(str(attribute_errors), " Errors have occurred now")
            break
            # samples_label_q, targets_label_q, records_label_q, samples_label_k, targets_label_k, records_label_k, indicator_label, labeltype_label = prefetcher_label.next()
            # samples_unlabel_q, targets_unlabel_q, records_unlabel_q, samples_unlabel_k, targets_unlabel_k, records_unlabel_k, indicator_unlabel, labeltype_unlabel = prefetcher_unlabel.next()
            # continue
        elif unlabeled_present and not label_present:

            print("LABELED DATA EXHAUSTED")
            unlabeled_only_iterations += 1
            batch_counter += 1
        # except AttributeError:
        #     attribute_errors += 1
        #     print("An AttributeError occurred. samples_unlabel_q might be None.")
        #     print(str(attribute_errors), "Attribute Errors have occurred")
        #     print("samples_unlabel_q:\n", samples_unlabel_q)
        #     # print("samples_unlabel_q properties:\n", dir(samples_unlabel_q))
        #     print("samples_unlabel_k:\n", samples_unlabel_k)
        #     # print("samples_unlabel_k properties:\n", dir(samples_unlabel_k))
        #     samples_label_q, targets_label_q, records_label_q, samples_label_k, targets_label_k, records_label_k, indicator_label, labeltype_label = prefetcher_label.next()
        #     samples_unlabel_q, targets_unlabel_q, records_unlabel_q, samples_unlabel_k, targets_unlabel_k, records_unlabel_k, indicator_unlabel, labeltype_unlabel = prefetcher_unlabel.next()
        #     continue

        if label_present:
            tensor_label_q = torch.squeeze(tensor_label_q)
            tensor_label_k = torch.squeeze(tensor_label_k)

        if unlabeled_present:
            tensor_unlabel_q = torch.squeeze(tensor_unlabel_q)
            tensor_unlabel_k = torch.squeeze(tensor_unlabel_k)
            # tensor_unlabel_q_original = tensor_unlabel_q

        samples = [tensor_unlabel_k]

        # for i in range(2):

        with torch.no_grad():
            outputs_unsup_q = model_teacher(samples)

        # pseudo label filtering
        # if train_setting == 'coco_add_semi' or train_setting == 'voc_semi':
        #     pseudo_label_k = unified_filter_pseudo_labels(outputs_unsup_q, targets_unlabel_q, targets_unlabel_k,
        #                                                   records_unlabel_q, records_unlabel_k, pixels,
        #                                                   label_type='Unsup', w_p=0.5, w_t=0.5, is_binary=False,
        #                                                   threshold=threshold)
        # if train_setting == 'coco_35to80_tagsU' or train_setting == 'coco_objects_tagsU':
        #     pseudo_label_k = unified_filter_pseudo_labels(outputs_unsup_q, targets_unlabel_q, targets_unlabel_k,
        #                                                   records_unlabel_q, records_unlabel_k, pixels,
        #                                                   label_type='tagsU', w_p=0.5, w_t=0.5, is_binary=False,
        #                                                   threshold=threshold)
        # if train_setting == 'coco_35to80_point' or train_setting == 'coco_objects_points':
        #     pseudo_label_k = unified_filter_pseudo_labels(outputs_unsup_q, targets_unlabel_q, targets_unlabel_k,
        #                                                   records_unlabel_q, records_unlabel_k, pixels,
        #                                                   label_type='pointsK', w_p=0.5, w_t=0.5, is_binary=False,
        #                                                   threshold=threshold)
        if train_setting == 'dvd' or train_setting == 'bees_omni' or train_setting == 'voc_omni' or train_setting == 'objects_omni' or train_setting == 'crowdhuman_omni':
            if labeltype_unlabel[0] == 'tagsU':
                pseudo_label_k = unified_filter_pseudo_labels(outputs_unsup_q, targets_unlabel_q, targets_unlabel_k,
                                                              records_unlabel_q, records_unlabel_k, pixels,
                                                              labeltype_unlabel[0], w_p, w_t, is_binary=False,
                                                              threshold=threshold)
            elif labeltype_unlabel[0] == 'tagsK':
                pseudo_label_k = unified_filter_pseudo_labels(outputs_unsup_q, targets_unlabel_q, targets_unlabel_k,
                                                              records_unlabel_q, records_unlabel_k, pixels,
                                                              labeltype_unlabel[0], w_p, w_t, is_binary=False,
                                                              threshold=threshold)
            elif labeltype_unlabel[0] == 'pointsU':
                pseudo_label_k = unified_filter_pseudo_labels(outputs_unsup_q, targets_unlabel_q, targets_unlabel_k,
                                                              records_unlabel_q, records_unlabel_k, pixels,
                                                              labeltype_unlabel[0], w_p=1, w_t=0, is_binary=False,
                                                              threshold=threshold)
            elif labeltype_unlabel[0] == 'pointsK':
                pseudo_label_k = unified_filter_pseudo_labels(outputs_unsup_q, targets_unlabel_q, targets_unlabel_k,
                                                              records_unlabel_q, records_unlabel_k, pixels,
                                                              labeltype_unlabel[0], w_p, w_t, is_binary=False,
                                                              threshold=threshold)
            elif labeltype_unlabel[0] == 'boxesEC' or labeltype_unlabel[0] == 'boxesU':
                pseudo_label_k = unified_filter_pseudo_labels(outputs_unsup_q, targets_unlabel_q, targets_unlabel_k,
                                                              records_unlabel_q, records_unlabel_k, pixels,
                                                              labeltype_unlabel[0], w_p, w_t, is_binary=False,
                                                              threshold=threshold)
            else:
                if train_setting == 'bees_omni' or train_setting == 'crowdhuman_omni':
                    pseudo_label_k = unified_filter_pseudo_labels(outputs_unsup_q, targets_unlabel_q, targets_unlabel_k,
                                                                  records_unlabel_q, records_unlabel_k, pixels,
                                                                  labeltype_unlabel[0], w_p, w_t, is_binary=True,
                                                                  threshold=threshold)
                else:
                    pseudo_label_k = unified_filter_pseudo_labels(outputs_unsup_q, targets_unlabel_q, targets_unlabel_k,
                                                                  records_unlabel_q, records_unlabel_k, pixels,
                                                                  labeltype_unlabel[0], w_p, w_t, is_binary=False,
                                                                  threshold=threshold)

        if not label_present:
            # break
            # else:
            print("LABELED DATA EXHAUSTED")
            unlabeled_samples = []
            pseudo_labels = []
            unlabeled_indicators = []
            second_iteration = False
            # with torch.no_grad():
            #     outputs_unsup_q = model_teacher(samples)
            for i in range(2):
                # skip_iteration += 1

                # samples_label_q_new, targets_label_q_new, records_label_q_new, samples_label_k_new, targets_label_k_new, records_label_k_new, indicator_label_new, labeltype_label_new = prefetcher_label.next()
                samples_unlabel_q_new, targets_unlabel_q_new, records_unlabel_q_new, samples_unlabel_k_new, targets_unlabel_k_new, records_unlabel_k_new, indicator_unlabel_new, labeltype_unlabel_new = prefetcher_unlabel.next()
                print("DATA fetched")
                if samples_unlabel_q_new is None:
                    end_reached = True
                    break

                tensor_unlabel_q_new = samples_unlabel_q_new.tensors
                tensor_unlabel_k_new = samples_unlabel_k_new.tensors
                tensor_unlabel_q_new = torch.squeeze(tensor_unlabel_q_new)
                tensor_unlabel_k_new = torch.squeeze(tensor_unlabel_k_new)

                unlabeled_samples.append(tensor_unlabel_q_new)
                unlabeled_indicators.append(indicator_unlabel_new[0])

                samples = [tensor_unlabel_k_new]

                with torch.no_grad():
                    outputs_unsup_q_new = model_teacher(samples)
                print("Teacher model done")
                if train_setting == 'dvd' or train_setting == 'bees_omni' or train_setting == 'voc_omni' or train_setting == 'objects_omni' or train_setting == 'crowdhuman_omni':
                    if labeltype_unlabel_new[0] == 'tagsU':
                        pseudo_label_k_new = unified_filter_pseudo_labels(outputs_unsup_q_new, targets_unlabel_q_new,
                                                                      targets_unlabel_k_new,
                                                                      records_unlabel_q_new, records_unlabel_k_new, pixels,
                                                                      labeltype_unlabel_new[0], w_p, w_t,
                                                                      is_binary=False,
                                                                      threshold=threshold)
                    elif labeltype_unlabel_new[0] == 'tagsK':
                        pseudo_label_k_new = unified_filter_pseudo_labels(outputs_unsup_q_new, targets_unlabel_q_new,
                                                                      targets_unlabel_k_new,
                                                                      records_unlabel_q_new, records_unlabel_k_new, pixels,
                                                                      labeltype_unlabel_new[0], w_p, w_t,
                                                                      is_binary=False,
                                                                      threshold=threshold)
                    elif labeltype_unlabel[0] == 'pointsU':
                        pseudo_label_k_new = unified_filter_pseudo_labels(outputs_unsup_q_new, targets_unlabel_q_new,
                                                                      targets_unlabel_k_new,
                                                                      records_unlabel_q_new, records_unlabel_k_new, pixels,
                                                                      labeltype_unlabel_new[0], w_p=1, w_t=0,
                                                                      is_binary=False,
                                                                      threshold=threshold)
                    elif labeltype_unlabel[0] == 'pointsK':
                        pseudo_label_k_new = unified_filter_pseudo_labels(outputs_unsup_q_new, targets_unlabel_q_new,
                                                                      targets_unlabel_k_new,
                                                                      records_unlabel_q_new, records_unlabel_k_new, pixels,
                                                                      labeltype_unlabel_new[0], w_p, w_t,
                                                                      is_binary=False,
                                                                      threshold=threshold)
                    elif labeltype_unlabel[0] == 'boxesEC' or labeltype_unlabel[0] == 'boxesU':
                        pseudo_label_k_new = unified_filter_pseudo_labels(outputs_unsup_q_new, targets_unlabel_q_new,
                                                                      targets_unlabel_k_new,
                                                                      records_unlabel_q_new, records_unlabel_k_new, pixels,
                                                                      labeltype_unlabel_new[0], w_p, w_t,
                                                                      is_binary=False,
                                                                      threshold=threshold)
                    else:
                        if train_setting == 'bees_omni' or train_setting == 'crowdhuman_omni':
                            pseudo_label_k_new = unified_filter_pseudo_labels(outputs_unsup_q_new, targets_unlabel_q_new,
                                                                      targets_unlabel_k_new,
                                                                      records_unlabel_q_new, records_unlabel_k_new, pixels,
                                                                      labeltype_unlabel_new[0], w_p, w_t,
                                                                          is_binary=True,
                                                                          threshold=threshold)
                        else:
                            pseudo_label_k_new = unified_filter_pseudo_labels(outputs_unsup_q_new, targets_unlabel_q_new,
                                                                      targets_unlabel_k_new,
                                                                      records_unlabel_q_new, records_unlabel_k_new, pixels,
                                                                      labeltype_unlabel_new[0], w_p, w_t,
                                                                          is_binary=False,
                                                                          threshold=threshold)
                print("pseudo labels filtered")
                pseudo_labels.append(pseudo_label_k_new[0])
        print("pseudo labels generated")
        if end_reached:
            continue
        samples = []
        if label_present:
            samples.append(tensor_label_q)
            samples.append(tensor_label_k)
            samples.append(tensor_unlabel_q)
        # elif batch_counter == 2:
        #     samples = temp_sample_storage
        #     samples.append(tensor_unlabel_q)
        else:
            temp_sample_storage.append(tensor_unlabel_q)
            # samples.append(unlabeled_samples[0])
            # samples.append(unlabeled_samples[1])

            # samples = samples + unlabeled_samples
            # samples.append(tensor_unlabel_q_new)


        # samples.append(tensor_unlabel_q)

        if label_present:
            # targets_label_q, targets_label_k, and pseudo_label_k are lists that contain a single dictionary each
            # targets becomes a list containing 3 dictionaries
            # print("targets_label_q:", len(targets_label_q))
            # print_object_info(targets_label_q[0])
            targets_label_q.append(targets_label_k[0])  # Dictionary
            targets_label_q.append(pseudo_label_k[0])  # Dictionary
            targets = targets_label_q  # list containing 3 dictionaries

            # print("targets_label_k:", len(targets_label_k))
            # print_object_info(targets_label_k[0])
            # print("pseudo_label_k:", len(pseudo_label_k))
            # print_object_info(pseudo_label_k[0])
            # print("targets:")
            # pprint.pprint(targets)
            # print("targets object type:\n", type(targets))
            # print("targets properties:\n", dir(targets))
            # print("targets_label_k object type:\n", type(targets_label_k[0]))  # dictionary
            # print("targets_label_k properties:\n", dir(targets_label_k[0]))
            # print("pseudo_label_k object type:\n", type(pseudo_label_k[0]))  # dictionary
            # print("pseudo_label_k properties:\n", dir(pseudo_label_k[0]))
            # wait = input("Press Any Key to continue.")
        # elif batch_counter == 2:
        #     targets = temp_target_storage
        #     targets.append(pseudo_label_k[0])

        else:
            # temp_target_storage.append(pseudo_label_k[0])
            targets = [pseudo_label_k[0]]
            # print_dict(pseudo_label_k[0])
            targets.append(pseudo_labels[0])
            # print_dict(pseudo_labels[0])
            targets.append(pseudo_labels[1])
            # print_dict(pseudo_labels[1])
            # targets = targets + pseudo_labels
            # print("pseudo_label_k object type:\n", type(pseudo_label_k[0]))  # dictionary
            # print("pseudo_label_k properties:\n", dir(pseudo_label_k[0]))
            # print("pseudo_label_k_new object type:\n", type(pseudo_label_k_new[0]))  # dictionary
            # print("pseudo_label_k_new properties:\n", dir(pseudo_label_k_new[0]))
            # wait = input("Press Any Key to continue.")

        # outputs = model_student(samples)
        # print("Student model done")

        indicators = []
        if label_present:
            indicators.append(indicator_label[0])
            indicators.append(indicator_label[0])
            indicators.append(indicator_unlabel[0])
        # elif batch_counter == 2:
        #     indicators = temp_indicator_storage
        #     indicators.append(indicator_unlabel[0])
        else:
            temp_indicator_storage.append(indicator_unlabel[0])
            indicators.append(unlabeled_indicators[0])
            indicators.append(unlabeled_indicators[1])
            # indicators = indicators + unlabeled_indicators
            # indicators.append(indicator_unlabel[0])
            # indicators.append(indicator_unlabel_new[0])

        # if not label_present:
            # indicators.append(indicator_unlabel_new[0])

        # Attempt at fixing the issue of random data samples causing the program to hang by using the future function of the concurrent library
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future = executor.submit(criterion, outputs, targets, indicators)
        #     try:
        #         loss_dict = future.result(timeout=10)
        #         print("= loss dict created =")
        #     except concurrent.futures.TimeoutError:
        #         print('Function took too long to complete.')
        #         samples_label_q, targets_label_q, records_label_q, samples_label_k, targets_label_k, records_label_k, indicator_label, labeltype_label = prefetcher_label.next()
        #         samples_unlabel_q, targets_unlabel_q, records_unlabel_q, samples_unlabel_k, targets_unlabel_k, records_unlabel_k, indicator_unlabel, labeltype_unlabel = prefetcher_unlabel.next()
        #         continue
        #     except:
        #         print("\nERROR OCCURRED\n")
        #         samples_label_q, targets_label_q, records_label_q, samples_label_k, targets_label_k, records_label_k, indicator_label, labeltype_label = prefetcher_label.next()
        #         samples_unlabel_q, targets_unlabel_q, records_unlabel_q, samples_unlabel_k, targets_unlabel_k, records_unlabel_k, indicator_unlabel, labeltype_unlabel = prefetcher_unlabel.next()
        #         continue
        # try:
        if not label_present and batch_counter != 2:
            print("Moving to next iteration")
            samples_label_q, targets_label_q, records_label_q, samples_label_k, targets_label_k, records_label_k, indicator_label, labeltype_label = prefetcher_label.next()
            samples_unlabel_q, targets_unlabel_q, records_unlabel_q, samples_unlabel_k, targets_unlabel_k, records_unlabel_k, indicator_unlabel, labeltype_unlabel = prefetcher_unlabel.next()
            continue

        outputs = model_student(samples)
        print("Student model done")

        loss_dict = criterion(outputs, targets, indicators)  # It's hanging here when running it on all the data
        print("loss dict done")
        if loss_dict is None:
            print("Moving to next iteration")
            samples_label_q, targets_label_q, records_label_q, samples_label_k, targets_label_k, records_label_k, indicator_label, labeltype_label = prefetcher_label.next()
            samples_unlabel_q, targets_unlabel_q, records_unlabel_q, samples_unlabel_k, targets_unlabel_k, records_unlabel_k, indicator_unlabel, labeltype_unlabel = prefetcher_unlabel.next()
            continue
        print("loss dict created")
        # except:
        #     print("\nERROR OCCURRED\n")
        #     samples_label_q, targets_label_q, records_label_q, samples_label_k, targets_label_k, records_label_k, indicator_label, labeltype_label = prefetcher_label.next()
        #     samples_unlabel_q, targets_unlabel_q, records_unlabel_q, samples_unlabel_k, targets_unlabel_k, records_unlabel_k, indicator_unlabel, labeltype_unlabel = prefetcher_unlabel.next()
        #     continue
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        print("loss value calculated")

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model_student.parameters(), max_norm)
        optimizer.step()

        print("optimizer done")

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        print("metrics logged")

        samples_label_q, targets_label_q, records_label_q, samples_label_k, targets_label_k, records_label_k, indicator_label, labeltype_label = prefetcher_label.next()
        samples_unlabel_q, targets_unlabel_q, records_unlabel_q, samples_unlabel_k, targets_unlabel_k, records_unlabel_k, indicator_unlabel, labeltype_unlabel = prefetcher_unlabel.next()

        print("next samples fetched")

        student_model_dict = {
            key: value for key, value in model_student.state_dict().items()
        }
        new_teacher_dict = OrderedDict()
        for key, value in model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] * (1 - ema_keep_rate) + value * ema_keep_rate
                )
        model_teacher.load_state_dict(new_teacher_dict)
        print("End of Iteration\n")
        if unlabeled_present and not label_present:
            if batch_counter == 2:
                temp_sample_storage = []
                temp_target_storage = []
                temp_indicator_storage = []
                batch_counter = 0
            else:
                batch_counter += 1
        iteration_count += 1

    # samples_left = 0
    # while samples_unlabel_q is not None:
    #     samples_left += 1
    #     samples_unlabel_q, targets_unlabel_q, records_unlabel_q, samples_unlabel_k, targets_unlabel_k, records_unlabel_k, indicator_unlabel, labeltype_unlabel = prefetcher_unlabel.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print(str(error_count), "Errors occurred this Epoch")
    print(str(unlabeled_only_iterations), "'Unlabeled Data Only' Iterations occurred this Epoch")
    print("Unlabeled Samples Remaining:", str(samples_left))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_burnin(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


def print_dict(d, max_array_length=3):
    for key, value in d.items():
        if isinstance(value, list) or isinstance(value, tuple):
            print(f"{key}: (length: {len(value)})")
            if len(value) > 0:
                print("First elements:", value[:min(len(value), max_array_length)])
        else:
            print(f"{key}: {value}")

def print_object_info(obj):
    # Print type of object
    print("Type:", type(obj))

    # If the object is a dictionary
    if isinstance(obj, dict):
        for key, value in obj.items():
            # If the value is a list, print the length and abbreviated version if it's long
            if isinstance(value, list):
                print(f"{key} (length: {len(value)}):", value if len(value) <= 10 else f"{value[:5]} ... {value[-5:]}")
            else:
                print(f"{key}:", value)
    else:
        # Print all attributes and their values
        for attr in dir(obj):
            if not (attr.startswith("__") and attr.endswith("__")):  # exclude Python's special methods
                try:
                    attr_value = getattr(obj, attr)
                    if isinstance(attr_value, list):
                        print(f"{attr} (length: {len(attr_value)}):", attr_value if len(attr_value) <= 10 else f"{attr_value[:5]} ... {attr_value[-5:]}")
                    else:
                        print(f"{attr}:", attr_value)
                except Exception as e:
                    print(f"{attr}: <error: {e}>")

    # Print the object in a pretty way
    # print("Pretty print:")
    # pprint.pprint(obj)

