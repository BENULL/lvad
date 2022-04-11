#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/4/8 上午1:08
"""
import os
import random
import collections
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_eval import get_dataset_and_loader
from utils.data_utils import trans_list
from utils.pose_seg_dataset import PoseSegDataset
from utils.pose_ad_argparse import init_parser, init_sub_args
from utils.scoring_utils import score_dataset
from utils.train_utils import Trainer, csv_log_dump
from visualization import visualizaion_predict_skeleton

def load_exp_data(res_npz_path):
    data = np.load(res_npz_path, allow_pickle=True)
    args, output_arr, rec_loss_arr = data['args'].tolist(), data['output_arr'], data['rec_loss_arr']
    # load test dataset
    transform_list = trans_list[:args.num_transform]

    dataset_args = {'transform_list': transform_list, 'debug': args.debug, 'headless': args.headless,
                    'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale, 'seg_len': args.seg_len,
                    'return_indices': True, 'return_metadata': True, 'hr': args.hr}

    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}

    dataset_args['seg_stride'] = 1  # No strides for test set
    dataset_args['train_seg_conf_th'] = 0.0
    dataset = PoseSegDataset(args.pose_path['test'], **dataset_args)
    loader = DataLoader(dataset, **loader_args, shuffle=False)
    return args, output_arr, rec_loss_arr, loader, dataset

if __name__ == '__main__':
    # parser = init_parser()
    # args = parser.parse_args()
    #
    # if args.seed == 999:  # Record and init seed
    #     args.seed = torch.initial_seed()
    # else:
    #     random.seed(args.seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     torch.manual_seed(args.seed)
    #     np.random.seed(args.seed)
    #
    # args, res_args = init_sub_args(args)
    # print(args)
    #
    # dataset, loader = get_dataset_and_loader(args)

    # npz_path = './data/exp_dir/Apr10_2052/checkpoints/res_7090_.npz'
    # npz_path = './data/exp_dir/res_7062_.npz'
    # args, output_arr, rec_loss_arr, loader, dataset = load_exp_data(npz_path)
    # from utils.scoring_utils_one_sample import get_dataset_scores
    #
    # score_vals = np.array(rec_loss_arr)  # [samples, ]
    #
    # gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores(score_vals, dataset.metadata, None, None, args)

    training_error_by_scene = np.load('/root/VAD/lvad/data/exp_dir/Apr11_1434/checkpoints/training_error_by_scene.npz',
                                      allow_pickle=True)
    _dict = training_error_by_scene['training_error_by_scene']
    print('123')