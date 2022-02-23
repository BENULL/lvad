#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/2/22 下午3:49
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_utils import trans_list
from utils.draw_utils import draw_predict_skeleton, draw_mask_skeleton
from utils.pose_seg_dataset import PoseSegDataset


def visualizaion_predict_skeleton(res_npz_path):
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

    for itern, data_arr in enumerate(loader):
        metas = data_arr[2]
        # data = data_arr[0]
        draw_predict_skeleton(metas, output_arr[itern],  date_time=args.ckpt_dir.split('/')[-3])
        # draw_mask_skeleton(data[:, :args.in_channels, args.seg_len-1, :].unsqueeze(2), output_arr[itern])
        # break


if __name__ == '__main__':
    # load res_npz
    npz_path = './data/exp_dir/Feb22_1624/checkpoints/res_6595_.npz'
    visualizaion_predict_skeleton(npz_path)

