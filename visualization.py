#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/2/22 下午3:49
"""
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_utils import trans_list
from utils.draw_utils import draw_predict_skeleton, draw_mask_skeleton, renderBbox, draw_mask_skeleton_seperate, \
    _OPENPOSE_EDGE_COLORS_RED, _OPENPOSE_POINT_COLORS_RED, RENDER_CONFIG_OPENPOSE
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
        data = data_arr[0]

        # draw_predict_skeleton(metas, output_arr[itern],  date_time=args.ckpt_dir.split('/')[-3], v_id='01_0014')
        # draw_predict_skeleton(metas, output_arr[itern], date_time=args.ckpt_dir.split('/')[-3], v_id='03_0033')


        # draw_mask_skeleton(data[:, :args.in_channels, args.seg_len-1, :].unsqueeze(2), output_arr[itern])

        draw_mask_skeleton_seperate(data.numpy(), output_arr[itern], metas, args.ckpt_dir.split('/')[2]+'_030041_0018')

        # break

import json
def visual_bbox():
    output_dict = defaultdict(dict)
    with open('/root/VAD/dataset/ShanghaiTechDataset/pose_result/testing/03_0041/alphapose-results.json', 'r') as file:
        pose_data = json.load(file)
        for person_image_pose in pose_data:
            image_id = person_image_pose['image_id'][:-4]
            idx = str(person_image_pose['idx'])
            # pose = dict(keypoints=person_image_pose['keypoints'], scores=person_image_pose['score'])
            output_dict[idx][image_id] = person_image_pose['box']

    # with open('/root/VAD/dataset/ShanghaiTechDataset/pose_result/03_0035/alphapose_reslut.json', 'r') as file:
    skeleton_json = output_dict

    origin_path = '/root/VAD/dataset/ShanghaiTechDataset/Testing/frames_part1/03_0041/'
    visualization_path = '/root/VAD/lvad/visualization/03_0041_bbox_red/'

    key_skeleton_dict = defaultdict(list)
    for person_id, skeleton_dict in skeleton_json.items():
        # vis for paper
        if person_id == '1':
            # for key, skeleton in skeleton_dict.items():
            #     keypoints = np.array(skeleton['keypoints']).reshape(-1,3)
            #     keypoints = keypoints17_to_coco18(keypoints)
            #     key_skeleton_dict[key].append(keypoints)

            for key, bbox in skeleton_dict.items():
                # keypoints = skeleton['box']
                # keypoints = keypoints17_to_coco18(keypoints)
                key_skeleton_dict[key].append(bbox)


    for key, skeletons in key_skeleton_dict.items():
        img_path = origin_path + f'{key}.jpg'
        output_path = visualization_path + f'{key}.jpg'
        img = cv2.imread(img_path)
        for skeleton in skeletons:
            img = renderBbox(img, skeleton, inplace=True)
            # img = renderPose(img, skeleton, inplace=True)
        cv2.imwrite(output_path, img)

if __name__ == '__main__':
    # load res_npz
    # npz_path = './data/exp_dir/May12_0324/checkpoints/res_7640_.npz'

    # npz_path = './data/exp_dir/Mar31_1751/checkpoints/res_6877_.npz'


    # npz_path = './data/exp_dir/May02_0018/checkpoints/res_7744_.npz'

    # visualizaion_predict_skeleton(npz_path)

    visual_bbox()


