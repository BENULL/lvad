#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/2/22 下午6:40
"""
import pathlib
import cv2
import os
import shutil
import numpy as np
import json
import collections

from utils.pose_seg_dataset import keypoints17_to_coco18

_OPENPOSE_POINT_COLORS = [
    (255, 255, 0), (255, 191, 0),
    (102, 255, 0), (255, 77, 0), (0, 255, 0),
    (255, 255, 77), (204, 255, 77), (255, 204, 77),
    (77, 255, 191), (255, 191, 77), (77, 255, 91),
    (255, 77, 204), (204, 255, 77), (255, 77, 191),
    (191, 255, 77), (255, 77, 127),
    (127, 255, 77), (255, 255, 0)]

_OPENPOSE_EDGES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16),
    (0, 15), (15, 17)
]

_OPENPOSE_EDGE_COLORS = [
    (0, 0, 255),
    (0, 84, 255), (0, 168, 0), (0, 255, 168),
    (84, 0, 168), (84, 84, 255), (84, 168, 0),
    (84, 255, 84), (168, 0, 255), (168, 84, 255),
    (168, 168, 0), (168, 255, 84), (255, 0, 0),
    (255, 84, 255), (255, 168, 0),
    (255, 255, 84), (255, 0, 168)]

_OPENPOSE_EDGE_COLORS_BLUE = [(237, 40, 33)] * 17
_OPENPOSE_EDGE_COLORS_RED = [(0, 0, 255)]*17
_OPENPOSE_POINT_COLORS_BLUE = [(255, 9, 9)] * 18
_OPENPOSE_POINT_COLORS_RED = [(0, 0, 255)]*18

RENDER_CONFIG_OPENPOSE = {
    'edges': _OPENPOSE_EDGES,
    'edgeColors': _OPENPOSE_EDGE_COLORS_BLUE,
    'edgeWidth': 1,
    'pointColors': _OPENPOSE_POINT_COLORS_BLUE ,
    'pointRadius': 2
}


def preparePoint(points, imageSize, invNorm):
    if invNorm == 'auto':
        invNorm = np.bitwise_and(points >= 0, points <= 1).all()

    if invNorm:
        w, h = imageSize
        trans = np.array([[w, 0], [0, h]])
        points = (trans @ points.T).T

    return points.astype(np.int32)


def renderPose(image, poses, inplace: bool = True, inverseNormalization='auto'):
    """绘制骨架

    参数
        image: 原图

        poses: 一组或多组关节点坐标

        config: 配置项

        inplace: 是否绘制在原图上

        inverseNormalization` 是否[True|False]进行逆归一化, 当值为auto时将根据坐标值自动确定

    返回
        输出图像, inplace为True时返回image, 为False时返回新的图像
    """
    poses = np.array(poses)
    if not inplace:
        image = image.copy()

    if len(poses.shape) == 2:
        poses = poses[None, :, :2]

    # symm_range
    # poses = (poses+1) / 2


    if inverseNormalization not in ['auto', True, False]:
        raise ValueError(f'Unknown "inverseNormalization" value {inverseNormalization}')

    _isPointValid = lambda point: point[0] != 0 and point[1] != 0
    _FILL_CIRCLE = -1
    for pose in poses:
        pose = preparePoint(pose, (image.shape[1], image.shape[0]), inverseNormalization)
        validPointIndices = set(filter(lambda i: _isPointValid(pose[i]), range(pose.shape[0])))
        for i, (start, end) in enumerate(RENDER_CONFIG_OPENPOSE['edges']):
            if start in validPointIndices and end in validPointIndices:
                cv2.line(image, tuple(pose[start]), tuple(pose[end]), RENDER_CONFIG_OPENPOSE['edgeColors'][i],
                         RENDER_CONFIG_OPENPOSE['edgeWidth'])

        for i in validPointIndices:
            cv2.circle(image, tuple(pose[i]), RENDER_CONFIG_OPENPOSE['pointRadius'],
                       tuple(RENDER_CONFIG_OPENPOSE['pointColors'][i]), _FILL_CIRCLE)

    return image


def renderBbox(image, box, inplace: bool = True, inverseNormalization='auto'):
    if not inplace:
        image = image.copy()

    if inverseNormalization not in ['auto', True, False]:
        raise ValueError(f'Unknown "inverseNormalization" value {inverseNormalization}')
    if len(box) == 4:
        box = np.array(box).reshape(2, 2)
        box = preparePoint(box, (image.shape[1], image.shape[0]), inverseNormalization)
        cv2.rectangle(image, tuple(box[0]), tuple(box[0]+box[1]), (255, 0, 0), thickness=1)
    return image

def draw_skeleton_on_ShanghaiTech():
    with open('/root/VAD/lvad/data/ShanghaiTech/pose/testing/tracked_person/01_0014_alphapose_tracked_person.json', 'r') as file:
        skeleton_json = json.load(file)
        visualization_path = '/root/VAD/lvad/visualization/01_0014/'
        key_skeleton_dict = collections.defaultdict(list)
        for person_id, skeleton_dict in skeleton_json.items():
            for key, skeleton in skeleton_dict.items():
                keypoints = np.array(skeleton['keypoints']).reshape(-1,3)
                keypoints = keypoints17_to_coco18(keypoints)
                key_skeleton_dict[key].append(keypoints)
        for key, skeletons in key_skeleton_dict.items():
            img_path = visualization_path + f'{key}.jpg'
            img = cv2.imread(img_path)
            for skeleton in skeletons:
                img = renderPose(img, skeleton, inplace=True)
            cv2.imwrite(img_path, img)



def draw_predict_skeleton( metas, output_arr, date_time):

    RENDER_CONFIG_OPENPOSE['pointColors'] = _OPENPOSE_POINT_COLORS_RED
    RENDER_CONFIG_OPENPOSE['edgeColors'] = _OPENPOSE_EDGE_COLORS_RED

    video_path = '/root/VAD/lvad/visualization/01_0014/'
    visualization_path = f'/root/VAD/lvad/visualization/{date_time}_01_0014_res/'
    shutil.rmtree(visualization_path)
    shutil.copytree(video_path, visualization_path)


    # npz_path = './data/exp_dir/Feb22_1624/checkpoints/res_6595_.npz'
    # data = np.load(npz_path, allow_pickle=True)
    # args, output_arr, rec_loss_arr = data['args'].tolist(), data['output_arr'], data['rec_loss_arr']
    # output = output_arr[0]

    for predict, meta in zip(output_arr, metas):
        if meta[0] == 1 and meta[1] == 14:
            img_idx = meta[4]
            # DRAW
            img_path = visualization_path + f'{img_idx:03d}.jpg'
            img = cv2.imread(img_path)
            predict = predict.squeeze(1).T
            predict = (predict+1)/2
            renderPoseImage = renderPose(img, predict, inplace=False)
            cv2.imwrite(img_path, renderPoseImage)




if __name__ == '__main__':
    # draw_skeleton_on_ShanghaiTech()
    draw_predict_skeleton()