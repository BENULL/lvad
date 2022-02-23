#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/1/10 下午8:32
"""
import pathlib
import numpy as np
import json
from tqdm import tqdm
import shutil

if __name__ == '__main__':
    dataset_dir_path = pathlib.Path('/root/VAD/dataset/data/HR-ShanghaiTech/testing/frame_level_masks')
    output_dir = pathlib.Path('/root/VAD/lvad/data/HR-ShanghaiTech/pose/testing/frame_level_masks')
    # for scene_clip_dir in tqdm(dataset_dir_path.iterdir()):
    #     for clip in scene_clip_dir.iterdir():
    #         trajectory_filename = f'{scene_clip_dir.stem}_{clip.stem}_alphapose_tracked_person.json'
    #         data = dict()
    #         for person_trajectory in clip.glob('*.csv'):
    #             person_idx = int(person_trajectory.stem)
    #             trajectory = np.loadtxt(person_trajectory, dtype=np.float32, delimiter=',', ndmin=2)
    #             trajectory_frames, trajectory_coordinates = trajectory[:, 0].astype(np.int32), trajectory[:, 1:]
    #             person_frame_data = {}
    #             for frame_idx, keypoints in enumerate(trajectory_coordinates):
    #                 keypoints = keypoints.reshape((-1, 2))
    #                 keypoints = np.column_stack((keypoints, np.ones(keypoints.shape[0])))
    #                 keypoints = list(keypoints.flatten())
    #                 person_frame_data[str(int(trajectory_frames[frame_idx]))] = dict(keypoints=keypoints)
    #             data[str(person_idx)] = person_frame_data
    #         with open(output_dir/trajectory_filename, 'w') as file:
    #             json.dump(data, file)


    for _dir in dataset_dir_path.iterdir():
        for cliip in _dir.iterdir():
            shutil.copy(cliip, output_dir)





