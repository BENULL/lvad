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
import subprocess
from collections import defaultdict
def generateTrajectorySHTech():
    dataset_dir_path = pathlib.Path('/root/VAD/dataset/ShanghaiTechDataset/Testing/frames_part2')
    for clip_dir in tqdm(dataset_dir_path.iterdir()):
        command = f'/root/miniconda3/envs/python38/bin/python /root/VAD/AlphaPose/scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth --indir {clip_dir} --outdir /root/VAD/dataset/ShanghaiTechDataset/pose_result/testing/{clip_dir.name} --pose_track'
        resultsCommond = subprocess.run(command, shell=True)

    # dataset_dir_path = pathlib.Path('/root/VAD/dataset/ShanghaiTechDataset/training/videos')
    # for clip_dir in dataset_dir_path.iterdir():
    #     out_path = pathlib.Path(f'/root/VAD/dataset/ShanghaiTechDataset/pose_result/training/{clip_dir.name[:-4]}')
    #     if out_path.exists():
    #         continue
    #     command = f'/root/miniconda3/envs/python38/bin/python /root/VAD/AlphaPose/scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth --video {clip_dir} --outdir /root/VAD/dataset/ShanghaiTechDataset/pose_result/training/{clip_dir.name[:-4]} --pose_track'
    #     resultsCommond = subprocess.run(command, shell=True)

def generateTrajectory():
    dataset_dir_path = pathlib.Path('/root/VAD/dataset/avenue/training/frames')
    for clip_dir in tqdm(dataset_dir_path.iterdir()):
        command = f'/root/miniconda3/envs/python38/bin/python /root/VAD/AlphaPose/scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth --indir {clip_dir} --outdir /root/VAD/dataset/avenue/pose_result/training/{clip_dir.name} --pose_track'
        resultsCommond = subprocess.run(command, shell=True)

    dataset_dir_path = pathlib.Path('/root/VAD/dataset/avenue/testing/frames')
    for clip_dir in tqdm(dataset_dir_path.iterdir()):
        command = f'/root/miniconda3/envs/python38/bin/python /root/VAD/AlphaPose/scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth --indir {clip_dir} --outdir /root/VAD/dataset/avenue/pose_result/testing/{clip_dir.name} --pose_track'
        resultsCommond = subprocess.run(command, shell=True)


def generateTrajectory_IITB():

    dataset_dir_path = pathlib.Path('/root/VAD/dataset/Train_IITB_Corridor/Train')
    for clip_dir in tqdm(dataset_dir_path.iterdir()):
        out_path = pathlib.Path(f'/root/VAD/dataset/IITB-Corridor/pose_result/training/{clip_dir.name}')
        if out_path.exists():
            continue
        video = clip_dir.name + '.avi'
        command = f'/root/miniconda3/envs/python38/bin/python /root/VAD/AlphaPose/scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth --video {clip_dir/video} --outdir /root/VAD/dataset/IITB-Corridor/pose_result/training/{clip_dir.name} --pose_track --gpus 2,3'
        resultsCommond = subprocess.run(command, shell=True)

    # dataset_dir_path = pathlib.Path('/root/VAD/dataset/Test_IITB-Corridor/Test')
    # for clip_dir in tqdm(dataset_dir_path.iterdir()):
    #     video = clip_dir.name + '.avi'
    #     out_path = pathlib.Path(f'/root/VAD/dataset/IITB-Corridor/pose_result/testing/{clip_dir.name}')
    #
    #     if out_path.exists():
    #         continue
    #     command = f'/root/miniconda3/envs/python38/bin/python /root/VAD/AlphaPose/scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml --checkpoint pretrained_models/fast_421_res152_256x192.pth --video {clip_dir/video} --outdir /root/VAD/dataset/IITB-Corridor/pose_result/testing/{clip_dir.name} --pose_track --gpus 0,1'
    #     resultsCommond = subprocess.run(command, shell=True)



def processToDataset():
    dataset_dir_path = pathlib.Path('/root/VAD/dataset/IITB-Corridor/pose_result')
    output_dir = pathlib.Path('/root/VAD/lvad/data/IITB/pose')
    for dataset in dataset_dir_path.iterdir():
        phase = dataset.name
        # if phase != 'testing':
        #     continue
        for clip_pose in dataset.iterdir():
            output_dict = defaultdict(dict)
            if not (clip_pose/'alphapose-results.json').exists():
                continue
            with open(clip_pose/'alphapose-results.json', 'r') as file:
                pose_data = json.load(file)
                for person_image_pose in pose_data:
                    image_id = person_image_pose['image_id'][:-4]
                    idx = str(person_image_pose['idx'])
                    pose = dict(keypoints=person_image_pose['keypoints'], scores=person_image_pose['score'])
                    output_dict[idx][image_id]=pose
            output_filepath = output_dir/phase/'tracked_person'/f'01_{clip_pose.name}_alphapose_tracked_person.json'
            with open(output_filepath, 'w') as f:
                json.dump(output_dict, f)


if __name__ == '__main__':
    # dataset_dir_path = pathlib.Path('/root/VAD/dataset/data/HR-ShanghaiTech/testing/frame_level_masks')
    # output_dir = pathlib.Path('/root/VAD/lvad/data/HR-ShanghaiTech/pose/testing/frame_level_masks')
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


    # for _dir in dataset_dir_path.iterdir():
    #     for cliip in _dir.iterdir():
    #         shutil.copy(cliip, output_dir)


    # generateTrajectory()
    # generateTrajectorySHTech()
    # generateTrajectory_IITB()
    processToDataset()




