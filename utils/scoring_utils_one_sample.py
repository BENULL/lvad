import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from sklearn import mixture
from joblib import dump, load
import collections
from sklearn.preprocessing import quantile_transform

from utils.draw_utils import draw_anomaly_score_curve
from utils.pose_seg_dataset import HUMAN_IRRELATED_CLIPS


def cal_clip_roc_auc(gt_arr, scores_arr):
    auc_arr = []
    for gt, scores in zip(gt_arr, scores_arr):
        try:
            auc_arr.append(roc_auc_score(gt, scores))
        except ValueError:
            auc_arr.append(0)
    return auc_arr


def normalize_scores(score_arrs):
    score_arrs_normalized = []
    for scores in score_arrs:
        score_max = np.max(scores, axis=0)
        score_normalized = (scores / score_max) * 0.99
        score_arrs_normalized.append(score_normalized)
    return score_arrs_normalized

def quantile_transform_errors(score_arrs):
    score_arrs_transformed = []
    for scores in score_arrs:
        scores_smoothed = quantile_transform(scores.reshape(-1, 1)).reshape(-1)
        score_arrs_transformed.append(scores_smoothed)
    return score_arrs_transformed

def smooth_scores(score_arrs, sigma=40):
    score_arrs_smoothed = []
    for scores in score_arrs:
        scores_smoothed = gaussian_filter1d(scores, sigma)
        score_arrs_smoothed.append(scores_smoothed)
    return score_arrs_smoothed

def get_dataset_scores_by_rec_add_pre_score(scores, metadata, person_keys, max_clip=None, scene_id=None, args=None):
    """
    :param scores: [samples, T]
    :param metadata: [samples, 5] scene_id, clip_id, person_id, start_frame_idx, end_frame_idx, x_g, y_g, w, h
    :param max_clip:
    :param scene_id:
    :param args:
    :return:
    """
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    metadata_np = np.array(metadata)
    per_frame_scores_root = f'{args.data_dir}/pose/testing/test_frame_mask/'
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    # calculate per clip
    for clip in clip_list:
        scene_id, clip_id = clip.split('.')[0].split('_')
        if args.hr and f'{scene_id}_{clip_id}' in HUMAN_IRRELATED_CLIPS:
            continue
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        scene_id, clip_id = int(scene_id), int(clip_id)
        # find the sample index of the clip
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        clip_metadata = metadata[clip_metadata_inds]
        # person_idxs set (person in clip)
        clip_fig_idxs = set([arr[2] for arr in clip_metadata])
        clip_frame_num = clip_gt.shape[0]
        scores_zeros = np.zeros(clip_frame_num)  # [clip frames, ]
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

        seg_len = args.seg_len
        # scores_per_frame = [[0] for _ in range(clip_frame_num)]
        for person_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 2] == person_id))[0]
            pid_segment_scores = scores[person_metadata_inds] # [N, T ]
            # start frame index
            pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds])

            pid_segment_rec_scores = np.zeros(clip_frame_num)
            pid_segment_pre_scores = np.zeros(clip_frame_num)

            for segment_scores, start in zip(pid_segment_scores, pid_frame_inds):
                pid_segment_rec_scores[start:start+seg_len//2] = np.max((pid_segment_rec_scores[start:start+seg_len//2], segment_scores[:seg_len//2]), axis=0)
                pid_segment_pre_scores[start+seg_len//2:start+seg_len] = np.max((pid_segment_pre_scores[start+seg_len//2:start+seg_len], segment_scores[seg_len//2:]), axis=0)

            clip_person_scores_dict[person_id] = 0.3*pid_segment_rec_scores + 0.7*pid_segment_pre_scores

        clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))  # [persons, frames_score]
        clip_score = np.amax(clip_ppl_score_arr, axis=0)
        fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]
        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)
        dataset_score_ids_arr.append(fig_score_id)
        dataset_metadata_arr.append([scene_id, clip_id])
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr


def get_dataset_scores_by_sample_frame_score(scores, metadata, person_keys, max_clip=None, scene_id=None, args=None):
    """
    :param scores: [samples, T]
    :param metadata: [samples, 5] scene_id, clip_id, person_id, start_frame_idx, end_frame_idx, x_g, y_g, w, h
    :param max_clip:
    :param scene_id:
    :param args:
    :return:
    """
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    metadata_np = np.array(metadata)
    # per_frame_scores_root = 'data/pose/testing/test_frame_mask/'
    per_frame_scores_root = f'{args.data_dir}/pose/testing/test_frame_mask/'
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    # calculate per clip
    for clip in clip_list:
        scene_id, clip_id = clip.split('.')[0].split('_')
        if args.hr and f'{scene_id}_{clip_id}' in HUMAN_IRRELATED_CLIPS:
            continue
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        scene_id, clip_id = int(scene_id), int(clip_id)
        # find the sample index of the clip
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        clip_metadata = metadata[clip_metadata_inds]
        # person_idxs set (person in clip)
        clip_fig_idxs = set([arr[2] for arr in clip_metadata])
        clip_frame_num = clip_gt.shape[0]
        scores_zeros = np.zeros(clip_frame_num)  # [clip frames, ]
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

        seg_len = args.seg_len
        # scores_per_frame = [[0] for _ in range(clip_frame_num)]
        for person_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 2] == person_id))[0]
            pid_segment_scores = scores[person_metadata_inds] # [T,]
            pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds])
            # pid_scores = np.zeros((person_metadata_inds.shape[0], clip_frame_num))
            pid_segment_cnt = np.zeros(clip_frame_num)
            pid_segment_avg_scores = np.zeros(clip_frame_num)
            for segment_scores, start in zip(pid_segment_scores, pid_frame_inds):
                # pid_segment_cnt[start:start+seg_len] += 1
                pid_segment_avg_scores[start:start+seg_len] = np.max((pid_segment_avg_scores[start:start+seg_len],segment_scores),axis=0)

            #     pid_segment_avg_scores[start:start+seg_len] += segment_scores
            # pid_segment_cnt[pid_segment_cnt == 0] = 1
            # pid_segment_avg_scores = pid_segment_avg_scores / pid_segment_cnt
            clip_person_scores_dict[person_id] = pid_segment_avg_scores

        clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))  # [persons, frames_score]
        clip_score = np.amax(clip_ppl_score_arr, axis=0)
        fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]
        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)
        dataset_score_ids_arr.append(fig_score_id)
        dataset_metadata_arr.append([scene_id, clip_id])
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr


def get_dataset_scores_by_sample_avg_score(scores, metadata, person_keys, max_clip=None, scene_id=None, args=None):
    """
    :param scores: [samples, ]
    :param metadata: [samples, 5] scene_id, clip_id, person_id, start_frame_idx, end_frame_idx, x_g, y_g, w, h
    :param max_clip:
    :param scene_id:
    :param args:
    :return:
    """
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    metadata_np = np.array(metadata)
    # per_frame_scores_root = 'data/pose/testing/test_frame_mask/'
    per_frame_scores_root = f'{args.data_dir}/pose/testing/test_frame_mask/'
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    # calculate per clip
    for clip in clip_list:
        scene_id, clip_id = clip.split('.')[0].split('_')
        if args.hr and f'{scene_id}_{clip_id}' in HUMAN_IRRELATED_CLIPS:
            continue
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        scene_id, clip_id = int(scene_id), int(clip_id)
        # find the sample index of the clip
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        clip_metadata = metadata[clip_metadata_inds]
        # person_idxs set (person in clip)
        clip_fig_idxs = set([arr[2] for arr in clip_metadata])

        clip_score = np.zeros(clip_gt.shape[0])  # [clip frames, ]
        fig_score_id = np.zeros(clip_gt.shape[0])-1

        # clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

        clip_person_scores_dict = collections.defaultdict(int)

        # calculate anomaly score for each skeleton instance
        for person_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 2] == person_id))[0]
            pid_scores = scores[person_metadata_inds]

            clip_person_scores_dict[person_id] = np.mean(pid_scores)

            # pid_frame_inds = np.array([metadata[i][4] for i in person_metadata_inds])
            # clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores

        # calculate frame anomaly score
        clip_person_keys_dict = {int(key.split('_')[-1]): person_keys[key] for key in person_keys.keys() if clip.split('.')[0] in key}
        for frame_idx in range(clip_gt.shape[0]):
            frame_person_idxs = [p_idx for p_idx, p_keys in clip_person_keys_dict.items() if frame_idx in p_keys]
            if frame_person_idxs:
                max_idx = np.argmax([clip_person_scores_dict[idx] for idx in frame_person_idxs])
                clip_score[frame_idx] = clip_person_scores_dict[frame_person_idxs[max_idx]]
                fig_score_id[frame_idx] = frame_person_idxs[max_idx]

        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)
        dataset_score_ids_arr.append(fig_score_id)
        dataset_metadata_arr.append([scene_id, clip_id])
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr



def score_dataset(score_vals, metadata, person_keys, max_clip=None, scene_id=None, args=None):
    score_vals = np.array(score_vals)  # [samples, ]

    gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores(score_vals, metadata, max_clip, scene_id, args)

    # gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores_by_sample_avg_score(score_vals, metadata, person_keys, max_clip, scene_id, args)

    # gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores_by_sample_frame_score(score_vals, metadata, person_keys, max_clip, scene_id, args)

    # gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores_by_rec_add_pre_score(score_vals, metadata, person_keys, max_clip, scene_id, args)


    # normalize to 0,1 and draw
    normalized_scores = normalize_scores(scores_arr)

    # quantile_transform
    # quantile_transformed_arrs = quantile_transform_errors(scores_arr)
    # smooth
    normalized_and_smooth_scores = smooth_scores(normalized_scores, args.sigma)

    draw_anomaly_score_curve(normalized_and_smooth_scores, metadata_arr, gt_arr, cal_clip_roc_auc(gt_arr, normalized_and_smooth_scores), args.ckpt_dir.split('/')[2])

    # draw_anomaly_score_curve(normalized_and_smooth_scores, metadata_arr, gt_arr, cal_clip_roc_auc(gt_arr, normalized_and_smooth_scores), args.ckpt_dir.split('/')[2])

    # macro auc calculate
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    auc, shift, sigma = score_align(scores_np, gt_np, sigma=args.sigma)

    # micro auc calculate
    micro_auc = roc_auc_score(gt_np, np.concatenate(normalized_and_smooth_scores))
    print(f'micro_auc = {micro_auc}')
    return auc, shift, sigma


def get_dataset_scores(scores, metadata, max_clip=None, scene_id=None, args=None):
    """

    :param scores: [samples, ]
    :param metadata: [samples, 5] scene_id, clip_id, person_id, start_frame_idx, end_frame_idx
    :param max_clip:
    :param scene_id:
    :param args:
    :return:
    """
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    metadata_np = np.array(metadata)
    # per_frame_scores_root = 'data/pose/testing/test_frame_mask/'
    per_frame_scores_root = f'{args.data_dir}/pose/testing/test_frame_mask/'
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    # calculate per clip
    for clip in clip_list:
        scene_id, clip_id = clip.split('.')[0].split('_')
        if args.hr and f'{scene_id}_{clip_id}' in HUMAN_IRRELATED_CLIPS:
            continue
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        scene_id, clip_id = int(scene_id), int(clip_id)
        # find the sample index of the clip
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        clip_metadata = metadata[clip_metadata_inds]
        # person_idxs set (person in clip)
        clip_fig_idxs = set([arr[2] for arr in clip_metadata])
        scores_zeros = np.zeros(clip_gt.shape[0])  # [clip frames, ]
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}
        for person_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 2] == person_id))[0]
            pid_scores = scores[person_metadata_inds]
            pid_frame_inds = np.array([metadata[i][4] for i in person_metadata_inds])
            clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores

        clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))  # [persons, frames_score]
        clip_score = np.amax(clip_ppl_score_arr, axis=0)
        fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]
        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)
        dataset_score_ids_arr.append(fig_score_id)
        dataset_metadata_arr.append([scene_id, clip_id])
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr


def score_align(scores_np, gt, seg_len=12, sigma=40):
    # TODO score shift
    # scores_shifted = np.zeros_like(scores_np)
    shift = seg_len + (seg_len // 2) - 1
    # scores_shifted[shift:] = scores_np[:-shift]


    # TODO normalized calculate auc


    # scores_smoothed = scores_np
    scores_smoothed = gaussian_filter1d(scores_np, sigma)
    auc = roc_auc_score(gt, scores_smoothed)
    return auc, shift, sigma


def avg_scores_by_trans(scores, gt, num_transform=5, ret_first=False):
    """

    :param scores:
    :param gt:
    :param num_transform:
    :param ret_first:
    :return:
    """
    score_mask, scores_by_trans, scores_tavg = dict(), dict(), dict()
    gti = {'normal': 1, 'abnormal': 0}
    for k, gt_val in gti.items():
        score_mask[k] = scores[gt == gt_val]
        scores_by_trans[k] = score_mask[k].reshape(-1, num_transform)
        scores_tavg[k] = scores_by_trans[k].mean(axis=1)

    gt_trans_avg = np.concatenate([np.ones_like(scores_tavg['normal'], dtype=np.int),
                                   np.zeros_like(scores_tavg['abnormal'], dtype=np.int)])
    scores_trans_avg = np.concatenate([scores_tavg['normal'], scores_tavg['abnormal']])
    if ret_first:
        scores_first_trans = dict()
        for k, v in scores_by_trans.items():
            scores_first_trans[k] = v[:, 0]
        scores_first_trans = np.concatenate([scores_first_trans['normal'], scores_first_trans['abnormal']])
        return scores_trans_avg, gt_trans_avg, scores_first_trans
    else:
        return scores_trans_avg, gt_trans_avg


if __name__ == '__main__':
    pass