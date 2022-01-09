import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from sklearn import mixture
from joblib import dump, load


def score_dataset(score_vals, metadata, max_clip=None, scene_id=None):
    score_vals = np.array(score_vals)
    gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores(score_vals, metadata, max_clip, scene_id)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    auc, shift, sigma = score_align(scores_np, gt_np)
    return auc, shift, sigma


def get_dataset_scores(scores, metadata, max_clip=None, scene_id=None):
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    metadata_np = np.array(metadata)
    per_frame_scores_root = 'data/pose/testing/test_frame_mask/'
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    for clip in clip_list:
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        scene_id, clip_id = [int(i) for i in clip.split('.')[0].split('_')]
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        clip_metadata = metadata[clip_metadata_inds]
        # person_idxs
        clip_fig_idxs = set([arr[2] for arr in clip_metadata])
        scores_zeros = np.zeros(clip_gt.shape[0])
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}
        for person_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 2] == person_id))[0]
            pid_scores = scores[person_metadata_inds]

            pid_frame_inds = np.array([metadata[i][4] for i in person_metadata_inds])
            clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores

        clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
        clip_score = np.amax(clip_ppl_score_arr, axis=0)
        fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]
        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)
        dataset_score_ids_arr.append(fig_score_id)
        dataset_metadata_arr.append([scene_id, clip_id])
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr


def score_align(scores_np, gt, seg_len=12, sigma=40):
    # scores_shifted = np.zeros_like(scores_np)
    shift = seg_len + (seg_len // 2) - 1
    # scores_shifted[shift:] = scores_np[:-shift]

    # scores_smoothed = gaussian_filter1d(scores_np, sigma)
    auc = roc_auc_score(gt, scores_np)
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
