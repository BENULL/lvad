import math
import torch
import numpy as np


def get_aff_trans_mat(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False):
    """
    Generate affine transfomation matrix (torch.tensor type) for transforming pose sequences
    :rot is given in degrees
    """
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))
    flip_mat = torch.eye(3, dtype=torch.float32)
    if flip:
        flip_mat[0, 0] = -1.0
    trans_scale_mat = torch.tensor([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=torch.float32)
    rot_mat = torch.tensor([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=torch.float32)
    aff_mat = torch.matmul(rot_mat, trans_scale_mat)
    aff_mat = torch.matmul(flip_mat, aff_mat)
    return aff_mat


def apply_pose_transform(pose, trans_mat):
    """ Given a set of pose sequences of shape (Channels, Time_steps, Vertices, M[=num of figures])
    return its transformed form of the same sequence. 3 Channels are assumed (x, y, conf) """

    # We isolate the confidence vector, replace with ones, than plug back after transformation is done
    conf = np.expand_dims(pose[2], axis=0)
    ones_vec = np.ones_like(conf)
    pose_w_ones = np.concatenate([pose[:2], ones_vec], axis=0)
    if len(pose.shape) == 3:
        einsum_str = 'ktv,ck->ctv'
    else:
        einsum_str = 'ktvm,ck->ctvm'
    pose_transformed_wo_conf = np.einsum(einsum_str, pose_w_ones, trans_mat)
    pose_transformed = np.concatenate([pose_transformed_wo_conf[:2], conf], axis=0)
    return pose_transformed


class PoseTransform(object):
    """ A general class for applying transformations to pose sequences, empty init returns identity """

    def __init__(self, sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, trans_mat=None):
        """ An explicit matrix overrides all parameters"""
        if trans_mat is not None:
            self.trans_mat = trans_mat
        else:
            self.trans_mat = get_aff_trans_mat(sx, sy, tx, ty, rot, flip)

    def __call__(self, x):
        x = apply_pose_transform(x, self.trans_mat)
        return x


trans_list = [
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=0, flip=False),  # 0
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=0, flip=True),  # 3
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=90, flip=False),  # 6
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=90, flip=True),  # 9
    PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=45, flip=False),  # 12
]

def normalize_pose_by_vid(pose_data, seg_meta, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    :param vid_res:
    :param symm_range:
    :return:
    """
    vid_res = kwargs.get('vid_res', [856, 480])
    symm_range = kwargs.get('symm_range', True) # Means shift data to [-1, 1] range
    sub_mean = kwargs.get('sub_mean', False)
    scale = kwargs.get('scale', False)
    scale_proportional = kwargs.get('scale_proportional', False)

    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    # pose_data_centered = pose_data_normalized

    # if symm_range:  # Means shift data to [-1, 1] range
    #     pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1

    # pose_data_zero_mean = pose_data_centered
    # if sub_mean or scale or scale_proportional:  # Inner frame scaling requires mean subtraction
    #     # pose_data_zero_mean = pose_data_centered
    #     mean_kp_val = np.mean(pose_data_zero_mean[..., :2], (1, 2))
    #     pose_data_zero_mean[..., :2] -= mean_kp_val[:, None, None, :]
    #
    # max_kp_xy = np.max(np.abs(pose_data_centered[..., :2]), axis=(1, 2))
    # max_kp_coord = max_kp_xy.max(axis=1)
    #
    # pose_data_scaled = pose_data_zero_mean
    # if scale:
    #     # Scale sequence to maximize the [-1,1] frame
    #     # Removes average position from all keypoints, than scales in x and y to fill the frame
    #     # Loses body proportions
    #     pose_data_scaled[..., :2] = pose_data_scaled[..., :2] / max_kp_xy[:, None, None, :]
    #
    # elif scale_proportional:
    #     # Same as scale but normalizes by the same factor
    #     # (smaller axis, i.e. divides by larger fraction value)
    #     # Keeps propotions
    #     pose_data_scaled[..., :2] = pose_data_scaled[..., :2] / max_kp_coord[:, None, None, None]

    return pose_data_normalized, seg_meta


def normalize_pose_by_video(pose_data):
    vid_res = torch.Tensor([640, 360]).to(pose_data.device)
    pose_data_normalized = pose_data.clone()
    pose_data_normalized = pose_data_normalized.permute(0, 2, 3, 1)

    pose_data_normalized[..., :2] = pose_data_normalized[..., :2] / vid_res[None, None, None, :]
    pose_data_normalized = pose_data_normalized.permute(0, 3, 1, 2)
    return pose_data_normalized

# def re_normalize_pose_by_video(normalized):

def normalize_pose(pose_data, segs_meta=None, **kwargs):
    """
    Normalize keypoint values by bounding box
    :param pose_data: Formatted as [N, C, T, V], e.g. (Batch=64, Frames=12, 18, 3)
    :return:
    """

    # N, C, T, V to N, T, V, C
    pose_xy_local = pose_data.clone()
    pose_xy_local = pose_xy_local.permute(0, 2, 3, 1)

    max_kp_xy = torch.max(pose_xy_local[..., :2], dim=2)[0]
    min_kp_xy = torch.min(pose_xy_local[..., :2], dim=2)[0]

    xy_global = (max_kp_xy + min_kp_xy)/2
    bounding_box_wh = max_kp_xy - min_kp_xy

    # xy_global = xy_global.astype(int)
    # bounding_box_wh = bounding_box_wh.astype(int)
    pose_xy_local[..., :2] = (pose_xy_local[..., :2] - xy_global[:, :, None, :]) / bounding_box_wh[:, :, None, :]

    pose_xy_perceptual = pose_data.clone()
    pose_xy_perceptual = pose_xy_perceptual.permute(0, 2, 3, 1)
    pose_xy_perceptual[..., :2] = (pose_xy_perceptual[..., :2] - min_kp_xy[:, :, None, :]) / bounding_box_wh[:, :, None, :]


    pose_xy_local = pose_xy_local.permute(0, 3, 1, 2)
    pose_xy_perceptual = pose_xy_perceptual.permute(0, 3, 1, 2)

    # segs_meta add [x_g, y_g, w, h]
    # segs_meta = np.concatenate((segs_meta, xy_global, bounding_box_wh), axis=1)

    # pose_normalized_by_bbox = pose_data.clone()
    # pose_normalized_by_bbox[..., :2] = (pose_data[..., :2] - min_kp_xy[:, :, None, :]) / bounding_box_wh[:, :, None, :]

    return pose_xy_local, pose_xy_perceptual, xy_global, bounding_box_wh

def re_normalize_pose(local_pose, normalized_pose, xy_global, bounding_box_wh):
    """
    :param normalized_pose: [N, C, T, V]
    :return:
    """

    # if len(normalized_pose.shape()) == 2:
    #     normalized_pose = normalized_pose[None]
    re_normalized_pose = normalized_pose.clone()
    re_normalized_pose = re_normalized_pose.permute(0, 2, 3, 1)
    re_normalized_local_pose = local_pose.clone()
    re_normalized_local_pose = re_normalized_local_pose.permute(0, 2, 3, 1)

    # xy_global = np.array(meta[:2])
    # bounding_box_wh = np.array(meta[2:])

    min_kp_xy = (2*xy_global-bounding_box_wh)/2

    re_normalized_local_pose[..., :2] = re_normalized_local_pose[..., :2] * bounding_box_wh[:, :, None, :] + xy_global[:, :, None, :]
    re_normalized_pose[..., :2] = re_normalized_pose[..., :2] * bounding_box_wh[:, :, None, :] + min_kp_xy[:, :, None, :]

    re_normalized_pose = re_normalized_pose.permute(0, 3, 1, 2)
    re_normalized_local_pose = re_normalized_local_pose.permute(0, 3, 1, 2)

    return re_normalized_local_pose, re_normalized_pose
