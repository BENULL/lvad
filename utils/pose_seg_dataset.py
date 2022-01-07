import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import normalize_pose

from utils.patch_utils import gen_clip_seg_data_np


class PoseSegDataset(Dataset):
    """
    Generates a dataset with two objects, an np array holding sliced pose sequences
    and an object array holding file name, person index and start time for each sliced seq


    If path_to_patches is provided uses pre-extracted patches. If lmdb_file or vid_dir are
    provided extracts patches from them, while hurting performance.
    """
    def __init__(self, path_to_json_dir,
                 transform_list=None,
                 return_indices=False, return_metadata=False, debug=False,
                 dataset_clips=None,
                 **dataset_args):
        super().__init__()
        self.path_to_json = path_to_json_dir
        self.headless = dataset_args.get('headless', False)
        self.debug = debug
        num_clips = 5 if debug else None
        if dataset_clips is not None:
            num_clips = dataset_clips
        self.return_indices = return_indices
        self.return_metadata = return_metadata

        if transform_list is None or transform_list == []:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(transform_list)
        self.transform_list = transform_list
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        self.segs_data_np, self.segs_meta, self.person_keys = gen_dataset(path_to_json_dir, num_clips=num_clips,
                                                                          ret_keys=True, **dataset_args)
        # Convert person keys to ints
        self.person_keys = {k: [int(i) for i in v] for k, v in self.person_keys.items()}
        self.segs_meta = np.array(self.segs_meta)
        self.metadata = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

    def __getitem__(self, index):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = index // self.num_samples
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
        else:
            sample_index = index
            data_transformed = data_numpy = np.array(self.segs_data_np[index])
            trans_index = 0  # No transformations

        seg_metadata = self.segs_meta[sample_index]
        ret_arr = [data_transformed, trans_index]
        if self.return_metadata:
            ret_arr += [seg_metadata]
        if self.return_indices:
            ret_arr += [index]
        return ret_arr

    def __len__(self):
        return self.num_transform * self.num_samples


def gen_dataset(person_json_root, num_clips=None, normalize_pose_segs=True,
                kp18_format=True, ret_keys=False, **dataset_args):
    segs_data_np = []
    segs_meta = []
    person_keys = dict()
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    seg_len = dataset_args.get('seg_len', 12)
    vid_res = dataset_args.get('vid_res', [856, 480])
    headless = dataset_args.get('headless', False)
    seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)

    dir_list = os.listdir(person_json_root)
    json_list = sorted([fn for fn in dir_list if fn.endswith('.json')])
    if num_clips:
        json_list = json_list[:num_clips]  # For debugging purposes
    for person_dict_fn in json_list:
        scene_id, clip_id = person_dict_fn.split('_')[:2]
        clip_json_path = os.path.join(person_json_root, person_dict_fn)
        with open(clip_json_path, 'r') as f:
            clip_dict = json.load(f)
        clip_segs_data_np, clip_segs_meta, clip_keys = gen_clip_seg_data_np(clip_dict, start_ofst, seg_stride,
                                                                            seg_len, scene_id=scene_id,
                                                                            clip_id=clip_id, ret_keys=ret_keys)
        segs_data_np.append(clip_segs_data_np)
        segs_meta += clip_segs_meta
        person_keys = {**person_keys, **clip_keys}
    segs_data_np = np.concatenate(segs_data_np, axis=0)
    if normalize_pose_segs:
        segs_data_np = normalize_pose(segs_data_np, vid_res=vid_res, **dataset_args)
    if kp18_format and segs_data_np.shape[-2] == 17:
        segs_data_np = keypoints17_to_coco18(segs_data_np)
    if headless:
        segs_data_np = segs_data_np[:, :, :14]
    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)
    if seg_conf_th > 0.0:
        segs_data_np, segs_meta = seg_conf_th_filter(segs_data_np, segs_meta, seg_conf_th)
    if ret_keys:
        return segs_data_np, segs_meta, person_keys
    else:
        return segs_data_np, segs_meta



def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18


# TODO ?seg_conf_th_filter
def seg_conf_th_filter(segs_data_np, segs_meta, seg_conf_th=2.0):
    seg_len = segs_data_np.shape[2]
    conf_vals = segs_data_np[:, 2]
    sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len
    seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
    seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
    return seg_data_filt, seg_meta_filt


if __name__ == '__main__':
    dataset_args = {'transform_list': [], 'debug': True, 'headless': False, 'return_indices': True, 'return_metadata': True}
    loader_args = {'batch_size': 6, 'num_workers': 0, 'pin_memory': False}
    poseDataset = PoseSegDataset('../data/pose/training/tracked_person/', **dataset_args)
    loader = DataLoader(poseDataset, **loader_args, shuffle=False)
    for itern, data_arr in enumerate(loader):
        print(data_arr)


