import os
import random
import collections
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.agcn import Model as AGCN

from utils.data_utils import trans_list
from utils.optim_utils.optim_init import init_optimizer, init_scheduler
from utils.pose_seg_dataset import PoseSegDataset
from utils.pose_ad_argparse import init_parser, init_sub_args
from utils.train_utils import Trainer, csv_log_dump


def main():
    parser = init_parser()
    args = parser.parse_args()
    log_dict = collections.defaultdict(int)

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args, res_args = init_sub_args(args)
    print(args)

    dataset, loader = get_dataset_and_loader(args)

    model = AGCN(graph='graph.graph.Graph', seq_len=args.seg_len,)
    loss = nn.MSELoss()

    optimizer_f = init_optimizer(args.optimizer, lr=args.lr)
    scheduler_f = init_scheduler(args.sched, lr=args.lr, epochs=args.epochs)
    trainer = Trainer(args, model, loss, loader['train'], loader['test'], optimizer_f=optimizer_f,
                      scheduler_f=scheduler_f)

    print(f"train dataset lens {len(loader['train'].dataset)}")
    print(f"test dataset lens {len(loader['test'].dataset)}")
    fn, log_dict['loss'] = trainer.train(args=args)
    print(f"model loss {log_dict['loss']}")

    output_arr = trainer.test(args.epochs, ret_output=True, args=args)
    # print(loss_arr)

    # max_clip = 5 if args.debug else None
    # auc, dp_shift, dp_sigma = score_dataset(dp_scores_tavg, metadata, max_clip=max_clip)

    # Logging and recording results
    # print("Done with {} AuC for {} samples and {} trans".format(dp_auc, dp_scores_tavg.shape[0], args.num_transform));
    # log_dict['auc'] = 100 * auc
    csv_log_dump(args, log_dict)


def get_dataset_and_loader(args):

    transform_list = trans_list[:args.num_transform]

    dataset_args = {'transform_list': transform_list, 'debug': args.debug, 'headless': args.headless,
                    'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale, 'seg_len': args.seg_len,
                    'return_indices': True, 'return_metadata': True}

    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}

    dataset, loader = dict(), dict()
    for split in ['train', 'test']:
        dataset_args['seg_stride'] = args.seg_stride if split is 'train' else 1  # No strides for test set
        dataset_args['train_seg_conf_th'] = args.train_seg_conf_th if split is 'train' else 0.0
        dataset[split] = PoseSegDataset(args.pose_path[split], **dataset_args)
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    return dataset, loader


def save_result_npz(args, scores, scores_tavg, metadata, sfmax_maxval, auc, dp_auc=None):
    debug_str = '_debug' if args.debug else ''
    auc_int = int(1000 * auc)
    dp_auc_str = ''
    if dp_auc is not None:
        dp_auc_int = int(1000 * dp_auc)
        dp_auc_str = '_dp{}'.format(dp_auc_int)
    auc_str = '_{}'.format(auc_int)
    res_fn = args.ae_fn.split('.')[0] + '_res{}{}{}.npz'.format(dp_auc_str, auc_str, debug_str)
    res_path = os.path.join(args.ckpt_dir, res_fn)
    np.savez(res_path, scores=scores, sfmax_maxval=sfmax_maxval, args=args, metadata=metadata,
             scores_tavg=scores_tavg, dp_best=dp_auc)


if __name__ == '__main__':
    main()

