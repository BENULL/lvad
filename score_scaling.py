import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity
import scipy.stats as stats

from motion_stat import load_exp_data
from utils.scoring_utils import get_dataset_scores_by_rec_add_pre_score, normalize_scores, smooth_scores, score_align


class ScoreNormalization:
    def __init__(self, method="KDE", options=None):
        self.method = method
        self.options = {} if options is None else options
        self.name = "_".join([method] + ["-".join([str(key), str(value)]) for key, value in self.options.items()])
        if self.method == "KDE":
            kernel = self.options.get("kernel", "gaussian")
            bandwidth = self.options.get("bandwidth", 0.75)
            self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        elif self.method == "histogram":
            raise ("Not implemented method {}".format(self.method))
        elif self.method == "gamma":
            self.k = self.options.get("k", 1)
            self.loc = self.options.get("loc", 0)
            self.theta = self.options.get("theta", 1.5)
        elif self.method == "chi2":
            self.df = self.options.get("df", 2)
            self.loc = self.options.get("loc", 0)
            self.scale = self.options.get("scale", 0.5)
        else:
            raise ("Invalid method {}".format(self.method))

    def fit(self, X):
        if self.method == "KDE":
            self.kde.fit(X)
        elif self.method == "histogram":
            raise ("Not implemented method {}".format(self.method))
        elif self.method == "gamma":
            self.fit_k, self.fit_loc, self.fit_theta = stats.gamma.fit(X, self.k, loc=self.loc, scale=self.theta)
        elif self.method == "chi2":
            self.fit_df, self.fit_loc, self.fit_scale = stats.chi2.fit(X, self.df, loc=self.loc, scale=self.scale)
            pass
        else:
            raise ("Invalid method {}".format(self.method))

    def score(self, x):
        if self.method == "KDE":
            return self.kde.score_samples(x)
        elif self.method == "gamma":
            return 1 - stats.gamma.cdf(x, self.fit_k, self.fit_loc, self.fit_theta)
        elif self.method == "chi2":
            return 1 - stats.chi2.cdf(x, self.fit_df, self.fit_loc, self.fit_scale)
        elif self.method == "histogram":
            raise ("Not implemented method {}".format(self.method))
        else:
            raise ("Invalid method {}".format(self.method))

    transform = score

    def get_fit_params_string(self):
        if self.method == "KDE":
            return ""
        elif self.method == "gamma":
            return "fit_k %3.1f fit_loc %3.1f fit_theta %3.1f" % (self.fit_k, self.fit_loc, self.fit_theta)
        elif self.method == "chi2":
            return "fit_df %s, fit_loc %s, fit_scale %s" % (self.fit_df, self.fit_loc, self.fit_scale)
        elif self.method == "histogram":
            return ""
        else:
            raise ("Invalid method {}".format(self.method))

    def get_fit_params(self):
        if self.method == "KDE":
            return ""
        elif self.method == "gamma":
            return self.fit_k, self.fit_loc, self.fit_theta
        elif self.method == "chi2":
            return self.fit_df, self.fit_loc, self.fit_scale
        elif self.method == "histogram":
            return ""
        else:
            raise ("Invalid method {}".format(self.method))


def normalizing_lstm_autoencoder(model, all_training_score, all_testing_score, training_sample_rate=1, ):
    test_normed_score = {}
    train_normed_score = {}
    normaliz_test_scores = []
    for scene_id in all_testing_score.keys():
        training_score = np.array(all_training_score[scene_id])
        testing_score = np.array(all_testing_score[scene_id])

        is_non_zero_training_score = training_score > 0.0
        non_zero_training_score = training_score[is_non_zero_training_score].reshape(-1, 1)
        non_zero_training_score = non_zero_training_score[0::training_sample_rate]

        is_non_zero_testing_score = testing_score > 0.0
        non_zero_testing_score = testing_score[is_non_zero_testing_score].reshape(-1, 1)

        model.fit(non_zero_training_score)
        # train_normed_score[scene_id] = model.score(non_zero_training_score)
        test_normed_score[scene_id] = model.score(non_zero_testing_score)
        # training_score = training_score.ravel()
        # training_score[is_non_zero_training_score] = train_normed_score[scene_id].ravel()

        testing_score = testing_score.ravel()
        testing_score[is_non_zero_testing_score] = test_normed_score[scene_id].ravel()

        normaliz_test_scores.extend(testing_score)

    print(model.name + '_' + model.get_fit_params_string())

    return normaliz_test_scores


if __name__ == '__main__':
    training_file = "/root/VAD/lvad/data/exp_dir/Apr11_2046/checkpoints/training_error_by_scene.npz"
    # testing_file = "/root/VAD/lvad/data/exp_dir/Apr11_1454/checkpoints/testing_error_by_scene.npz"
    all_training_score = np.load(training_file, allow_pickle=True)['training_error_by_scene'].item()
    # all_testing_score = np.load(testing_file, allow_pickle=True)['testing_error_by_scene'].item()

    npz_path = './data/exp_dir/Apr11_2046/checkpoints/res_7146_.npz'
    args, output_arr, rec_loss_arr, loader, dataset = load_exp_data(npz_path)
    score_vals = np.array(rec_loss_arr)  # [samples, ]

    gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores_by_rec_add_pre_score(score_vals, dataset.metadata, None, None, None, args)

    from collections import defaultdict

    testing_error_by_scene = defaultdict(list)
    for scores, meta in zip(scores_arr, metadata_arr):
        testing_error_by_scene[meta[0]].extend(scores)

    all_testing_score = testing_error_by_scene

    models = [ScoreNormalization(method="KDE", options={"kernel": "gaussian", "bandwidth": 0.01}),
              ScoreNormalization(method="KDE", options={"kernel": "gaussian", "bandwidth": 0.001}),
              ScoreNormalization(method="KDE", options={"kernel": "gaussian", "bandwidth": 0.0001}),
              ScoreNormalization(method="gamma", options={}),
              ScoreNormalization(method="chi2", options={})]

    normalized_scores = normalize_scores(scores_arr)

    # smooth
    normalized_and_smooth_scores = smooth_scores(normalized_scores, 12)

    # macro auc calculate
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    macro_auc, _, _ = score_align(scores_np, gt_np, sigma=12)

    # micro auc calculate
    micro_auc = roc_auc_score(gt_np, np.concatenate(normalized_and_smooth_scores))
    print(f'origin_micro_auc = {micro_auc}')
    print(f'origin_macro_auc = {macro_auc}')

    for model in models[:1]:
        normaliz_test_scores = normalizing_lstm_autoencoder(model, all_training_score, all_testing_score,
                                                            training_sample_rate=1)
        auc = roc_auc_score(gt_np, normaliz_test_scores)
        print(f'model_auc = {auc}')

