import os
import pickle
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn import metrics

ml_names = ['melody', 'articulation', 'rhythm_complexity', 'rhythm_stability', 'dissonance', 'tonal_stability', 'minorness']

def list_files_deep(dir_path, full_paths=False, filter_ext=None):
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(dir_path, '')):
        if len(filenames) > 0:
            for f in filenames:
                if full_paths:
                    all_files.append(os.path.join(dirpath, f))
                else:
                    all_files.append(f)

    if filter_ext is not None:
        return [f for f in all_files if os.path.splitext(f)[1] in filter_ext]
    else:
        return all_files


def save(model, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    try:
        torch.save(model.module.state_dict(), path)
    except AttributeError:
        torch.save(model.state_dict(), path)


def pickledump(data, fp):
    d = os.path.dirname(fp)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(fp, 'wb') as f:
        pickle.dump(data, f)


def pickleload(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def dumptofile(data, fp):
    d = os.path.dirname(fp)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(fp, 'w') as f:
        print(data, file=f)


def print_dict(dict, round):
    for k, v in dict.items():
        print(f"{k}:{np.round(v, round)}")


def log_dict(logger, dict, round=None, delimiter='\n'):
    log_str = ''
    for k, v in dict.items():
        if isinstance(round, int):
            try:
                log_str += f"{k}: {np.round(v, round)}{delimiter}"
            except:
                log_str += f"{k}: {v}{delimiter}"
        else:
            log_str += f"{k}: {v}{delimiter}"
    logger.info(log_str)


def load_model(model_weights_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_weights_path))
    else:
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()


def inf(dl):
    """Infinite dataloader"""
    while True:
        for x in iter(dl): yield x


def choose_rand_index(arr, num_samples):
    return np.random.choice(arr.shape[0], num_samples, replace=False)


def compute_metrics(y, y_hat, metrics_list, **kwargs):
    metrics_res = {}
    for metric in metrics_list:
        Y, Y_hat = y, y_hat
        if metric in ['rocauc-macro', 'rocauc']:
            metrics_res[metric] = metrics.roc_auc_score(Y, Y_hat, average='macro')
        if metric == 'rocauc-micro':
            metrics_res[metric] = metrics.roc_auc_score(Y, Y_hat, average='micro')
        if metric in ['prauc-macro', 'prauc']:
            metrics_res[metric] = metrics.average_precision_score(Y, Y_hat, average='macro')
        if metric == 'prauc-micro':
            metrics_res[metric] = metrics.average_precision_score(Y, Y_hat, average='micro')

        if metric == 'corr_avg':
            corr, pval = [], []
            for i in range(kwargs.get("num_cols", 7)):
                c, p = pearsonr(Y[:, i], Y_hat[:, i])
                corr.append(c)
                pval.append(p)
            metrics_res['corr_avg'] = np.mean(corr)
            metrics_res['pval_avg'] = np.mean(pval)

        if metric == 'corr':
            corr, pval = [], []
            for i in range(kwargs.get("num_cols", 7)):
                c, p = pearsonr(Y[:, i], Y_hat[:, i])
                corr.append(c)
                pval.append(p)
            metrics_res['corr'] = corr
            metrics_res['pval'] = pval

        if metric == 'mae':
            metrics_res[metric] = metrics.mean_absolute_error(Y, Y_hat)

        if metric == 'r2':
            metrics_res[metric] = metrics.r2_score(Y, Y_hat)

        if metric == 'mse':
            metrics_res[metric] = metrics.mean_squared_error(Y, Y_hat)

    return metrics_res


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, save_dir='.', saved_model_name="model_chkpt",
                 condition='minimize'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.saved_model_name = saved_model_name
        self.save_path = os.path.join(self.save_dir, self.saved_model_name + '.pt')
        self.condition = condition
        assert condition in ['maximize', 'minimize']
        self.metric_best = np.Inf if condition == 'minimize' else -np.Inf

    def __call__(self, metric, model):

        score = metric if self.condition == 'maximize' else -metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Metric improved ({self.condition}) ({self.metric_best:.6f} --> {metric:.6f}).  Saving model to {os.path.join(self.save_dir, self.saved_model_name + ".pt")}')
        torch.save(model.state_dict(), self.save_path)
        self.metric_best = metric
