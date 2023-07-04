import numpy as np
from scipy.stats import spearmanr
from scipy.stats.mstats import gmean
from scipy.stats import hmean
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import recall_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import mean_absolute_percentage_error as mape_skl
from copy import copy


def check_arrays(y_true, y_pred, dtype=None):
    if dtype is None:
        dtype = type(y_true[0])
    if isinstance(y_true, list):
        y_true = np.asarray(y_true, dtype=dtype)
    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred, dtype=dtype)
    return y_true, y_pred


def check_n_replace_zero(y, epsilon='eps64'):
    y_no_zero = copy(y)
    if 0 in y:
        print('Warning: when y_true contains a zero value, metric is ill defined.')
        index_eq_0 = np.where(y == 0)
        index_neq_0 = np.where(y != 0)
        if isinstance(epsilon, str):
            if epsilon == 'min':
                epsilon = np.min(y[index_neq_0])
            elif 'eps' in epsilon:
                prec = '64'
                if '16' in epsilon or '32' in epsilon or '64' in epsilon:
                    prec = epsilon[3:]
                epsilon = float(np.finfo(np.__dict__['float' + prec]).eps)
        assert isinstance(epsilon, int) or isinstance(epsilon, float), 'Error: epsilon arg value is not properly set.'
        y_no_zero[index_eq_0] = epsilon
    return y_no_zero


def compute_mean_absolute_error(y_true, y_pred):
    y_true, y_pred = check_arrays(y_true, y_pred)
    return mean_absolute_error(y_true, y_pred)


# DIR
def compute_g_mean(y_true, y_pred, verbose=True):
    ytu = np.unique(y_true)
    ypu = np.unique(y_pred)

    y_true, y_pred = check_arrays(y_true, y_pred)
    if len(y_true) == len(y_pred) == 0:
        return np.nan
    # TODO: aggiungere controllo sulle label, devono essere ordinali di tipo 0-based
    if (y_true < 0).any() or (y_pred < 0).any():
        if verbose:
            print('Warning: not 0-based ordinal labels')
        y_true = np.where(y_true < 0, 0, y_true)
        y_pred = np.where(y_pred < 0, 0, y_pred)
    # gmean_ret = np.mean(geometric_mean_score(y_true, y_pred, average=None))
    # return gmean_ret
    return compute_g_mean_multiclass(y_true, y_pred)


def compute_accuracy(y_true, y_pred, verbose=True):
    y_true, y_pred = check_arrays(y_true, y_pred)
    if len(y_true) == len(y_pred) == 0:
        return np.nan
    return accuracy_score(y_true, y_pred)


def compute_balaccuracy(y_true, y_pred, verbose=True):
    y_true, y_pred = check_arrays(y_true, y_pred)
    if len(y_true) == len(y_pred) == 0:
        return np.nan
    return balanced_accuracy_score(y_true, y_pred)


def compute_g_mean_multiclass(y_true, y_pred, verbose=True):
    """
    Compute g-mean as the geometric mean of the recalls of all classes
    """
    recalls = recall_score(y_true, y_pred, average=None)
    nonzero_recalls = recalls[recalls != 0]
    is_zero_recall = False
    unique_y_true = list(set(y_true))
    # print('unique y true',unique_y_true)
    for i, recall in enumerate(recalls):
        # print(i,recall)
        if recall == 0 and i in unique_y_true:
            is_zero_recall = True
    if is_zero_recall:
        if verbose:
            print('Warning: the returned gmean is computed on recalls with a ground-truth')
        gmean_ret = gmean(recalls)
    else:
        gmean_ret = gmean(nonzero_recalls)
    return gmean_ret


# RMSE
def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = check_arrays(y_true, y_pred)
    if len(y_true) == len(y_pred) == 0:
        return np.nan
    return mean_squared_error(y_true, y_pred, squared=False)


# NRMSE
def normalized_root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = check_arrays(y_true, y_pred)
    if len(y_true) == len(y_pred) == 0:
        return np.nan
    range = np.max(y_true) - np.min(y_true)
    assert range != 0, 'Error: metric is severely ill defined.'
    return root_mean_squared_error(y_true, y_pred) / range


# MAPE
def mean_absolute_percentage_error(y_true, y_pred, epsilon='eps64', perc=True):
    y_true, y_pred = check_arrays(y_true, y_pred)
    if len(y_true) == len(y_pred) == 0:
        return np.nan
    y_true_no_zero = check_n_replace_zero(y_true, epsilon)
    mape = np.nanmean(np.abs((y_true - y_pred) / y_true_no_zero))
    return mape * 100 if perc else mape


# sMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon='eps64', perc=True, factor=1.):
    y_true, y_pred = check_arrays(y_true, y_pred)
    if len(y_true) == len(y_pred) == 0:
        return np.nan
    y_sum = check_n_replace_zero(np.abs(y_true) + np.abs(y_pred), epsilon)
    smape = np.nanmean(np.abs((y_true - y_pred)) / (y_sum / factor))
    return smape * 100 if perc else smape


# MAPE2
def mean_absolute_percentage_error_2(y_true, y_pred):
    y_true, y_pred = check_arrays(y_true, y_pred)
    if len(y_true) == len(y_pred) == 0:
        return np.nan
    return mape_skl(y_true, y_pred)


# https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/
# SRCC
def spearman_rank_correlation_coefficient(y_true, y_pred, alpha=.05):
    y_true, y_pred = check_arrays(y_true, y_pred)
    if len(y_true) == len(y_pred) == 0:
        return np.nan
    coef, p = spearmanr(y_true, y_pred)
    # if p > alpha:
    #     print('Warning: the p-value is over alpha (fail to reject H0). p-value=%.4f, alpha=%.2f' % (p, alpha))
    return coef


def hierarchical_precision_score(y_true, y_pred, dependencies, average=None):
    """
    dependencies: dictionary of lists, with each key referring to a class and lists are [ancestors] (w/o the root).
    If not set (TODO), the dependencies will be inferred from the y_true or y_pred vectors, otherwise dependencies are
    applied to both y_true and y_pred.
    e.g. if the hierarchy is
          /-> C
    A -> B -> D
     \-> E -> F -> G
    dependencies will be dict([(B, []), (C, [B]), (D, [B]), (E, []), (F, [E]), (G, [F, E])])
    """
    y_true, y_pred = check_arrays(y_true, y_pred, dtype=None)
    assert average in ['macro', 'micro', None], 'average "%s" is not available.' % average
    classes, classes_counts = np.unique(np.concatenate((y_true, y_pred)), return_counts=True)
    hp = []
    for c in classes:
        class_index = np.where(y_pred == c)[0]
        num = den = 0
        for index in class_index:
            a_true = dependencies[y_true[index]] + [y_true[index]]
            a_pred = dependencies[y_pred[index]] + [y_pred[index]]
            num += len([p for p in a_pred if p in a_true])
            den += len(a_pred)
        hp.append(num / den if den != 0 else 0)
    if average is None:
        return hp
    elif average == 'macro':
        return np.mean(hp)
    else:  # average == 'micro'
        return np.average(hp, weights=classes_counts)

def hierarchical_recall_score(y_true, y_pred, dependencies, average=None):
    """
    dependencies: dictionary of lists, with each key referring to a class and lists are [ancestors] (w/o the root).
    If not set (TODO), the dependencies will be inferred from the y_true or y_pred vectors, otherwise dependencies are
    applied to both y_true and y_pred.
    e.g. if the hierarchy is
          /-> C
    A -> B -> D
     \-> E -> F -> G
    dependencies will be dict([(B, []), (C, [B]), (D, [B]), (E, []), (F, [E]), (G, [F, E])])
    """
    y_true, y_pred = check_arrays(y_true, y_pred, dtype=None)
    assert average in ['macro', 'micro', None], 'average "%s" is not available.' % average
    classes, classes_counts = np.unique(np.concatenate((y_true, y_pred)), return_counts=True)
    hr = []
    for c in classes:
        class_index = np.where(y_true == c)[0]
        num = den = 0
        for index in class_index:
            a_true = dependencies[y_true[index]] + [y_true[index]]
            a_pred = dependencies[y_pred[index]] + [y_pred[index]]
            num += len([p for p in a_pred if p in a_true])
            den += len(a_true)
        hr.append(num / den if den != 0 else 0)
    if average is None:
        return hr
    elif average == 'macro':
        return np.mean(hr)
    else:  # average == 'micro'
        return np.average(hr, weights=classes_counts)


def hierarchical_f1_score(y_true, y_pred, dependencies, average=None):
    """
    dependencies: dictionary of lists, with each key referring to a class and lists are [ancestors] (w/o the root).
    If not set (TODO), the dependencies will be inferred from the y_true or y_pred vectors, otherwise dependencies are
    applied to both y_true and y_pred.
    e.g. if the hierarchy is
          /-> C
    A -> B -> D
     \-> E -> F -> G
    dependencies will be dict([(B, []), (C, [B]), (D, [B]), (E, []), (F, [E]), (G, [F, E])])
    """
    y_true, y_pred = check_arrays(y_true, y_pred, dtype=None)
    hp = hierarchical_precision_score(y_true, y_pred, dependencies, average=None)
    hr = hierarchical_recall_score(y_true, y_pred, dependencies, average=None)
    hf1 = [hmean([p, r]) for p, r in zip(hp, hr)]
    if average is None:
        return hf1
    elif average == 'macro':
        return np.mean(hf1)
    else:  # average == 'micro'
        _, classes_counts = np.unique(np.concatenate((y_true, y_pred)), return_counts=True)
        return np.average(hf1, weights=classes_counts)


def select_best_mod(df, app, group, on_col, v='min'):
    N_mean = {}
    N_std = {}
    n_list = np.unique(df['n_states'].values)

    for n in n_list:
        N_mean[n] = df[(df['App'] == app) & (df['n_states'] == n)][on_col].mean()
        N_std[n] = df[(df['App'] == app) & (df['n_states'] == n)][on_col].std()
    if v == 'min':
        N_best = min(N_mean, key=N_mean.get)
    elif v == 'max':
        N_best = max(N_mean, key=N_mean.get)

    N_best_mean = N_mean[N_best]
    N_best_std = N_std[N_best]

    # print('%s ------> [%s] µ=%s std='%(app,N_best, N_best_mean,N_best_std))
    return N_best, N_best_mean, N_best_std
