import sys
import numpy as np
import pandas as pd
#from sklearn.metrics import recall_score
#from scipy.stats import spearmanr
#from sklearn.metrics import mean_squared_error, mean_absolute_error, 
import math

#sys.path.append(".")
#from utils import metrics_lib as mlib
#import metrics_lib as mlib

from scipy.stats import spearmanr
from scipy.stats.mstats import gmean
from scipy.stats import hmean
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
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

# MASE
def mean_absolute_squared_error(y_true, y_pred, y_true_naive,y_pred_naive, is_dir=False):
    #from epftoolbox.evaluation import MASE
    y_true, y_pred = check_arrays(y_true, y_pred)
    if len(y_true) == len(y_pred) == 0:
        return np.nan
    range = np.max(y_true) - np.min(y_true)
    assert range != 0, 'Error: metric is severely ill defined.'
    if is_dir:
        return (1-compute_g_mean(y_true,y_pred))/(1-compute_g_mean(y_true_naive,y_pred_naive))
    return mean_absolute_error(y_true, y_pred)/mean_absolute_error(y_true_naive,y_pred_naive)
    # if is_dir:
    #     return compute_g_mean(y_true[1:],y_true[:-1])/compute_g_mean(y_true, y_pred)
    #return mean_squared_error(y_true, y_pred, squared=False)/mean_squared_error(y_true[1:],y_true[:-1], squared=False)

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


map_metric_function={
    'RMSE':root_mean_squared_error,
    'NRMSE':normalized_root_mean_squared_error,
    'MAPE':mean_absolute_percentage_error,
    'SMAPE':symmetric_mean_absolute_percentage_error,
    'R2':r2_score,
    'SPEARMAN':spearman_rank_correlation_coefficient,
    'ACCURACY':compute_accuracy,
    'BALACCURACY':compute_balaccuracy,
    'GMEAN':compute_g_mean,
    'MASE': mean_absolute_squared_error
}


def wrapper_dir_metric(BFs_true, BFs_pred, dmet='GMEAN',on='PKT', skip_first=False):
    
    if dmet not in ['GMEAN','ACCURACY','BALACCURACY','MASE']:
        print('DIR Metric not supported...Exit')
        exit()
    if on == 'PKT':
        signs_t = [ti[1:] for ti in BFs_true] if skip_first else BFs_true
        signs_t = np.concatenate(signs_t)
        signs_t = np.asarray(signs_t, dtype=np.float64)

        signs_p = [pi[1:] for pi in BFs_pred] if skip_first else BFs_pred
        signs_p = np.concatenate(signs_p)
        signs_p = np.asarray(signs_p, dtype=np.float64)
        signs_t = np.sign(signs_t)
        signs_p = np.sign(signs_p)
        if dmet=='MASE':
            if not skip_first:
                print('WARNING: skip first is False')
            signs_t_naive = [t[1:] for t in BFs_true] 
            signs_t_naive = np.concatenate(signs_t_naive)
            signs_t_naive = np.asarray(signs_t_naive, dtype=np.float64)
            
            signs_p_naive = [p[:-1] for p in BFs_true]
            signs_p_naive = np.concatenate(signs_p_naive)
            signs_p_naive = np.asarray(signs_p_naive, dtype=np.float64)
            signs_t_naive = np.sign(signs_t_naive)
            signs_p_naive = np.sign(signs_p_naive)
            return mean_absolute_squared_error(signs_t, signs_p, signs_t_naive, signs_p_naive,True)
        else:
            return map_metric_function[dmet](signs_t, signs_p)
    
    elif on == 'BF':

        signs_t = [np.sign(x[1:]) for x in BFs_true] if skip_first else [np.sign(x) for x in BFs_true]
        signs_p = [np.sign(x[1:]) for x in BFs_pred] if skip_first else [np.sign(x) for x in BFs_pred]
        if dmet=='MASE':
            if not skip_first:
                print('WARNING: skip first is False')    
            signs_t_naive = [np.sign(t[1:]) for t in BFs_true] 
            signs_p_naive = [np.sign(p[:-1]) for p in BFs_true]
            return [mean_absolute_squared_error(t, p,tn,tp,True) for (t, p, tn, tp) in zip(signs_t, signs_p,signs_t_naive,signs_p_naive)]
        else:
            return [map_metric_function[dmet](t, p) for (t, p) in zip(signs_t, signs_p) ]
    elif on == 'MEAN':
        signs_t = [np.sign(x[1:]) for x in BFs_true] if skip_first else [np.sign(x) for x in BFs_true]
        signs_p = [np.sign(x[1:]) for x in BFs_pred] if skip_first else [np.sign(x) for x in BFs_pred]
        if dmet=='MASE':
            if not skip_first:
                print('WARNING: skip first is False') 
            signs_t_naive = [np.sign(t[1:]) for t in BFs_true] 
            signs_p_naive = [np.sign(p[:-1]) for p in BFs_true]
            return  np.nanmean([mean_absolute_squared_error(t, p,tn,tp,True) for (t, p, tn, tp) in zip(signs_t, signs_p,signs_t_naive,signs_p_naive)])
        else:
            return np.nanmean([map_metric_function[dmet](t, p) for (t, p) in zip(signs_t, signs_p)])
        
def wrapper_dir_metric_baseline(BFs_true, dmet='GMEAN',on='PKT'):
    if dmet not in ['GMEAN','ACCURACY','BALACCURACY']:
        print('DIR Metric not supported...Exit')
        exit()
        
    BFs_true_effective_next = [t[1:] for t in BFs_true if len(t) > 1]
    BFs_true_predicted = [t[:-1] for t in BFs_true if len(t) > 1]

    if on == 'PKT':
        t = np.concatenate(BFs_true_effective_next)
        t = np.asarray(t, dtype=np.float64)

        p = np.concatenate(BFs_true_predicted)
        p = np.asarray(p, dtype=np.float64)

        signs_t = np.sign(t)
        signs_p = np.sign(p)

        return map_metric_function[dmet](signs_t, signs_p)

    elif on == 'BF':
        signs_t = [np.sign(x) for x in BFs_true_effective_next]
        signs_p = [np.sign(x) for x in BFs_true_predicted]

        return [map_metric_function[dmet](t, p) for (t, p) in zip(signs_t, signs_p)]
    elif on == 'MEAN':
        signs_t = [np.sign(x) for x in BFs_true_effective_next]
        signs_p = [np.sign(x) for x in BFs_true_predicted]
        return np.nanmean([map_metric_function[dmet](t, p) for (t, p) in zip(signs_t, signs_p)])

#####################################################################################################
#               BASELINE
#####################################################################################################
def compute_metrics_baseline(BFs_true, metrics, absolute=False, on='PKT'):
    
    out = []
    BFs_true_effective_next = [t[1:] for t in BFs_true if len(t) > 1]
    BFs_true_predicted = [t[:-1] for t in BFs_true if len(t) > 1]

    if on == 'PKT':

        pkts_t = np.concatenate(BFs_true_effective_next)
        pkts_t = np.asarray(pkts_t, dtype=np.float64)

        pkts_p = np.concatenate(BFs_true_predicted)
        pkts_p = np.asarray(pkts_p, dtype=np.float64)

        if absolute:
            out = map_metric_function[metrics](np.abs(pkts_t), np.abs(pkts_p))
        else:
            out = map_metric_function[metrics](pkts_t, pkts_p)
    elif on == 'BF' or on == 'MEAN':

        if absolute:
            out = np.array([map_metric_function[metrics](np.abs(t), np.abs(p)) for (t, p) in
                                zip(BFs_true_effective_next, BFs_true_predicted)], dtype=np.float64)
        else:
            out = np.array([map_metric_function[metrics](t, p) for (t, p) in
                                zip(BFs_true_effective_next, BFs_true_predicted)], dtype=np.float64)

        out = np.nanmean(out) if on == 'MEAN' else out
    return out


def compute_conditioned_metrics_baseline(BFs_true, metrics, direction, absolute=False, on='PKT'):
    
    out = []
    BFs_true_effective_next = [t[1:] for t in BFs_true if len(t) > 1]
    BFs_true_predicted = [t[:-1] for t in BFs_true if len(t) > 1]

    if direction == 'down':
        BFs_pred_cond = [[a for (a, b) in zip(p, t) if np.sign(b) == 1] for (p, t) in
                         zip(BFs_true_predicted, BFs_true_effective_next)]
        BFs_true_cond = [[b for b in t if np.sign(b) == 1] for t in BFs_true_effective_next]

    elif direction == 'up':
        BFs_pred_cond = [[a for (a, b) in zip(p, t) if np.sign(b) == -1] for (p, t) in
                         zip(BFs_true_predicted, BFs_true_effective_next)]
        BFs_true_cond = [[b for b in t if np.sign(b) == -1] for t in BFs_true_effective_next]
    else:
        print('Error in direction')
        exit()

    # elimino le liste vuote
    BFs_pred_cond = [p for p in BFs_pred_cond if len(p) > 0]
    BFs_true_cond = [p for p in BFs_true_cond if len(p) > 0]

    if on == 'PKT':

        pkts_t = np.concatenate(BFs_true_cond)
        pkts_t = np.asarray(pkts_t, dtype=np.float64)

        pkts_p = np.concatenate(BFs_pred_cond)
        pkts_p = np.asarray(pkts_p, dtype=np.float64)

        if absolute:
            out = map_metric_function[metrics](np.abs(pkts_t), np.abs(pkts_p))

        else:
            out = map_metric_function[metrics](pkts_t, pkts_p)

    elif on == 'BF' or on == 'MEAN':

        if absolute:
            out = np.array([map_metric_function[metrics](np.abs(t), np.abs(p)) for (t, p) in
                                zip(BFs_true_cond, BFs_pred_cond)], dtype=np.float64)
        else:
            out = np.array([map_metric_function[metrics](t, p) for (t, p) in
                                zip(BFs_true_cond, BFs_pred_cond)], dtype=np.float64)
        out = np.nanmean(out) if on == 'MEAN' else out
    else:
        print('on not supported')
    return out


###################################################################################################
###################################################################################################

# METRICS REAL vs PREDICTED
def compute_PACKET_metrics(BFs_true, BFs_pred, metrics, abs=False):
    pkts_true = np.concatenate(BFs_true)
    pkts_true = np.asarray(pkts_true, dtype=np.float64)

    pkts_pred = np.concatenate(BFs_pred)
    pkts_pred = np.asarray(pkts_pred, dtype=np.float64)

    check = True if np.shape(pkts_true)[0] != np.shape(pkts_pred)[0] else False
    if check:
        print('Error in BFs length')
        exit()

    out = []
    if abs:
        if metrics == 'APE':
            # print([(t,p) for (t,p) in zip(pkts_true,pkts_pred)] )
            # print('\n\n\n\n\n\n\n\n\n\n\n\n')
            out = [(np.abs((np.abs(t) - np.abs(p)) / np.abs(t))) * 100 for (t, p) in zip(pkts_true, pkts_pred)]
        elif metrics == 'RSE':

            out = [np.abs(np.abs(t) - np.abs(p)) for (t, p) in zip(pkts_true, pkts_pred)]
    else:
        if metrics == 'APE':
            out = [(np.abs((t - p) / t)) * 100 for (t, p) in zip(pkts_true, pkts_pred)]
        elif metrics == 'RSE':
            out = [np.abs(t - p) for (t, p) in zip(pkts_true, pkts_pred)]

    out = np.asarray(out, dtype=np.float64)
    return out

def compute_metrics(BFs_true, BFs_pred, metrics, absolute=False, on='PKT', deconstructed=False, skip_first=False, BFs_true_ref=None, BFs_pred_ref=None):
    #print(BFs_true)
    out = []
    check = [True if np.shape(t)[0] != np.shape(p)[0] else False for (t, p) in zip(BFs_true, BFs_pred)]

    for i, (t, p) in enumerate(zip(BFs_true, BFs_pred)):
        if len(t) != len(p):
            print('#%s %s/%s' % (i, len(t), len(p)))
    if True in list(set(check)) and not deconstructed:
        print('Error in BFs length')
        exit()
    elif True in list(set(check)):
        min_lens = [np.min([len(t), len(p)]) for t, p in zip(BFs_true, BFs_pred)]
        BFs_true = [bf[:min_len] for min_len, bf in zip(min_lens, BFs_true)]
        BFs_pred = [bf[:min_len] for min_len, bf in zip(min_lens, BFs_pred)]

    if on == 'PKT':
        pkts_t = [t[1:] for t in BFs_true] if skip_first else BFs_true #[t for t in BFs_true] 
        pkts_t = np.concatenate(pkts_t)
        pkts_t = np.asarray(pkts_t, dtype=np.float64)
        
        pkts_p = [p[1:] for p in BFs_pred] if skip_first else BFs_pred#[p for p in BFs_pred]
        pkts_p = np.concatenate(pkts_p)
        pkts_p = np.asarray(pkts_p, dtype=np.float64)
        if metrics=='MASE':
            if not skip_first:
                print('WARNING: skip first is False')
            if BFs_true_ref is None:
                pkts_t_naive = [t[1:] for t in BFs_true] 
                pkts_t_naive = np.concatenate(pkts_t_naive)
                pkts_t_naive = np.asarray(pkts_t_naive, dtype=np.float64)
            else:
                print('overriding true values for reference model')
                pkts_t_naive = [t[1:] for t in BFs_true_ref] 
                pkts_t_naive = np.concatenate(pkts_t_naive)
                pkts_t_naive = np.asarray(pkts_t_naive, dtype=np.float64)
                check2=True if np.shape(pkts_t)[0]==np.shape(pkts_t_naive)[0] else False
                if not check2:
                    print('WARNING: #MODEL true values != #REFERENCE true values')
            if BFs_pred_ref is None:
                print('MASE, computing prediction from first model')
                pkts_p_naive = [p[:-1] for p in BFs_pred]
                pkts_p_naive = np.concatenate(pkts_p_naive)
                pkts_p_naive = np.asarray(pkts_p_naive, dtype=np.float64)
            else:
                #print('MASE, using specified reference value')
                pkts_p_naive = [p[:-1] for p in BFs_pred_ref]
                pkts_p_naive = np.concatenate(pkts_p_naive)
                pkts_p_naive = np.asarray(pkts_p_naive, dtype=np.float64)
                check2=True if np.shape(pkts_p)[0]==np.shape(pkts_p_naive)[0] else False
                if not check2:
                    print('WARNING: #MODEL predictions != #REFERENCE predictions')
                    
            if absolute:
                out = map_metric_function[metrics](np.abs(pkts_t), np.abs(pkts_p),np.abs(pkts_t_naive), np.abs(pkts_p_naive))
            else:
                out = map_metric_function[metrics](pkts_t, pkts_p,pkts_t_naive, pkts_p_naive)
        else:
            if absolute:
                out = map_metric_function[metrics](np.abs(pkts_t), np.abs(pkts_p))
            else:
                out = map_metric_function[metrics](pkts_t, pkts_p)


    elif on == 'BF' or on == 'MEAN':
        if metrics=='MASE':
            pkts_t = [t[1:] for t in BFs_true] if skip_first else BFs_true
            pkts_p = [p[1:] for p in BFs_pred] if skip_first else BFs_pred
            pkts_t_naive = [t[1:] for t in BFs_true] 
            pkts_p_naive = [p[:-1] for p in BFs_true]
            if absolute:
                out = np.array([map_metric_function[metrics](np.abs(t), np.abs(p), np.abs(tn), np.abs(pn)) for (t, p,tn,pn) in zip(pkts_t,pkts_p,pkts_t_naive,pkts_p_naive)], dtype=np.float64)
            else:
                out = np.array([map_metric_function[metrics](t, p, tn, pn) for (t, p,tn,pn) in zip(pkts_t,pkts_p,pkts_t_naive,pkts_p_naive)], dtype=np.float64)
        else:
            if absolute:
                out = np.array([map_metric_function[metrics](np.abs(t[1:] if skip_first else t), np.abs(p[1:] if skip_first else p)) for (t, p) in zip(BFs_true, BFs_pred) if len(t) > 1], dtype=np.float64)
            else:
                out = np.array([map_metric_function[metrics](t[1:] if skip_first else t, p[1:] if skip_first else p) for (t, p) in
                                    zip(BFs_true, BFs_pred) if len(t) > 1], dtype=np.float64)
        out = np.nanmean(out) if on == 'MEAN' else out
    return out

def compute_conditioned_metrics(BFs_true, BFs_pred, metrics, direction, absolute=False, on='PKT'):
    out = []
    check = [True if np.shape(t)[0] != np.shape(p)[0] else False for (t, p) in zip(BFs_true, BFs_pred)]

    if True in list(set(check)):
        print('Error in BFs length')
        exit()

    if direction == 'down':
        BFs_pred_cond = [[a for (a, b) in zip(p[1:], t[1:]) if np.sign(b) == 1] for (p, t) in zip(BFs_pred, BFs_true)]
        BFs_true_cond = [[b for b in t[1:] if np.sign(b) == 1] for t in BFs_true]

    elif direction == 'up':
        BFs_pred_cond = [[a for (a, b) in zip(p[1:], t[1:]) if np.sign(b) == -1] for (p, t) in zip(BFs_pred, BFs_true)]
        BFs_true_cond = [[b for b in t[1:] if np.sign(b) == -1] for t in BFs_true]
    else:
        print('Error in direction')
        exit()

    # elimino le liste vuote
    BFs_pred_cond = [p for p in BFs_pred_cond if len(p) > 0]
    BFs_true_cond = [p for p in BFs_true_cond if len(p) > 0]

    if on == 'PKT':
        pkts_t = np.concatenate(BFs_true_cond)
        pkts_t = np.asarray(pkts_t, dtype=np.float64)

        pkts_p = np.concatenate(BFs_pred_cond)
        pkts_p = np.asarray(pkts_p, dtype=np.float64)

        if absolute:
            out = map_metric_function[metrics](np.abs(pkts_t), np.abs(pkts_p))
        else:
            out = map_metric_function[metrics](pkts_t, pkts_p)
    elif on == 'BF' or on == 'MEAN':
        if absolute:
            out = np.array([map_metric_function[metrics](np.abs(t), np.abs(p)) for (t, p) in
                                zip(BFs_true_cond, BFs_pred_cond) if len(t) > 1], dtype=np.float64)
        else:
            out = np.array([map_metric_function[metrics](t, p) for (t, p) in
                                zip(BFs_true_cond, BFs_pred_cond) if len(t) > 1], dtype=np.float64)

        out = np.nanmean(out) if on == 'MEAN' else out
    else:
        print('on not supported')

    return out


def select_best_mod(df, app, group, on_col, v='min'):
    N_mean = {}
    N_std = {}
    n_list = np.unique(df['n_states'].values)

    for n in n_list:
        N_mean[n] = df[(df['App'] == app) & ((df['n_states'] == n))][on_col].mean()
        N_std[n] = df[(df['App'] == app) & ((df['n_states'] == n))][on_col].std()
    if v == 'min':
        N_best = min(N_mean, key=N_mean.get)
    elif v == 'max':
        N_best = max(N_mean, key=N_mean.get)

    N_best_mean = N_mean[N_best]
    N_best_std = N_std[N_best]
    print('**********')
    print(on_col, N_mean)
    print('%s ------> [%s] µ=%s std=%s' % (app, N_best, N_best_mean, N_best_std))
    print('**********')
    return N_best, N_best_mean, N_best_std


def compute_AIC(lk, N, k):
    AIC = -2 * lk + 2 * (N * (N - 1) + k * N + N - 1)
    return AIC


def compute_BIC(lk, N, k, Nt):
    BIC = -2 * lk + (N * (N - 1) + k * N + N - 1) * math.log(Nt)
    return BIC


def wrapper_AIC(likelihoods, N, biflows, n_params):
    nbiflows = np.shape(biflows)[0]
    print('Biflows number: %s' % nbiflows)

    # ntransitions=np.sum([len(pks)-1 for pks in biflows])
    # print('Transitions number: %s'%ntransitions)

    # likelihoods_mean=np.max(likelihoods)/nbiflows
    likelihoods_mean = np.max(likelihoods)

    AIC = compute_AIC(likelihoods_mean, N, n_params)

    return AIC


def wrapper_BIC(likelihoods, N, biflows, n_params):
    nbiflows = np.shape(biflows)[0]
    print('Biflows number: %s' % nbiflows)

    ntransitions = np.sum([len(pks) - 1 for pks in biflows])
    print('Transitions number: %s' % ntransitions)

    # likelihoods_mean=np.max(likelihoods)/nbiflows
    likelihoods_mean = np.max(likelihoods)

    BIC = compute_BIC(likelihoods_mean, N, n_params, ntransitions)

    return BIC


def gen_df_best_synthetic(dictr, dictr_base):
    df = pd.DataFrame(columns=[
        'App',
        'Type',
        'NPL_PKT',
        'RMSE_PL_MEAN_PKT',
        'RMSE_PL_STD_PKT',
        'NPL_BF',
        'RMSE_PL_MEAN_BF',
        'RMSE_PL_STD_BF',
        'NIAT_PKT',
        'RMSE_IAT_MEAN_PKT',
        'RMSE_IAT_STD_PKT',
        'NIAT_BF',
        'RMSE_IAT_MEAN_BF',
        'RMSE_IAT_STD_BF',
        'NDIR_PKT',
        'RMSE_DIR_MEAN_PKT',
        'RMSE_DIR_STD_PKT',
        'NDIR_BF',
        'RMSE_DIR_MEAN_BF',
        'RMSE_DIR_STD_BF'
    ])

    for app, res in dictr.items():
        df = df.append({
            'App': app,
            'Type': res[0],
            'NPL_PKT': res[1],
            'RMSE_PL_MEAN_PKT': res[2],
            'RMSE_PL_STD_PKT': res[3],
            'NPL_BF': res[4],
            'RMSE_PL_MEAN_BF': res[5],
            'RMSE_PL_STD_BF': res[6],
            'NIAT_PKT': res[7],
            'RMSE_IAT_MEAN_PKT': res[8],
            'RMSE_IAT_STD_PKT': res[9],
            'NIAT_BF': res[10],
            'RMSE_IAT_MEAN_BF': res[11],
            'RMSE_IAT_STD_BF': res[12],
            'NDIR_PKT': res[13],
            'RMSE_DIR_MEAN_PKT': res[14],
            'RMSE_DIR_STD_PKT': res[15],
            'NDIR_BF': res[16],
            'RMSE_DIR_MEAN_BF': res[17],
            'RMSE_DIR_STD_BF': res[18]
        }, ignore_index=True)

    for app, res in dictr_base.items():
        df = df.append({
            'App': app,
            'Type': res[0],
            'NPL_PKT': res[1],
            'RMSE_PL_MEAN_PKT': res[2],
            'RMSE_PL_STD_PKT': res[3],
            'NPL_BF': res[4],
            'RMSE_PL_MEAN_BF': res[5],
            'RMSE_PL_STD_BF': res[6],
            'NIAT_PKT': res[7],
            'RMSE_IAT_MEAN_PKT': res[8],
            'RMSE_IAT_STD_PKT': res[9],
            'NIAT_BF': res[10],
            'RMSE_IAT_MEAN_BF': res[11],
            'RMSE_IAT_STD_BF': res[12],
            'NDIR_PKT': res[13],
            'RMSE_DIR_MEAN_PKT': res[14],
            'RMSE_DIR_STD_PKT': res[15],
            'NDIR_BF': res[16],
            'RMSE_DIR_MEAN_BF': res[17],
            'RMSE_DIR_STD_BF': res[18]
        }, ignore_index=True)

    return df


def round_down(num, divisor):
    return num - (num % divisor)


def select_best_models_for_all(hmm_results_df, metrics, s, opt='both'):
    df = pd.DataFrame(columns=[
        'App',
        'PL_PKT',
        'PL_BF',
        'IAT_PKT',
        'IAT_BF',
        'DIR_PKT',
        'DIR_BF',
        'MET'
    ])

    apps = np.unique(hmm_results_df['App'].values)
    for a, app in enumerate(apps):
        if opt == 'both' or opt == 'pkt':
            n_best_DIR_pkt, mean_best_DIR_pkt, std_best_DIR_pkt = select_best_mod(hmm_results_df, app, 'n_states',
                                                                                  '%s_PKT_GMean' % (s), v='max')
            print('%s ------> [%s] DIR G-mean PKT (µ,std)=(%s,%s)' % (
            app, n_best_DIR_pkt, mean_best_DIR_pkt, std_best_DIR_pkt))
        if opt == 'both' or opt == 'bf':
            n_best_DIR_bf, mean_best_DIR_bf, std_best_DIR_bf = select_best_mod(hmm_results_df, app, 'n_states',
                                                                               '%s_MEAN_GMean' % (s), v='max')
            print('%s ------> [%s] DIR G-mean BF (µ,std)=(%s,%s)' % (
            app, n_best_DIR_bf, mean_best_DIR_bf, std_best_DIR_bf))
        for met in metrics:

            if opt == 'both' or opt == 'pkt':
                n_best_PL_pkt, mean_best_PL_pkt, std_best_PL_pkt = select_best_mod(hmm_results_df, app, 'n_states',
                                                                                   '%s_abs_%s_PL' % (s, met))
                n_best_IAT_pkt, mean_best_IAT_pkt, std_best_IAT_pkt = select_best_mod(hmm_results_df, app, 'n_states',
                                                                                      '%s_%s_IAT' % (s, met))

                print('%s ------> [%s] PL %s PKT (µ,std)=(%s,%s)' % (
                app, n_best_PL_pkt, met, mean_best_PL_pkt, std_best_PL_pkt))
                print('%s ------> [%s] IAT %s PKT (µ,std)=(%s,%s)' % (
                app, n_best_IAT_pkt, met, mean_best_IAT_pkt, std_best_IAT_pkt))

            if opt == 'both' or opt == 'bf':
                n_best_PL_bf, mean_best_PL_bf, std_best_PL_bf = select_best_mod(hmm_results_df, app, 'n_states',
                                                                                '%s_abs_%s_PL_Mean' % (s, met))
                n_best_IAT_bf, mean_best_IAT_bf, std_best_IAT_bf = select_best_mod(hmm_results_df, app, 'n_states',
                                                                                   '%s_%s_IAT_Mean' % (s, met))

                print('%s ------> [%s] PL %s BF (µ,std)=(%s,%s)' % (
                app, n_best_PL_bf, met, mean_best_PL_bf, std_best_PL_bf))
                print('%s ------> [%s] IAT %s BF (µ,std)=(%s,%s)' % (
                app, n_best_IAT_bf, met, mean_best_IAT_bf, std_best_IAT_bf))

            df = df.append({
                'App': app,
                'PL_PKT': n_best_PL_pkt if opt == 'both' or opt == 'pkt' else 0,
                'PL_BF': n_best_PL_bf if opt == 'both' or opt == 'bf' else 0,
                'IAT_PKT': n_best_IAT_pkt if opt == 'both' or opt == 'pkt' else 0,
                'IAT_BF': n_best_IAT_bf if opt == 'both' or opt == 'bf' else 0,
                'DIR_PKT': n_best_DIR_pkt if opt == 'both' or opt == 'pkt' else 0,
                'DIR_BF': n_best_DIR_bf if opt == 'both' or opt == 'bf' else 0,
                'MET': met
            }, ignore_index=True)

    return df


def select_best_by_AIC_or_BIC(hmm_results_df, met='AIC'):
    df = pd.DataFrame(columns=[
        'App',
        '%s1' % met,
        '%s2' % met
    ])

    apps = np.unique(hmm_results_df['App'].values)
    grouped_apps_df = hmm_results_df.groupby('App')
    for a, app in enumerate(apps):
        ghmm_results_df = grouped_apps_df.get_group(app)

        means = ghmm_results_df.groupby('n_states')['%s' % met].mean()
        nlist = np.unique(ghmm_results_df['n_states'].values)

        MET1 = means[means == means.min()].index[0]
        print('[%s] N with best Mean: %s' % (met, MET1))

        folds_list = np.unique(ghmm_results_df['Fold'].values)

        best_df = ghmm_results_df.loc[ghmm_results_df.groupby('Fold')['%s' % met].idxmin()]
        nlist = np.unique(ghmm_results_df['n_states'].values)

        # l'elemento della lista più vicino
        MET2 = min(nlist, key=lambda x: abs(x - best_df['n_states'].sum() / np.shape(best_df['Fold'].values)[0]))
        # MET2=int(round(best_df['n_states'].sum()/np.shape(best_df['Fold'].values)[0]))

        print('[%s] Best N by Fold: %s' % (met, MET2))
        df = df.append({
            'App': ghmm_results_df['App'].values[0],
            '%s1' % met: MET1,
            '%s2' % met: MET2
        }, ignore_index=True)

    return df


# BASELINE

def metric_computation(df, metric, feature, dset, opt='both', discr=''):
    print('%s %s computation on %s' % (feature, metric, dset))

    deconstructed = 'deconstructed' in discr

    if opt == 'both' or opt == 'pkt':
        #print('Mean on PKT')
        df['%s_%s_%s%s' % (dset, metric, feature, discr)] = df.parallel_apply(
            lambda x: compute_metrics(x['%s_actual_%ss%s' % (dset, feature, discr)],
                                      x['%s_predicted_%ss%s' % (dset, feature, discr)],
                                      metric, False, 'PKT', deconstructed), axis=1)

    if opt == 'both' or opt == 'bf':
        #print('Mean on BF')
        # METRICS FOR BFs
        df['%s_%s_%s_Mean%s' % (dset, metric, feature, discr)] = df.parallel_apply(
            lambda x: compute_metrics(x['%s_actual_%ss%s' % (dset, feature, discr)],
                                      x['%s_predicted_%ss%s' % (dset, feature, discr)],
                                      metric, False, 'MEAN', deconstructed), axis=1)
    return df


def cond_dir_metric_computation(hmm_results_df, met, s, opt='both'):
    
    if opt == 'both' or opt == 'pkt':
        # METRICS FOR PACKETS
        hmm_results_df['%s_abs_%s_PL' % (s, met)] = hmm_results_df.parallel_apply(
            lambda x: compute_metrics(x['%s_actual_PLs' % (s)], x['%s_predicted_PLs' % (s)], met, True, 'PKT'), axis=1)

        # METRICS FOR PACKETS
        hmm_results_df['%s_%s_PL_DOWN' % (s, met)] = hmm_results_df.parallel_apply(
            lambda x: compute_conditioned_metrics(x['%s_actual_PLs' % (s)], x['%s_predicted_PLs' % (s)], met, 'down',
                                                  False, 'PKT'), axis=1)

        # METRICS FOR PACKETS
        hmm_results_df['%s_%s_PL_UP' % (s, met)] = hmm_results_df.parallel_apply(
            lambda x: compute_conditioned_metrics(x['%s_actual_PLs' % (s)], x['%s_predicted_PLs' % (s)], met, 'up',
                                                  False, 'PKT'), axis=1)

        # METRICS FOR PACKETS
        hmm_results_df['%s_abs_%s_PL_DOWN' % (s, met)] = hmm_results_df.parallel_apply(
            lambda x: compute_conditioned_metrics(x['%s_actual_PLs' % (s)], x['%s_predicted_PLs' % (s)], met, 'down',
                                                  True, 'PKT'), axis=1)

        # METRICS FOR PACKETS
        hmm_results_df['%s_abs_%s_PL_UP' % (s, met)] = hmm_results_df.parallel_apply(
            lambda x: compute_conditioned_metrics(x['%s_actual_PLs' % (s)], x['%s_predicted_PLs' % (s)], met, 'up',
                                                  True, 'PKT'), axis=1)

    if opt == 'both' or opt == 'bf':
        # METRICS FOR BFs
        hmm_results_df['%s_abs_%s_PL_Mean' % (s, met)] = hmm_results_df.parallel_apply(
            lambda x: compute_metrics(x['%s_actual_PLs' % (s)], x['%s_predicted_PLs' % (s)], met, True, 'MEAN'), axis=1)

        # METRICS FOR BFs
        hmm_results_df['%s_%s_PL_Mean_DOWN' % (s, met)] = hmm_results_df.parallel_apply(
            lambda x: compute_conditioned_metrics(x['%s_actual_PLs' % (s)], x['%s_predicted_PLs' % (s)], met, 'down',
                                                  False, 'MEAN'), axis=1)

        # METRICS FOR BFs
        hmm_results_df['%s_%s_PL_Mean_UP' % (s, met)] = hmm_results_df.parallel_apply(
            lambda x: compute_conditioned_metrics(x['%s_actual_PLs' % (s)], x['%s_predicted_PLs' % (s)], met, 'up',
                                                  False, 'MEAN'), axis=1)

        # METRICS FOR BFs
        hmm_results_df['%s_abs_%s_PL_Mean_DOWN' % (s, met)] = hmm_results_df.parallel_apply(
            lambda x: compute_conditioned_metrics(x['%s_actual_PLs' % (s)], x['%s_predicted_PLs' % (s)], met, 'down',
                                                  True, 'MEAN'), axis=1)

        # METRICS FOR BFs
        hmm_results_df['%s_abs_%s_PL_Mean_UP' % (s, met)] = hmm_results_df.parallel_apply(
            lambda x: compute_conditioned_metrics(x['%s_actual_PLs' % (s)], x['%s_predicted_PLs' % (s)], met, 'up',
                                                  True, 'MEAN'), axis=1)

    return hmm_results_df


def dir_metric_computation(hmm_results_df, st, dmet='GMEAN',opt='both'):
    #print('Accuracy computation')
    if opt == 'both' or opt == 'pkt':
        hmm_results_df['%s_PKT_%s' % (st,dmet)] = hmm_results_df.parallel_apply(
            lambda x: wrapper_dir_metric(x['%s_actual_PLs' % st], x['%s_predicted_PLs' % st],dmet), axis=1)
    if opt == 'both' or opt == 'bf':
        print('****************Accuracy on BFS****************')
        hmm_results_df['%s_MEAN_%s' % (st,dmet)] = hmm_results_df.parallel_apply(
            lambda x: wrapper_dir_metric(x['%s_actual_PLs' % st], x['%s_predicted_PLs' % st],dmet, 'MEAN'), axis=1)
        print('********************************************')
    return hmm_results_df

def baseline_computation_on_Row_or_Fold(hmm_results_df, s, met, v, on='row', fold='1', opt='both', discr=''):

    if on == 'fold':
        first_row_4fold = hmm_results_df[hmm_results_df.Fold == fold].iloc[0]
        col = first_row_4fold['%s_actual_%ss%s' % (s, v, discr)]

        if opt == 'both' or opt == 'pkt':
            # METRICS FOR PACKETS BASELINE
            mtx = compute_metrics_baseline(col, met, False, 'PKT')
            print(mtx)
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_%s_%s_BASE%s' % (s, met, v, discr)] = mtx

        if opt == 'both' or opt == 'bf':
            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            mtx = compute_metrics_baseline(col, met, False, 'MEAN')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_%s_%s_Mean_BASE%s' % (s, met, v, discr)] = mtx

    elif on == 'row':
        if opt == 'both' or opt == 'pkt':
            # METRICS FOR PACKETS BASELINE
            hmm_results_df['%s_%s_%s_BASE%s' % (s, met, v, discr)] = hmm_results_df[
                '%s_actual_%ss%s' % (s, v, discr)].parallel_apply(compute_metrics_baseline, args=(met, False, 'PKT'))
        if opt == 'both' or opt == 'bf':
            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            hmm_results_df['%s_%s_%s_Mean_BASE%s' % (s, met, v, discr)] = hmm_results_df[
                '%s_actual_%ss%s' % (s, v, discr)].parallel_apply(compute_metrics_baseline, args=(met, False, 'MEAN'))

    return hmm_results_df


def dir_baseline_computation_on_Row_or_Fold(hmm_results_df, s, met, on='row', fold='1', opt='both'):
    if on == 'fold':
        #print('on Fold #%s %s-set %s' % (fold, s, met))
        first_row_4fold = hmm_results_df[hmm_results_df.Fold == fold].iloc[0]
        col = first_row_4fold['%s_actual_PLs' % (s)]

        if opt == 'both' or opt == 'pkt':
            # METRICS FOR PACKETS BASELINE
            mtx = compute_metrics_baseline(col, met, True, 'PKT')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_abs_%s_PL_BASE' % (s, met)] = mtx

            # Conditioned Metrics

            # METRICS FOR PACKETS BASELINE
            mtx = compute_conditioned_metrics_baseline(col, met, 'down', False, 'PKT')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_%s_PL_BASE_DOWN' % (s, met)] = mtx

            # METRICS FOR PACKETS BASELINE
            mtx = compute_conditioned_metrics_baseline(col, met, 'up', False, 'PKT')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_%s_PL_BASE_UP' % (s, met)] = mtx

            # METRICS FOR PACKETS BASELINE
            mtx = compute_conditioned_metrics_baseline(col, met, 'down', True, 'PKT')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_abs_%s_PL_BASE_DOWN' % (s, met)] = mtx

            # METRICS FOR PACKETS BASELINE
            mtx = compute_conditioned_metrics_baseline(col, met, 'up', True, 'PKT')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_abs_%s_PL_BASE_UP' % (s, met)] = mtx

        if opt == 'both' or opt == 'bf':
            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            mtx = compute_metrics_baseline(col, met, True, 'MEAN')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_abs_%s_PL_Mean_BASE' % (s, met)] = mtx

            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            mtx = compute_conditioned_metrics_baseline(col, met, 'down', False, 'MEAN')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_%s_PL_Mean_BASE_DOWN' % (s, met)] = mtx

            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            mtx = compute_conditioned_metrics_baseline(col, met, 'up', False, 'MEAN')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_%s_PL_Mean_BASE_UP' % (s, met)] = mtx

            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            mtx = compute_conditioned_metrics_baseline(col, met, 'down', True, 'MEAN')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_abs_%s_PL_Mean_BASE_DOWN' % (s, met)] = mtx

            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            mtx = compute_conditioned_metrics_baseline(col, met, 'up', True, 'MEAN')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_abs_%s_PL_Mean_BASE_UP' % (s, met)] = mtx

    elif on == 'row':

        if opt == 'both' or opt == 'pkt':
            # METRICS FOR PACKETS BASELINE
            hmm_results_df['%s_abs_%s_PL_BASE' % (s, met)] = hmm_results_df['%s_actual_PLs' % (s)].parallel_apply(
                compute_metrics_baseline, args=(met, True, 'PKT'))

            # METRICS FOR PACKETS BASELINE
            hmm_results_df['%s_%s_PL_BASE_DOWN' % (s, met)] = hmm_results_df['%s_actual_PLs' % (s)].parallel_apply(
                compute_conditioned_metrics_baseline, args=(met, 'down', False, 'PKT'))

            # METRICS FOR PACKETS BASELINE
            hmm_results_df['%s_%s_PL_BASE_UP' % (s, met)] = hmm_results_df['%s_actual_PLs' % (s)].parallel_apply(
                compute_conditioned_metrics_baseline, args=(met, 'up', False, 'PKT'))

            # METRICS FOR PACKETS BASELINE
            hmm_results_df['%s_actual_PLs' % (s)].parallel_apply(compute_conditioned_metrics_baseline,
                                                                 args=(met, 'down', True, 'PKT'))
            hmm_results_df['%s_abs_%s_PL_BASE_DOWN' % (s, met)] = hmm_results_df['%s_actual_PLs' % (s)].parallel_apply(
                compute_conditioned_metrics_baseline, args=(met, 'down', True, 'PKT'))

            # METRICS FOR PACKETS BASELINE
            hmm_results_df['%s_abs_%s_PL_BASE_UP' % (s, met)] = hmm_results_df['%s_actual_PLs' % (s)].parallel_apply(
                compute_conditioned_metrics_baseline, args=(met, 'up', True, 'PKT'))

        if opt == 'both' or opt == 'bf':
            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            hmm_results_df['%s_abs_%s_PL_Mean_BASE' % (s, met)] = hmm_results_df['%s_actual_PLs' % (s)].parallel_apply(
                compute_metrics_baseline, args=(met, True, 'MEAN'))

            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            hmm_results_df['%s_%s_PL_Mean_BASE_DOWN' % (s, met)] = hmm_results_df['%s_actual_PLs' % (s)].parallel_apply(
                compute_conditioned_metrics_baseline, args=(met, 'down', False, 'MEAN'))

            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            hmm_results_df['%s_%s_PL_Mean_BASE_UP' % (s, met)] = hmm_results_df['%s_actual_PLs' % (s)].parallel_apply(
                compute_conditioned_metrics_baseline, args=(met, 'up', False, 'MEAN'))

            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            hmm_results_df['%s_abs_%s_PL_Mean_BASE_DOWN' % (s, met)] = hmm_results_df[
                '%s_actual_PLs' % (s)].parallel_apply(compute_conditioned_metrics_baseline,
                                                      args=(met, 'down', True, 'MEAN'))

            # METRICS FOR BFs BASELINE compute_Mean_BF_metrics_baseline
            hmm_results_df['%s_abs_%s_PL_Mean_BASE_UP' % (s, met)] = hmm_results_df[
                '%s_actual_PLs' % (s)].parallel_apply(compute_conditioned_metrics_baseline,
                                                      args=(met, 'up', True, 'MEAN'))

    return hmm_results_df




def baseline_dir_metric_computation_on_Row_or_Fold(hmm_results_df, s, on='row', fold='1',dmet='GMEAN' ,opt='both'):
    #print(on, fold, opt)
    if on == 'fold':
        #print('GMean baseline on PKT')
        first_row_4fold = hmm_results_df[hmm_results_df.Fold == fold].iloc[0]
        col = first_row_4fold['%s_actual_PLs' % (s)]

        if opt == 'both' or opt == 'pkt':
            mtx = wrapper_dir_metric_baseline(col,dmet)
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_PKT_%s_BASE' %(s,dmet)] = mtx

        if opt == 'both' or opt == 'bf':
            print('bf in', on, fold, opt)
            mtx = wrapper_dir_metric_baseline(col,dmet ,'MEAN')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_MEAN_%s_BASE' %(s,dmet)] = mtx
            print('*****************************************************')
    elif on == 'row':

        if opt == 'both' or opt == 'pkt':
            print('GMean baseline on PKT')
            hmm_results_df['%s_PKT_%s_BASE' % (s,dmet)] = hmm_results_df.parallel_apply(
                lambda x: wrapper_dir_metric_baseline(x['%s_actual_PLs' % s],dmet), axis=1)

        if opt == 'both' or opt == 'bf':
            print('****************GMean baseline on BFS****************')
            hmm_results_df['%s_MEAN_%s_BASE' % (s,dmet)] = hmm_results_df.parallel_apply(
                lambda x: wrapper_dir_metric_baseline(x['%s_actual_PLs' % s],dmet ,'MEAN'), axis=1)
            print('*****************************************************')
    return hmm_results_df
'''
def baseline_gmean_computation_on_Row_or_Fold(hmm_results_df, s, on='row', fold='1',dmet='GMEAN' ,opt='both'):
    #print(on, fold, opt)
    if on == 'fold':
        #print('GMean baseline on PKT')
        first_row_4fold = hmm_results_df[hmm_results_df.Fold == fold].iloc[0]
        col = first_row_4fold['%s_actual_PLs' % (s)]

        if opt == 'both' or opt == 'pkt':
            # GMean FOR PACKETS BASELINE
            mtx = wrapper_dir_metric_baseline(col,dmet)
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_PKT_%s_BASE' %(s,dmet)] = mtx

        if opt == 'both' or opt == 'bf':
            print('bf in', on, fold, opt)
            print('****************GMean baseline on BFS****************')
            # GMean Mean for BFs Baseline
            mtx = wrapper_dir_metric_baseline(col,dmet ,'MEAN')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_MEAN_%s_BASE' %(s,dmet)] = mtx
            print('*****************************************************')
    elif on == 'row':

        if opt == 'both' or opt == 'pkt':
            print('GMean baseline on PKT')
            hmm_results_df['%s_PKT_%s_BASE' % (s,dmet)] = hmm_results_df.parallel_apply(
                lambda x: wrapper_dir_metric_baseline(x['%s_actual_PLs' % s],dmet), axis=1)

        if opt == 'both' or opt == 'bf':
            print('****************GMean baseline on BFS****************')
            hmm_results_df['%s_MEAN_%s_BASE' % (s,dmet)] = hmm_results_df.parallel_apply(
                lambda x: wrapper_dir_metric_baseline(x['%s_actual_PLs' % s],dmet ,'MEAN'), axis=1)
            print('*****************************************************')
    return hmm_results_df


def baseline_accuracy_computation_on_Row_or_Fold(hmm_results_df, s, on='row', fold='1', opt='both'):
    #print(on, fold, opt)
    if on == 'fold':
        #print('Accuracy baseline on PKT')
        first_row_4fold = hmm_results_df[hmm_results_df.Fold == fold].iloc[0]
        col = first_row_4fold['%s_actual_PLs' % (s)]

        if opt == 'both' or opt == 'pkt':
            # GMean FOR PACKETS BASELINE
            mtx = wrapper_accuracy_baseline(col)
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_PKT_Accuracy_BASE' % s] = mtx

        if opt == 'both' or opt == 'bf':
            print('bf in', on, fold, opt)
            print('****************Accuracy baseline on BFS****************')
            # GMean Mean for BFs Baseline
            mtx = wrapper_accuracy_baseline(col, 'MEAN')
            hmm_results_df.loc[hmm_results_df['Fold'] == fold, '%s_MEAN_Accuracy_BASE' % s] = mtx
            print('*****************************************************')
    elif on == 'row':

        if opt == 'both' or opt == 'pkt':
            print('Accuracy baseline on PKT')
            hmm_results_df['%s_PKT_Accuracy_BASE' % s] = hmm_results_df.parallel_apply(
                lambda x: wrapper_accuracy_baseline(x['%s_actual_PLs' % s]), axis=1)

        if opt == 'both' or opt == 'bf':
            print('****************Accuracy baseline on BFS****************')
            hmm_results_df['%s_MEAN_Accuracy_BASE' % s] = hmm_results_df.parallel_apply(
                lambda x: wrapper_accuracy_baseline(x['%s_actual_PLs' % s], 'MEAN'), axis=1)
            print('*****************************************************')
    return hmm_results_df
'''