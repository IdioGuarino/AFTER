import csv
import errno
import os,glob,shutil
import sys
import warnings
from _ast import arg
from ast import literal_eval
from builtins import object
#from collections import OrderedDict
from itertools import permutations,product # combinations, count, 
import math
from math import sqrt
from pprint import pprint
from pprint import pprint
import pickle as pk         
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
#from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from numpy import dtype
from scipy.linalg import eig, eigvals
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.utils.random import sample_without_replacement
from statsmodels.distributions.empirical_distribution import ECDF

from sklearn.model_selection import KFold


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# warnings.filterwarnings(action="error", category=np.ComplexWarning)

################################################################################################################################################################################################################################################################
# Poiché per un dato biflusso l'IAT di un pacchetto i è calcolato rispetto al pacchetto i-1 indipendentemente dalla direzione di quest'ultimo,
# i valori vengono ri-calcolati in modo tale che l'IAT del pacchetto i sia calcolato rispetto al pacchetto j che lo precede nella stessa direzione.
# Input:
#        iats:        lista di iat
# Output:
#        iats_new:    lista di iat
def compute_iat(iats):
    iats_new = []
    cup = 0
    cdown = 0

    for i, iati in enumerate(iats):
        if iati < 0 and cup == 0:
            cup = 1
            # iats_new.append(0)
        elif iati > 0 and cdown == 0:
            cdown = 1
            # iats_new.append(0)
        else:
            iat_t = iats[i]
            j = i - 1

            while j >= 0 and np.sign(iats[j]) != np.sign(iats[i]):
                iat_t -= iats[j]
                j = j - 1

            iats_new.append(iat_t)

    return iats_new


################################################################################################################################################################################################################################################################
def converter(x):
    return literal_eval(x)


################################################################################################################################################################################################################################################################
# data una lista di liste, elimina tutte le liste nulle
def drop_empty_list(x, n=0):
    y = []
    for i in x:
        if len(i) > n:
            y.append(i)
    return y


def dropNull(x):
    return x[x.notnull()]


################################################################################################################################################################################################################################################################
# elimina dalle liste i pl nulli
# pls:         lista di liste delle payload-length non binned
def remove_zero_pl(pls):
    return [pl for pl in pls if pl != 0]


################################################################################################################################################################################################################################################################
# rimuove il primo iat all'interno del biflusso
def remove_first_iat(iats):
    return iats[1:]


################################################################################################################################################################################################################################################################
# rimuove dalla lista 'list' tutti gli elementi che assumono un determinato valore 'item'
def remove_item(seq, item):
    return [value for value in seq if value != item]


################################################################################################################################################################################################################################################################
# converte gli iat in micros, i valori piu' piccoli di 1 micros vengono settati a 1.0 micros
def to_micros(seq):
    return [value * pow(10, 6) if value * pow(10, 6) >= 1.0 else 1.0 for value in seq]


################################################################################################################################################################################################################################################################
# converte gli iat in micros, i valori piu' piccoli di 1 micros vengono settati a 1.0 micros
def to_micros_with_saturation(seq, lower, upper):
    # return [value*pow(10,6) if value*pow(10,6)>=1.0 else 1.0 for value in list]
    seq_out = []
    for v in seq:
        if (v >= lower and v <= upper):
            seq_out.append(v * pow(10, 6))
        elif (v < lower):
            seq_out.append(lower * pow(10, 6))
        else:
            seq_out.append(upper * pow(10, 6))

    return seq_out


################################################################################################################################################################################################################################################################
# estrae tutti gli elementi compatibili con la direzione indicata
# Input:
#        pl_bin:     lista di pl
#        dir:        indica la direzione. 'up'=Upstream 'down'=Downstream
# Output:
#        ritorna i valori estratti in base alla direzione
def extract_pl_by_dir(pl_bin, dir):
    if dir == 'up':
        return [value for value in pl_bin if value < 0]
    else:
        return [value for value in pl_bin if value > 0]


################################################################################################################################################################################################################################################################
# applica l'operatore abs() agli elementi di una lista in input
def abs_list(l):
    return [abs(item) for item in l]


################################################################################################################################################################################################################################################################
# restituisce il massimo in modulo di una lista
def max_list(l):
    return max([abs(i) for i in l])


################################################################################################################################################################################################################################################################
# riceve in input una lista di liste e restituisce l'input sottoforma di lista singola
# Input:
#        data:            lista di liste
# Output:
#        single_list:     lista 
def to_single_list(data):
    data = list(data)
    single_list = []

    # creo un'unica lista per tutti i valori
    for i in data:
        for j in i:
            single_list.append(j)
    return single_list


################################################################################################################################################################################################################################################################
# Effettua la predizione sulla base della matrice di transizione
# Input:
#        A:       matrice di transizione
#        obs:     lista di valori da utilizzare per effettuare la predizione
# Output:
#        obs_p:   lista dei valori predetti
def predict(obs, A):
    arg = np.argmax(A, axis=1)
    obs_p = [arg[i] for i in obs]
    return obs_p


nop_memoize = dict()


def nth_order_predict(obs, nth_order_tm, nbins, order=1, stationary_distr=None, debug=False):
    history = [tuple(obs[i:i + order]) for i in range(len(obs) - (order - 1))]
    if stationary_distr is None:
        not_found_index = math.ceil(nbins / 2) - 1
        predictions = [np.argmax(nth_order_tm.get(h, nop_memoize[
            nbins] if nbins in nop_memoize else nop_memoize.setdefault(nbins, [0 if i != not_found_index else 1 for i in
                                                                               range(nbins)]))) for h in history]
    else:
        predictions = [np.argmax(nth_order_tm.get(h, stationary_distr)) for h in history]
    if debug:
        print('nth_order_predict')
        print(obs)
        print(predictions)
        input()
    found_count = sum([h in nth_order_tm for h in history])
    not_found_count = sum([not h in nth_order_tm for h in history])
    eprint('Sequences found in tm: %s' % found_count)
    eprint('Sequences not found in tm: %s' % not_found_count)
    return predictions


nopp_memoize = dict()


def nth_order_predict_proba(obs, nth_order_tm, nbins, order=1, stationary_distr=None):
    try:
        history = [tuple(obs[i:i + order]) for i in range(len(obs) - (order - 1))]
    except:
        print(obs)
        print(order)
        print(type(obs))
        exit()
    if stationary_distr is None:
        prediction_probas = [nth_order_tm.get(h,
                                              nopp_memoize[nbins] if nbins in nopp_memoize else nopp_memoize.setdefault(
                                                  nbins, np.ones(nbins) / nbins)) for h in history]
    else:
        prediction_probas = [nth_order_tm.get(h, stationary_distr) for h in history]
    return prediction_probas


def baseline_predict(obs, order=1):
    history = [tuple(obs[i:i + order]) for i in range(len(obs) - (order - 1))]
    predictions = [h[-1] for h in history]
    return predictions


################################################################################################################################################################################################################################################################
def compute_rmse(e, p):
    e = np.asarray(e, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return sqrt(mean_squared_error(e, p))


################################################################################################################################################################################################################################################################
def df_error_plot(ax, x, y, hue, hue_order, data, savepath=None, format='pdf', major_loc=100):
    # mostro le linee della griglia sia in orrizzontale che in verticale
    # ax.yaxis.grid(True,linestyle='--', linewidth=0.5, zorder=1) # Show the horizontal gridlines
    # ax.xaxis.grid(True) # Show the vertical gridlines

    ax = sns.pointplot(x=x, y=y, hue=hue, hue_order=hue_order, data=data, join=False, scale=0.5, dodge=0.25, ci='sd',
                       zorder=10, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # ax.legend(loc='upper left', bbox_to_anchor=(1,1),ncol=1, handletextpad=0.1, columnspacing=0.1)
    # sns.set_style("whitegrid")

    ax.grid(b=True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)

    # ax.yaxis.set_major_locator(MultipleLocator(major_loc))

    ax.legend(loc='best', handletextpad=0.1, columnspacing=0.1, frameon=True, ncol=len(hue_order), fontsize=10,
              framealpha=0.5)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, format=format, bbox_inches='tight', pad_inches=0.0)
    # else:
    #    plt.show()


################################################################################################################################################################################################################################################################


def plotECDF(data_in, ax, xlabel=None, x2label=None, label=None, survival=None, color=None, linestyle="solid",
             xscale="lin", yscale="lin", xlim=None, x2lim=None, ylim=(0, 1), majorxTicks=None, extendLeft=0,
             savefig=None, format="pdf"):
    #    if color == None:
    #        plt.rc('lines', linewidth=4)
    #        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k', 'm'])))
    #
    #            +cycler('linestyle', ['-', '--', ':', '-.'])))

    # data = list(data)
    data = data_in.copy()
    data.sort()

    # Remove traces with duration < 10 seconds
    #    data = [x for x in data if x >= 10]

    ecdf = ECDF(data)
    y = ecdf(data) if not survival else 1 - ecdf(data)
    y = list(y)

    #    print len(data)
    #    print len(y)
    #    if addZeroZero:
    #        data.insert(0, 0)
    #        y = list(y)
    #        y.insert(0, 0)

    #    print len(data)
    #    print len(y)

    if extendLeft is not None:
        data.insert(0, extendLeft)  # value (with null probability)
        if not survival:
            y.insert(0, 0.0)  # probability
        else:
            y.insert(0, 1.0)  # probability (CCDF)

    if color is None:
        ax.step(data, y, label=label, linestyle=linestyle, linewidth=2, where="post")
    else:
        ax.step(data, y, label=label, color=color, linestyle=linestyle, linewidth=2, where="post")

    #    ax.plot(data, y, label=label, color=color, linestyle=linestyle)

    #    ax.plot(np.percentile(data, 5), 0.05, 'Dm')
    #    ax.plot(np.percentile(data, 95), 0.95, 'Dm')
    #    ax.plot(np.percentile(data, 50), 0.50, 'Dm')
    #    ax.plot(np.mean(data), y[np.abs(data-np.mean(data)).argmin()], 'Dm')

    #    ax.axvspan(np.percentile(data, 5), np.percentile(data, 95), linewidth=0, alpha=0.20, color='grey')

    print('5-percentile: %.4f' % np.percentile(data, 5))
    print('95-percentile: %.4f' % np.percentile(data, 95))
    print('Median: %.4f' % np.percentile(data, 50))
    #    print('Mean: %.4f' % np.mean(data))

    ax.yaxis.grid(True, ls='--')

    ylabel = "CDF" if not survival else "CCDF"

    ax2 = None

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16)
    if x2label:
        ax2 = ax.twiny()
        ax2.set_xlabel(x2label, fontsize=16)

    if xscale == "log":
        ax.set_xscale("log")
    #        ax.set_xticks([i*10**exp for exp in range(-4, 2) for i in range(1, 10)], minor=True)
    #        ax.set_xticks([10**exp for exp in range(-4, 3) ], minor=False)

    if yscale == "log":
        ax.set_yscale("log")
    #        ax.set_yticks([i*10**exp for exp in range(-1, 9) for i in range(1, 1)], minor=True)

    if xscale == "symlog":
        ax.set_xscale("symlog")

    if majorxTicks:
        ax.set_xticks(majorxTicks, minor=False)

    ax.minorticks_on()
    #    ax.tick_params(axis='x',which='minor',top='off')
    #    ax.tick_params(axis='x',which='major',top='off')
    #    ax.tick_params(axis='y',which='major',right='off')
    #    ax.tick_params(axis='y',which='minor',right='off')
    ax.tick_params(axis='x', which='minor', top=False)
    ax.tick_params(axis='x', which='major', top=False, labelsize=14)
    ax.tick_params(axis='y', which='major', right=False, labelsize=14)
    ax.tick_params(axis='y', which='minor', right=False)

    majorFormatter = FormatStrFormatter('%g')
    #    majorFormatter = FormatStrFormatter('%.2f')
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_major_formatter(majorFormatter)

    if xlim:
        ax.set(xlim=xlim)
    else:
        ax.set(xlim=(min(data), max(data)))

    if x2lim:
        if ax2 is None:
            ax2 = ax.twiny()
        ax2.set(xlim=x2lim)

        ax2.minorticks_on()
        ax2.tick_params(axis='x', which='minor', top=True)
        ax2.tick_params(axis='x', which='major', top=True, labelsize=14)

        ax2.xaxis.set_major_formatter(majorFormatter)

    if ylim:
        ax.set(ylim=ylim)

    if label is not None:
        ax.legend(loc='lower right', fontsize=16) if not survival else ax.legend(loc="upper right", fontsize=16)

    if savefig:
        plt.tight_layout()
        plt.savefig(savefig, format=format, bbox_inches='tight', pad_inches=0.05)


################################################################################################################################################################################################################################################################

def to_bin(val, bin_ranges, dir=False):
    gmin = np.min(bin_ranges)
    gmax = np.max(bin_ranges)
    nb = np.shape(bin_ranges)[0] - 1

    # print(bin_ranges)
    # input()
    val_bin = []
    if dir:

        for x in val:
            if x >= 0.0:
                # print('enter:%s\n' %x)
                if x <= gmin:
                    val_bin.append(nb)
                    # print('[a] exit:%s\n' %nb)
                elif x >= gmax:
                    # print('x=%s------>%s\n'%(x,(2*nb-1)))
                    val_bin.append(2 * nb - 1)
                    # print('[b] exit:%s\n' %(2*nb-1))
                else:
                    for i, v in np.ndenumerate(bin_ranges):
                        if x < v:
                            val_bin.append(nb + i[0] - 1)
                            # print('[c] exit:%s\n' %(nb+i[0]-1))
                            break
            elif x < 0.0:
                # print('enter:%s\n' %x)
                if abs(x) <= gmin:
                    val_bin.append(nb - 1)
                    # print('[a] exit:%s\n' %(nb-1))
                elif abs(x) >= gmax:
                    # print('x=%s------>0\n'%x)
                    val_bin.append(0)
                    # print('[b] exit:0\n')
                else:
                    for i, v in np.ndenumerate(bin_ranges):
                        if abs(x) < v:
                            val_bin.append(nb - i[0])
                            # print('[c] exit:%s\n' %(nb-i[0]))
                            break
                            # print('val:\n%s\n' %val)
        # print('val_bin:\n%s\n' %val_bin)
    else:

        for x in val:
            # print('enter:%s\n' %x)
            if x <= gmin:
                val_bin.append(0)
                # print('[a] exit: 0\n')
            elif x >= gmax:
                # print('x=%s------>%s\n'%(x,(nb-1)))
                val_bin.append(nb - 1)
                # print('[b] exit:%s\n' %(nb-1))
            else:
                for i, v in enumerate(bin_ranges):
                    # print(i,v)
                    if x < v:
                        val_bin.append(i - 1)

                        # print('[c] exit:%s\n' %(i-1))
                        break
    # print('val:\n%s\n' %val)
    # print('val_bin:\n%s\n' %val_bin)
    # input()
    return val_bin


def to_bin_zero_aware(bf, bin_ranges, val_col, dir_col=None, hasdir=False):
    # print('to_bin_zero_aware')
    nbins = len(bin_ranges) - 1
    val = bf[val_col]
    val_bin = []
    # print(val)
    # input()
    # print(bin_ranges)
    # input()

    if hasdir:
        dir = bf[dir_col]
        # print(dir)
        # input()
        for v, d in zip(val, dir):
            # print(d)
            # input()
            vb = -1
            if d == 1:  # Downstream (+1) [nbins, 2*nbins-1]
                for i, (low_range, hig_range) in enumerate(zip(bin_ranges, bin_ranges[1:])):
                    # print(low_range, v, hig_range)
                    # input()
                    try:
                        if low_range <= v < hig_range:
                            # print(low_range, v, hig_range)
                            # input()
                            vb = i + nbins
                            break
                    except:
                        print(low_range, v, hig_range)
                        input()
                if vb == -1:
                    vb = nbins if v < bin_ranges[0] else 2 * nbins - 1
            elif d == 0:  # Upstream (-1)
                for i, (low_range, hig_range) in enumerate(zip(bin_ranges, bin_ranges[1:])):
                    # print(-hig_range, v, -low_range)
                    # input()
                    if -hig_range < v <= -low_range:
                        # print(-hig_range, v, -low_range)
                        # input()
                        vb = nbins - 1 - i
                        break
                if vb == -1:
                    vb = 0 if v < -bin_ranges[-1] else nbins - 1
            val_bin.append(vb)
    else:
        for v in val:
            vb = -1
            for i, (low_range, hig_range) in enumerate(zip(bin_ranges, bin_ranges[1:])):
                if low_range <= v < hig_range:
                    vb = i
                    break
            if vb == -1:
                vb = 0 if v < bin_ranges[0] else nbins - 1
            val_bin.append(vb)
    # print(val_bin)
    # input()
    if len(val) != len(val_bin):
        print('in:%s out:%s' % (len(val), len(val_bin)))
    return val_bin


################################################################################################################################################################################################################################################################

def from_bin(val_bin, bin_ranges, dir=False):
    val = []
    nb = np.shape(bin_ranges)[0] - 1

    if dir:
        for x in val_bin:
            # print(x)
            if x >= nb:
                val.append((bin_ranges[x - nb + 1] + bin_ranges[x - nb]) / 2)
            else:
                val.append((-1) * ((bin_ranges[nb - x] + bin_ranges[(nb - 1) - x]) / 2))
        # print('binned:\n%s\n' %val_bin)
        # print('not binned:\n%s\n' %val)
    else:
        for x in val_bin:
            # val.append((bin_ranges[x+1]-bin_ranges[x])/2)
            val.append((bin_ranges[x + 1] + bin_ranges[x]) / 2)

    # print('binned:\n%s\n' %val_bin)
    # print('not binned:\n%s\n' %val)

    return val


def from_soft_bin(probas_bin, bin_ranges, hasdir=False):
    val = []
    nb = np.shape(bin_ranges)[0] - 1
    if hasdir:
        nb *= 2
    bin_centers = from_bin(range(nb), bin_ranges, hasdir)

    for proba in probas_bin:
        val.append(np.dot(bin_centers, proba))

    return val


def from_soft_bin2(probas_bin, bin_ranges, hasdir=False):
    """
    Differently from from_soft_bin, this function cope with directioned values
    In particular, predictions of values and directions are treated separately.
    """
    if not hasdir:
        return from_soft_bin(probas_bin, bin_ranges)
    val = []
    nb = np.shape(bin_ranges)[0] - 1
    bin_centers = from_bin(range(nb), bin_ranges)

    for proba in probas_bin:
        neg_proba = np.array(
            list(reversed(proba[:nb])))  # Probabilities associated to the negative (upstream) direction
        pos_proba = np.array(proba[nb:])  # Probabilitied associated to the positive (downstream) direction
        bent_proba = neg_proba + pos_proba  # Bent probabilities, to obtain a proba distr only on value
        pos_sign_proba = sum(pos_proba)  # Probability that the value is positive
        # Basing on previous probability, we choose the sign of value, in case of tie, random choice is performed
        sign = 1 if pos_sign_proba > .5 else -1 if pos_sign_proba < .5 else np.random.choice([-1, 1])
        val.append(sign * np.dot(bin_centers, bent_proba))

    return val


def from_soft_bin_joint(probas_bin, pl_bin_ranges, iat_bin_ranges, hasdir=False):
    pl_val = []
    iat_val = []
    pl_nb = np.shape(pl_bin_ranges)[0] - 1
    iat_nb = np.shape(iat_bin_ranges)[0] - 1
    if hasdir:
        pl_nb *= 2
    pl_bins = list(range(pl_nb))
    iat_bins = list(range(iat_nb))
    pl_bin_centers = from_bin(pl_bins, pl_bin_ranges, hasdir)
    iat_bin_centers = from_bin(iat_bins, iat_bin_ranges)

    # We try to built association between joint binned values and disjoint binned values related.
    pl_iat_bins = list(product(pl_bins, iat_bins))
    pl = [v[0] for v in pl_iat_bins]
    iat = [v[1] for v in pl_iat_bins]
    del pl_bins, iat_bins, pl_iat_bins
    pl_iat = generate_joint_binning([iat], [pl], iat_nb)[0]

    joint_disjoint_relation = dict([(pi, (p, i)) for pi, p, i in zip(pl_iat, pl, iat)])

    for proba in probas_bin:
        pl_temp = 0
        iat_temp = 0
        for i, p in enumerate(proba):
            pl_index, iat_index = joint_disjoint_relation[pl_iat[i]]
            pl_temp += pl_bin_centers[pl_index] * p
            iat_temp += iat_bin_centers[iat_index] * p
        pl_val.append(pl_temp)
        iat_val.append(iat_temp)

    return pl_val, iat_val


def from_soft_bin2_joint(probas_bin, pl_bin_ranges, iat_bin_ranges, hasdir=False):
    """
    Per-biflow
    :param probas_bin:
    :param pl_bin_ranges:
    :param iat_bin_ranges:
    :param hasdir:
    :return:
    """
    if not hasdir:
        return from_soft_bin_joint(probas_bin, pl_bin_ranges, iat_bin_ranges)
    pl_val = []
    iat_val = []
    pl_nb = np.shape(pl_bin_ranges)[0] - 1
    iat_nb = np.shape(iat_bin_ranges)[0] - 1
    if hasdir:
        pl_nb *= 2
    pl_bins = list(range(pl_nb))
    iat_bins = list(range(iat_nb))
    pl_bin_centers = from_bin(pl_bins, pl_bin_ranges, hasdir)
    iat_bin_centers = from_bin(iat_bins, iat_bin_ranges)

    # We try to built association between joint binned values and disjoint binned values related.
    pl_iat_bins = list(product(pl_bins, iat_bins))
    pl = [v[0] for v in pl_iat_bins]
    iat = [v[1] for v in pl_iat_bins]
    del pl_bins, iat_bins, pl_iat_bins
    pl_iat = generate_joint_binning([iat], [pl], iat_nb)[0]

    joint_disjoint_relation = dict([(pi, (p, i)) for pi, p, i in zip(pl_iat, pl, iat)])

    for proba in probas_bin:
        pl_temp = 0
        iat_temp = 0
        pos_proba = 0
        for i, p in enumerate(proba):
            pl_index, iat_index = joint_disjoint_relation[pl_iat[i]]
            pl_temp += abs(pl_bin_centers[pl_index] * p)
            iat_temp += iat_bin_centers[iat_index] * p
            if pl_index >= pl_nb / 2:
                pos_proba += p
        sign = 1 if pos_proba > .5 else -1 if pos_proba < .5 else np.random.choice([-1, 1])
        pl_val.append(sign * pl_temp)
        iat_val.append(iat_temp)

    return pl_val, iat_val


################################################################################################################################################################################################################################################################
# genera una matrice di transizione
# Input:
#        lists:        lista di liste, ciascuna relativa ad un biflusso
#        nbins:        numero di bins per i valori in input. Determina la dimensione della matrice in output
#        dir:          False indica che i valori sono tutti positivi. True indica che ai valori sono state applicate le direzioni sfruttando il segno +/-
#        savepath:     stringa che identifica il percorso in cui si vuole salvare la matrice in formato .csv
# Output:
#        tm:            matrice di transizione
def generate_tm(lists, nbins, dir=False, savepath=None, step=1):
    tm = np.zeros((2 * nbins, 2 * nbins), dtype=float, order='C') if dir else np.zeros((nbins, nbins), dtype=float,
                                                                                       order='C')
    for l in lists:
        for (i, j) in [(l[k], l[k + 1]) for k in range(0, len(l) - 1, step)]:
            tm[int(i), int(j)] += 1
    for row in tm:
        s = float(sum(row))
        if s > 0:
            row[:] = [f / s for f in row]
    if savepath:
        np.savetxt(savepath, tm, fmt='%.18f', delimiter=',', newline='\n', header='', footer='', comments='# ',
                   encoding=None)
    return tm


# def generate_nth_order_tm_ola(lists, nbins, order=1, savepath=None):/
#     nth_order_tm = np.zeros((nbins,) * (order + 1), dtype=float, order='C')
#     nth_order_transitions = [tuple(bf[i:i + order + 1]) for bf in lists for i in range(len(bf) - order)]
#     nth_order_transitions_count = dict(
#         (tuple(k), v) for k, v in zip(*np.unique(nth_order_transitions, return_counts=True, axis=0)))
#     # tuples = set(permutations(list(range(nbins)) * order, order))
#     # tuples = [v for v in product(*[range(nbins)] * order)] <--- FASTER
#     nth_order_tm_rows = dict([(k, np.zeros(nbins)) for k in [v for v in product(*[range(nbins)] * order)]])
#     for key in nth_order_tm_rows:
#         for i in range(nbins):
#             nth_order_tm_rows[key][i] = nth_order_transitions_count.get(key + (i,), 0)
#         nth_order_tm_rows[key] = [nth_order_transitions_count.get(key + (i,), 0) for i in range(nbins)]
#         row_sum = np.sum(nth_order_tm_rows[key])
#         if row_sum:
#             nth_order_tm_rows[key] = nth_order_tm_rows[key] / row_sum
#         for i in range(nbins):
#             nth_order_tm[key + (i,)] = nth_order_tm_rows[key][i]
#     print('tm_shape = %s' % list(nth_order_tm.shape))
#     if savepath:
#         np.savetxt(savepath, nth_order_tm, fmt='%.18f', delimiter=',', newline='\n', header='', footer='',
#                    comments='# ',
#                    encoding=None)
#     return nth_order_tm


def generate_nth_order_tm(lists, nbins, order=1, savepath=None):
    # nth_order_tm = np.zeros((nbins,) * (order + 1), dtype=float, order='C')
    nth_order_tm = dict()
    nth_order_transitions = [tuple(bf[i:i + order + 1]) for bf in lists for i in range(len(bf) - order)]
    nth_order_transitions_count = dict(
        (tuple(k), v) for k, v in zip(*np.unique(nth_order_transitions, return_counts=True, axis=0)))
    nth_order_histories = [k[:-1] for k in nth_order_transitions_count.keys()]
    for key in nth_order_histories:
        row = np.asarray([nth_order_transitions_count.get(key + (i,), 0) for i in range(nbins)])
        row_sum = np.sum(row)
        # No control on row_sum, because key + (i,) is in nth_order_transitions_count at least one time w/ a not 0 value
        row = row / row_sum
        nth_order_tm[key] = row
    # print('tm_shape = %s' % list(nth_order_tm.shape))
    print('tm_shape = %s + 1' % len(list(nth_order_tm.keys())[0]))
    # if savepath:
    #     np.savetxt(savepath, nth_order_tm, fmt='%.18f', delimiter=',', newline='\n', header='', footer='',
    #                comments='# ', encoding=None)
    return nth_order_tm


def generate_mean_tm(lists, averageby, nbins, dir=False, savepath=None):
    if isinstance(averageby, list):
        averageby = np.asarray(averageby)
    averageby_unique = np.unique(averageby)

    tms = []

    for avgby in averageby_unique:
        group_index = np.where(averageby == avgby)[0]
        group = [lists[i] for i in group_index]
        tms.append(generate_tm(group, nbins, dir))

    mean_tm = np.mean(tms, axis=0)

    return mean_tm


def generate_joint_trans_distr(lists, nbins, dir=False, savepath=None, step=1):
    if dir:
        jtd = np.zeros((2 * nbins, 2 * nbins), dtype=float, order='C')
    else:
        jtd = np.zeros((nbins, nbins), dtype=float, order='C')

    for l in lists:
        for (i, j) in [(l[k], l[k + 1]) for k in range(0, len(l) - 1, step)]:
            jtd[int(i), int(j)] += 1

    s = np.sum(jtd)
    if s:
        jtd = jtd / s

    if savepath:
        np.savetxt(savepath, jtd, fmt='%.18f', delimiter=',', newline='\n', header='', footer='', comments='# ',
                   encoding=None)

    return jtd


def generate_initial_distr(lists, nbins, dir=False, savepath=None):
    if dir:
        init_distr = np.zeros(2 * nbins, dtype=float, order='C')
    else:
        init_distr = np.zeros(nbins, dtype=float, order='C')

    for l in lists:
        try:
            init_distr[l[0]] += 1
        except IndexError:
            pass

    s = np.sum(init_distr)
    if s:
        init_distr = init_distr / s

    if savepath:
        np.savetxt(savepath, init_distr, fmt='%.18f', delimiter=',', newline='\n', header='', footer='', comments='# ',
                   encoding=None)

    return init_distr


def generate_stationary_distr(lists, nbins, dir=False, savepath=None):
    # Compute transition matrix
    tm = generate_tm(lists, nbins, dir)
    # Compute left eigvector of transition matrix
    eigv, left_eig = eig(tm, left=True, right=False)
    eigv1_index = []
    # Some eigenvalues 1 present a queue of some decimals
    for n_decimals in reversed(range(1, 6)):
        eigv = np.asarray([np.round(e, n_decimals) for e in eigv])
        eigv1_index = np.where(eigv == 1)[0]
        if len(eigv1_index) != 0:
            break
    if len(eigv1_index) == 0:
        print('eigv', np.max(eigv))
        input()
    if len(eigv1_index) > 1:
        print('Warning: more than one eigenvector is associated with eigenvalue 1. Markov Chain is reducible.')
        return
    try:
        warnings.simplefilter('error')
        non_normalized_stat_distr = np.asarray([float(v) for v in left_eig[:, eigv1_index[0]]])
        warnings.simplefilter('ignore')
    except np.ComplexWarning as w:
        print(w, savepath, file=sys.stderr)
        warnings.simplefilter('ignore')
        non_normalized_stat_distr = np.asarray([float(v) for v in left_eig[:, eigv1_index[0]]])
    if not ((non_normalized_stat_distr >= 0).all() or (non_normalized_stat_distr <= 0).all()):
        print('Warning: some values of eigenvector has opposite sign w.r.t. the others. Markov Chain is reducible.')
        return
    stat_distr = non_normalized_stat_distr / np.sum(non_normalized_stat_distr)
    if savepath:
        np.savetxt(savepath, stat_distr, fmt='%.18f', delimiter=',', newline='\n', header='', footer='', comments='# ',
                   encoding=None)
    return stat_distr


################################################################################################################################################################################################################################################################
def generate_PL_df(df, savepath=None):
    # creo una DF per eseguire le analisi sulle PL
    transition_matrix_df_PL = df.copy()
    del transition_matrix_df_PL['iat']

    transition_matrix_df_PL['L4_payload_bytes_dir'] = ""
    transition_matrix_df_PL['L4_payload_bytes_dir'] = transition_matrix_df_PL['L4_payload_bytes_dir'].astype('object')

    # PL elimino le direzioni relative ai pacchetti con payload nullo
    for i, subrow in transition_matrix_df_PL[['L4_payload_bytes', 'packet_dir']].iterrows():
        transition_matrix_df_PL.at[i, 'packet_dir'] = ['n' if a == 0 else b for (a, b) in
                                                       zip(subrow['L4_payload_bytes'], subrow['packet_dir'])]

    # PL elimino dalle sequenze L4_payload_bytes i pkt con payload nullo
    transition_matrix_df_PL['L4_payload_bytes'] = transition_matrix_df_PL['L4_payload_bytes'].apply(remove_zero_pl)

    # PL elimino dalle sequenze packet_dir i valori n
    transition_matrix_df_PL['packet_dir'] = transition_matrix_df_PL['packet_dir'].apply(remove_item, args=('n'))

    # PL elimina dal DF tutte le righe in cui compare una lista vuota in corrispondenza della colonna 'L4_payload_bytes'
    index = [i for i, v in transition_matrix_df_PL[['L4_payload_bytes']].iterrows() if len(v[0]) == 0]
    transition_matrix_df_PL.drop(index, inplace=True)

    # PL assegno i segni alle pl a seconda della direzione [Up=negativo Down=positivo]
    for i, subrow in transition_matrix_df_PL[['L4_payload_bytes', 'packet_dir']].iterrows():
        transition_matrix_df_PL.at[i, 'L4_payload_bytes_dir'] = [a if b == 1 else (-1 * a) for (a, b) in
                                                                 zip(subrow['L4_payload_bytes'], subrow['packet_dir'])]

    # PL aggiunge una colonna alla tabella chiamata L4_payload_bytes_bin che serve a memorizzare le L4_payload_bytes sottoposte a binning
    # transition_matrix_df_PL['L4_payload_bytes_bin'] = transition_matrix_df_PL['L4_payload_bytes'].apply(to_bin, args=(bin_ranges, ))

    # PL aggiunge una colonna alla tabella chiamata L4_payload_bytes_bin_dir che serve a memorizzare le L4_payload_bytes_dir sottoposte a binning
    # transition_matrix_df_PL['L4_payload_bytes_dir_bin'] = transition_matrix_df_PL['L4_payload_bytes_dir'].apply(to_bin, args=(bin_ranges, True))

    if savepath:
        # PL esporto la DF relativa alle PL in un file csv
        transition_matrix_df_PL.to_csv(path_or_buf=savepath, sep=';', na_rep=np.nan, header=True, index=False,
                                       decimal='.')

    return transition_matrix_df_PL


################################################################################################################################################################################################################################################################
def generate_IAT_df(df, savepath=None):
    ############################################################### PREPARAZIONE DF PER IAT
    # creo una DF per eseguire le analisi sugli IAT
    df_IAT = df.copy()

    # del transition_matrix_df_IAT['L4_payload_bytes']

    df_IAT['iat_micros'] = ""
    df_IAT['iat_micros'] = df_IAT['iat_micros'].astype('object')

    df_IAT['iat_micros_dir'] = ""
    df_IAT['iat_micros_dir'] = df_IAT['iat_micros_dir'].astype('object')

    df_IAT['iat_micros_dir_rsign'] = ""
    df_IAT['iat_micros_dir_rsign'] = df_IAT['iat_micros_dir_rsign'].astype('object')

    # IAT elimina dal DF tutte le righe in cui compare una lista vuota in corrispondenza della colonna 'iat'
    index = [i for i, v in df_IAT[['iat']].iterrows() if len(v[0]) == 0]
    df_IAT.drop(index, inplace=True)

    # IAT in questo caso gli iat relativi ai pacchetti con pl nulla vengono contrassegnati con *
    for i, subrow in df_IAT[['L4_payload_bytes', 'iat']].iterrows():
        df_IAT.at[i, 'iat'] = ['*%s' % b if a == 0 else b for (a, b) in zip(subrow['L4_payload_bytes'], subrow['iat'])]
    df_IAT['iat'] = df_IAT['iat'].apply(compute_delete_iat)

    # IAT elimina dal DF tutte le righe in cui compare una lista vuota in corrispondenza della colonna 'iat_micros'
    index = [i for i, v in df_IAT[['iat']].iterrows() if len(v[0]) == 0]
    df_IAT.drop(index, inplace=True)

    iats = df_IAT['iat'].values
    iats = to_single_list(iats)
    iats = np.asarray(iats, dtype=np.float32)

    iat_1p = 1e-6
    iat_99p = np.percentile(iats, 99)
    print('IAT 99p: %s\n' % iat_99p)

    # converto i valori degli iat in micros, i valori piu' piccoli di 1 micros vengono settati al valore 1.05 micros
    df_IAT['iat_micros'] = df_IAT['iat'].apply(to_micros_with_saturation, args=(iat_1p, iat_99p,))

    # ---------------------------------------------------------------------------------------->
    # ---------------------------------------------------------------------------------------->
    # ---------------------------------------------------------------------------------------->
    # PL elimino le direzioni relative ai pacchetti con payload nullo
    for i, subrow in df_IAT[['L4_payload_bytes', 'packet_dir']].iterrows():
        df_IAT.at[i, 'packet_dir'] = ['n' if a == 0 else b for (a, b) in
                                      zip(subrow['L4_payload_bytes'], subrow['packet_dir'])]
    # PL elimino dalle sequenze packet_dir i valori n
    df_IAT['packet_dir'] = df_IAT['packet_dir'].apply(remove_item, args=('n'))

    # IAT in questo caso gli iat relativi ai pacchetti con pl nulla vengono contrassegnati con *
    # for i, subrow in df_IAT[['L4_payload_bytes','iat_micros']].iterrows():
    #    df_IAT.at[i,'iat_micros']= ['*%s'%b if a==0 else b for (a,b) in zip(subrow['L4_payload_bytes'], subrow['iat_micros'])]
    # df_IAT['iat_micros'] = df_IAT['iat_micros'].apply(compute_delete_iat)

    del df_IAT['L4_payload_bytes']

    # IAT elimina dal DF tutte le righe in cui compare una lista vuota in corrispondenza della colonna 'iat_micros'
    # index=[i for i,v in df_IAT[['iat_micros']].iterrows() if len(v[0]) == 0 ]
    # df_IAT.drop( index, inplace=True )

    # ---------------------------------------------------------------------------------------->
    # ---------------------------------------------------------------------------------------->
    # ---------------------------------------------------------------------------------------->

    # IAT assegno i segni agli iat a seconda della direzione [Up=negativo Down=positivo]
    for i, subrow in df_IAT[['iat_micros', 'packet_dir']].iterrows():
        df_IAT.at[i, 'iat_micros_dir'] = [a if b == 1 else (-1 * a) for (a, b) in
                                          zip(subrow['iat_micros'], subrow['packet_dir'])]

    # IAT [FULL_DIR_RSIGN][UP][DOWN]
    df_IAT['iat_micros_dir_rsign'] = df_IAT['iat_micros_dir'].apply(compute_iat)

    # IAT elimino il primo IAT in ciascun biflusso
    df_IAT['iat_micros'] = df_IAT['iat_micros'].apply(remove_first_iat)
    df_IAT['iat_micros_dir'] = df_IAT['iat_micros_dir'].apply(remove_first_iat)

    del df_IAT['packet_dir']

    # IAT elimina dalla tabella i flussi con valori mancanti
    df_IAT = df_IAT.dropna()

    if savepath:
        # IAT esporto la DF relativa agli IAT in un file csv
        df_IAT.to_csv(path_or_buf=savepath, sep=';', na_rep=np.nan, header=True, index=False, decimal='.')

    return df_IAT


################################################################################################################################################################################################################################################################
def generate_PL_IAT_df(df, savepath=None):
    ############################################################### PREPARAZIONE DF PER PL E IAT CONGIUNTE
    # creo una DF per eseguire le analisi su PL e IAT congiunte.
    df_PL_IAT = df.copy()

    df_PL_IAT['iat_micros'] = ""
    df_PL_IAT['iat_micros'] = df_PL_IAT['iat_micros'].astype('object')

    df_PL_IAT['iat_micros_dir'] = ""
    df_PL_IAT['iat_micros_dir'] = df_PL_IAT['iat_micros_dir'].astype('object')

    df_PL_IAT['iat_micros_dir_rsign'] = ""
    df_PL_IAT['iat_micros_dir_rsign'] = df_PL_IAT['iat_micros_dir_rsign'].astype('object')

    # IAT elimina dal DF tutte le righe in cui compare una lista vuota in corrispondenza della colonna 'iat'
    index = [i for i, v in df_PL_IAT[['iat']].iterrows() if len(v[0]) == 0]
    df_PL_IAT.drop(index, inplace=True)

    # IAT in questo caso gli iat relativi ai pacchetti con pl nulla vengono contrassegnati con *
    for i, subrow in df_PL_IAT[['L4_payload_bytes', 'iat']].iterrows():
        df_PL_IAT.at[i, 'iat'] = ['*%s' % b if a == 0 else b for (a, b) in
                                  zip(subrow['L4_payload_bytes'], subrow['iat'])]
    df_PL_IAT['iat'] = df_PL_IAT['iat'].apply(compute_delete_iat)

    # IAT elimina dal DF tutte le righe in cui compare una lista vuota in corrispondenza della colonna 'iat_micros'
    index = [i for i, v in df_PL_IAT[['iat']].iterrows() if len(v[0]) == 0]
    df_PL_IAT.drop(index, inplace=True)

    iats = df_PL_IAT['iat'].values
    iats = to_single_list(iats)
    iats = np.asarray(iats, dtype=np.float32)

    iat_1p = 1e-6
    iat_99p = np.percentile(iats, 99)
    print('IAT 99p: %s\n' % iat_99p)

    # converto i valori degli iat in micros, i valori piu' piccoli di 1 micros vengono settati al valore 1.05 micros
    df_PL_IAT['iat_micros'] = df_PL_IAT['iat'].apply(to_micros_with_saturation, args=(iat_1p, iat_99p,))

    # ---------------------------------------------------------------------------------------->
    # ---------------------------------------------------------------------------------------->
    # ---------------------------------------------------------------------------------------->
    # PL elimino le direzioni relative ai pacchetti con payload nullo
    for i, subrow in df_PL_IAT[['L4_payload_bytes', 'packet_dir']].iterrows():
        df_PL_IAT.at[i, 'packet_dir'] = ['n' if a == 0 else b for (a, b) in
                                         zip(subrow['L4_payload_bytes'], subrow['packet_dir'])]
    # PL elimino dalle sequenze packet_dir i valori n
    df_PL_IAT['packet_dir'] = df_PL_IAT['packet_dir'].apply(remove_item, args=('n'))

    # ---------------------------------------------------------------------------------------->
    # ---------------------------------------------------------------------------------------->
    # ---------------------------------------------------------------------------------------->

    # IAT assegno i segni agli iat a seconda della direzione [Up=negativo Down=positivo]
    for i, subrow in df_PL_IAT[['iat_micros', 'packet_dir']].iterrows():
        df_PL_IAT.at[i, 'iat_micros_dir'] = [a if b == 1 else (-1 * a) for (a, b) in
                                             zip(subrow['iat_micros'], subrow['packet_dir'])]

    # IAT [FULL_DIR_RSIGN][UP][DOWN]
    df_PL_IAT['iat_micros_dir_rsign'] = df_PL_IAT['iat_micros_dir'].apply(compute_iat)

    # IAT elimino il primo IAT in ciascun biflusso
    df_PL_IAT['iat_micros'] = df_PL_IAT['iat_micros'].apply(remove_first_iat)
    df_PL_IAT['iat_micros_dir'] = df_PL_IAT['iat_micros_dir'].apply(remove_first_iat)

    # IAT elimina dalla tabella i flussi con valori mancanti
    df_PL_IAT = df_PL_IAT.dropna()

    df_PL_IAT['L4_payload_bytes_dir'] = ""
    df_PL_IAT['L4_payload_bytes_dir'] = df_PL_IAT['L4_payload_bytes_dir'].astype('object')

    # PL elimino le direzioni relative ai pacchetti con payload nullo
    for i, subrow in df_PL_IAT[['L4_payload_bytes', 'packet_dir']].iterrows():
        df_PL_IAT.at[i, 'packet_dir'] = ['n' if a == 0 else b for (a, b) in
                                         zip(subrow['L4_payload_bytes'], subrow['packet_dir'])]

    # PL elimino dalle sequenze L4_payload_bytes i pkt con payload nullo
    df_PL_IAT['L4_payload_bytes'] = df_PL_IAT['L4_payload_bytes'].apply(remove_zero_pl)

    # PL elimino dalle sequenze packet_dir i valori n
    df_PL_IAT['packet_dir'] = df_PL_IAT['packet_dir'].apply(remove_item, args=('n'))

    # PL elimina dal DF tutte le righe in cui compare una lista vuota in corrispondenza della colonna 'L4_payload_bytes'
    index = [i for i, v in df_PL_IAT[['L4_payload_bytes']].iterrows() if len(v[0]) == 0]
    df_PL_IAT.drop(index, inplace=True)

    # PL assegno i segni alle pl a seconda della direzione [Up=negativo Down=positivo]
    for i, subrow in df_PL_IAT[['L4_payload_bytes', 'packet_dir']].iterrows():
        df_PL_IAT.at[i, 'L4_payload_bytes_dir'] = [a if b == 1 else (-1 * a) for (a, b) in
                                                   zip(subrow['L4_payload_bytes'], subrow['packet_dir'])]

    if savepath:
        # IAT esporto la DF relativa agli IAT in un file csv
        df_PL_IAT.to_csv(path_or_buf=savepath, sep=';', na_rep=np.nan, header=True, index=False, decimal='.')

    return df_PL_IAT


################################################################################################################################################################################################################################################################


def plotECDF_Zoom(data_in, ax, xlabel=None, label=None, survival=None, color=None, linestyle="solid", xscale="lin",
                  yscale="lin", xlim=None, ylim=(0, 1), majorxTicks=None, extendLeft=0, savefig=None, format="pdf",
                  zoom=None, axins=None):
    #    if color == None:
    #        plt.rc('lines', linewidth=4)
    #        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k', 'm'])))
    #
    #            +cycler('linestyle', ['-', '--', ':', '-.'])))

    # data = list(data)
    data = data_in.copy()
    data.sort()

    # Remove traces with duration < 10 seconds
    #    data = [x for x in data if x >= 10]

    ecdf = ECDF(data)
    y = ecdf(data) if not survival else 1 - ecdf(data)
    y = list(y)

    #    print len(data)
    #    print len(y)
    #    if addZeroZero:
    #        data.insert(0, 0)
    #        y = list(y)
    #        y.insert(0, 0)

    #    print len(data)
    #    print len(y)

    if extendLeft != None:
        data.insert(0, extendLeft)  # value (with null probability)
        if not survival:
            y.insert(0, 0.0)  # probability
        else:
            y.insert(0, 1.0)  # probability (CCDF)

    if color == None:
        ax.step(data, y, label=label, linestyle=linestyle, linewidth=2, where="post")
    else:
        ax.step(data, y, label=label, color=color, linestyle=linestyle, linewidth=2, where="post")

    if zoom and axins:

        if color == None:
            axins.step(data, y, linestyle=linestyle, linewidth=2, where="post")
        else:
            axins.step(data, y, color=color, linestyle=linestyle, linewidth=2, where="post")

    #    ax.plot(data, y, label=label, color=color, linestyle=linestyle)

    #    ax.plot(np.percentile(data, 5), 0.05, 'Dm')
    #    ax.plot(np.percentile(data, 95), 0.95, 'Dm')
    #    ax.plot(np.percentile(data, 50), 0.50, 'Dm')
    #    ax.plot(np.mean(data), y[np.abs(data-np.mean(data)).argmin()], 'Dm')

    #    ax.axvspan(np.percentile(data, 5), np.percentile(data, 95), linewidth=0, alpha=0.20, color='grey')

    print('5-percentile: %.4f' % np.percentile(data, 5))
    print('95-percentile: %.4f' % np.percentile(data, 95))
    print('Median: %.4f' % np.percentile(data, 50))
    #    print('Mean: %.4f' % np.mean(data))

    ax.yaxis.grid(True, ls='--')

    ylabel = "CDF" if not survival else "CCDF"

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)

    if xscale == "log":
        ax.set_xscale("log")
    #        ax.set_xticks([i*10**exp for exp in range(-4, 2) for i in range(1, 10)], minor=True)
    #        ax.set_xticks([10**exp for exp in range(-4, 3) ], minor=False)

    if yscale == "log":
        ax.set_yscale("log")
    #        ax.set_yticks([i*10**exp for exp in range(-1, 9) for i in range(1, 1)], minor=True)

    if xscale == "symlog":
        ax.set_xscale("symlog")

    if majorxTicks:
        ax.set_xticks(majorxTicks, minor=False)

    ax.minorticks_on()
    #    ax.tick_params(axis='x',which='minor',top='off')
    #    ax.tick_params(axis='x',which='major',top='off')
    #    ax.tick_params(axis='y',which='major',right='off')
    #    ax.tick_params(axis='y',which='minor',right='off')
    ax.tick_params(axis='x', which='minor', top=False)
    ax.tick_params(axis='x', which='major', top=False, labelsize=12)
    ax.tick_params(axis='y', which='major', right=False, labelsize=12)
    ax.tick_params(axis='y', which='minor', right=False)

    majorFormatter = FormatStrFormatter('%g')
    #    majorFormatter = FormatStrFormatter('%.2f')
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_major_formatter(majorFormatter)

    if xlim:
        ax.set(xlim=xlim)
    else:
        ax.set(xlim=(min(data), max(data)))

    if ylim:
        ax.set(ylim=ylim)

    #    ax.set(xlim=xlim)
    #    legend = ax.legend(loc='lower right') if not survival else ax.legend(loc="upper right")

    if savefig:
        plt.tight_layout()
        plt.savefig(savefig, format=format, bbox_inches='tight', pad_inches=0.05)


################################################################################################################################################################################################################################################################
# genera una matrice di transizione
# Input:
#        lists:        lista di liste, ciascuna relativa ad un biflusso
#        nbins:        numero di bins per i valori in input. Determina la dimensione della matrice in output
#        dir:          False indica che i valori sono tutti positivi. True indica che ai valori sono state applicate le direzioni sfruttando il segno +/-
#        savepath:     stringa che identifica il percorso in cui si vuole salvare la matrice in formato .csv
# Output:
#        tm:            matrice di transizione
def generate_tm_2M(lists, nbins, dir=False, savepath=None):
    if dir:
        tm = np.zeros((2 * nbins, 2 * nbins, 2 * nbins), dtype=float, order='C')
    else:
        tm = np.zeros((nbins, nbins, nbins), dtype=float, order='C')

    for l in lists:
        for (i, j, k) in zip(l, l[1:], l[2:]):
            tm[i][j][k] += 1

    tiles = [list(zip(np.range(np.shape(tm)[0]), p)) for p in permutations(np.range(np.shape(tm)[1]))]
    print(tiles)

    for tile in tiles:
        s = float(sum(tm[tile[0]][tile[1]][:]))
        if s > 0:
            tm[tile[0]][tile[1]][:] = [f / s for f in tm[tile[0]][tile[1]][:]]

    return tm


'''
################################################################################################################################################################################################################################################################
#compara due liste di valori----> calcola MSE, RMSE e R^2, genera un grafico di regressione
#Input:
#        ax:         ax di matplotlib
#        x:          lista dei valori per ascisse
#        y:          lista dei valori per le ordinate  
#        xlabel:     label delle ascisse
#        ylabel:     label delle ordinate
#        title:      titolo
#        nb
def compare(ax, x,y, xlabel, ylabel, title, nb):
    
    data={xlabel: x, ylabel: y }
    df=pd.DataFrame(data=data)
    
    
    mse=mean_squared_error(x,y)
    
    #RMSE è una misura della deviazione media delle stime dai valori osservati
    #con RMSE sappiamo esplicitamente quanto le nostre previsioni si discostano, in media, dai valori effettivi nel set di dati.
    rmse=sqrt(mse)
    
    #R2 è la frazione della somma totale dei quadrati che è "spiegata" dalla regressione
    r2=r2_score(x,y)
    print('%s\nMSE: %s\nRMSE; %s\nR2: %s\n'%(title,mse,rmse,r2))
    #print('%s\nRMSE: %.2f\n'%(title,rmse))
    plt.title(title, loc='center')
    #print('nb: %s \n'%nb)
    anchored_text = AnchoredText('MSE: %.2f\nRMSE: %.2f\nR2: %.2f'%(mse,rmse,r2), loc='upper left',bbox_to_anchor=(1.,1.), bbox_transform=ax.transAxes, frameon=True)
    
    #anchored_text = AnchoredText('RMSE: %.2f\n'%rmse, loc=2, frameon=True)
    #sns.residplot(x=xlabel, y=ylabel, data=df, ax=ax, lowess=True)

    
    sns.regplot(x=xlabel, y=ylabel, data=df, ax=ax)
    ax.add_artist(anchored_text)
    plt.grid(True, linestyle='--', linewidth=1, alpha=0.5)
    plt.show()
'''


################################################################################################################################################################################################################################################################
# Gli IAT relativi ai pacchetti con PL nulla sono contrassegnati con *. La funzione ricalcola gli IAT per i pacchetti con PL non nullo
# Input:
#        iats:        lista di iat
# Output:
#        iats_new:    lista di iat
def compute_delete_iat(iats):
    iats_new = []

    w = 0
    for i in iats:
        # print(type(i),i)
        if '*' in str(i):
            w = w + float(i.replace('*', ''))
            # print(i,w)
        else:
            # print(i,(i+w))
            iats_new.append(i + w)
            w = 0
    # print('\n')
    # l'iat del primo pacchetto all'interno del biflusso deve essere 0.0
    if len(iats_new) > 0:
        iats_new[0] = 0.0

    return iats_new


# BY JMPR ############################################################################


def generate_joint_binning(iat_values_binned, pl_values_binned, iat_nbins, dir=False):
    # TODO: eliminate the next two rows.
    # if dir:
    #     iat_nbins = 2 * iat_nbins
    values_binned = np.zeros((len(iat_values_binned),), dtype=object)
    for i, (iat_value_binned, pl_value_binned) in enumerate(zip(iat_values_binned, pl_values_binned)):
        iat_value_binned = np.asarray(iat_value_binned)
        pl_value_binned = np.asarray(pl_value_binned)
        # print(iat_value_binned, pl_value_binned, iat_nbins)
        # input()
        values_binned[i] = list(iat_value_binned + pl_value_binned * iat_nbins)
    return values_binned


def unroll_joint_binning(values_binned, iat_nbins, dir=False):
    # if dir:
    #     iat_nbins = 2 * iat_nbins
    iat_values_binned = np.zeros((len(values_binned),), dtype=object)
    pl_values_binned = np.zeros((len(values_binned),), dtype=object)
    for i, value_binned in enumerate(values_binned):
        value_binned = np.asarray(value_binned)
        iat_values_binned[i] = list(value_binned % iat_nbins)
        pl_values_binned[i] = list(np.asarray(np.floor(value_binned / iat_nbins), dtype=int))
    return iat_values_binned, pl_values_binned


def make_dir(dir):
    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            traceback.print_exc(file=sys.stderr)


def get_bin_ranges(bin_ranges_filename, return_counts=False):
    with open(bin_ranges_filename, mode='r') as bin_ranges_file:
        csv_reader = csv.reader(bin_ranges_file, delimiter=',')
        next(csv_reader)
        bin_ranges = []
        for row in csv_reader:
            bin_ranges.append(float(row[0]))
    nbins = len(bin_ranges) - 1
    if return_counts:
        return bin_ranges, nbins
    return bin_ranges


def clean_df(df, field, n_packets=1):
    df = df[df[field].str.len() > n_packets].copy()
    return df


def remove_first_packet_of_fields(df, fields, dir=''):
    if not isinstance(fields, list):
        fields = [fields]
    for field in fields:
        df[field + dir] = df[field + dir].apply(lambda x: x[1:])
    return df


def VersionsSorter(versions):
    islist = False
    if isinstance(versions, list):
        islist = True
        versions = np.asarray(versions)
    sorting_index = np.argsort(versions)
    if isinstance(versions[0], str):
        sorting_index = np.argsort([int(v) for v in versions])
    if islist:
        return list(versions[sorting_index])
    else:
        return versions[sorting_index]


def split_without_replacement(n_population, n_samples, random_state=None):
    '''
    :param n_population: number of observations. Indexes came from the range [0, n_population).
    :param n_samples: number of samples. The number of unique random indexes from the range [0, n_population).
    :param random_state: seed of randomness.
    :return: two non-overlapping random set of indexes, each one of size n_samples.
    '''
    # Generating a random set of indexes of size n_samples from a population of n_population observations.
    indexes_0 = list(sample_without_replacement(n_population, n_samples, random_state=random_state))
    # Computing the complementary set of indexes
    complementary = list(set(range(n_population)) - set(indexes_0))
    # Generating a random index for the complementary one
    compl_index = sample_without_replacement(len(complementary), n_samples, random_state=random_state)
    # Save the random index from the complementary one
    indexes_1 = [complementary[i] for i in compl_index]
    return [indexes_0, indexes_1]


def compute_tolerance(X_actual, X_predicted, range):
    """
    :param X_actual:
    :param X_predicted:
    :param range:
    :return: tol = (|X_actual - X_predicted| / range) * 100
    """
    X_actual = np.asarray(X_actual)
    X_predicted = np.asarray(X_predicted)

    tol = (np.absolute(X_actual - X_predicted) / range) * 100
    tol[tol > 100] = 100

    return tol


def compute_abs_tolerance(X_actual, X_predicted, range):
    """
    :param X_actual:
    :param X_predicted:
    :param range:
    :return: tol = (|X_actual - X_predicted| / range) * 100
    """
    X_actual = np.asarray(X_actual)
    X_predicted = np.asarray(X_predicted)

    abs_tol = (np.absolute(X_actual - X_predicted))
    abs_tol[abs_tol > range] = range

    return abs_tol


def from_log(x):
    """
    :param x: log-value (e.g., log_10(iat))
    :return: 10^x
    """
    return 10 ** x


def dataset_split(n_samples, y=None, k=10, random_state=0):
    """
    :param n_samples: number of samples compose the dataset
    :param y: labels, for stratified splitting
    :param k: number of splits
    :param random_state: random state for the splitting function
    :return:
    """
    splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
    indexes = [(train_index, test_index) for train_index, test_index in splitter.split(np.zeros((n_samples, 1)))]
    return indexes






#UPDATE IDIO SINCE 22/05/2021

#*******************************************************************************************************************
#Utility
#*******************************************************************************************************************

def check_if_string_in_file(file_name, string_to_search):
    """ Check if any line in the file contains given string """
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            if string_to_search in line:
                return True
    return False

def search_string_in_file(file_name, string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                list_of_results.append((line_number, line.rstrip()))
    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results

def check_quintuple(fn,q,cpath):
    qsplit=q.split(',')
    src=qsplit[:2]
    dst=qsplit[2:4]
    proto=qsplit[-1]
    
    net=get_netstat_log_path(fn,cpath)
    #print(net)
    src=':'.join(src)
    dst=':'.join(dst)
    
    csrc=check_if_string_in_file(net,src)
    cdst=check_if_string_in_file(net,dst)
    #print(net,csrc,cdst)
    if csrc and cdst:
        return True
    else:
        #print(src,dst)
        return False
    
def get_timestamp(fn):
    fns=fn.split('_')[0]
    fns=fns.split('/')[-1]
    
    return fns

def get_netstat_log_path(fn,cpath):
    ts=get_timestamp(fn)
    return cpath+'/'+ts+'/netstat.log'
  
def get_duration(t0,tn,unit=1):
    #d0=datetime.fromtimestamp(t0)
    #dn=datetime.fromtimestamp(tn)
    #print(t0,d0)
    #print(tn,dn)
    #elapsed=(dn-d0).microseconds
    elapsed=(tn-t0)*unit
    return elapsed


def get_ratio_4_barchart(df,group_on,hue,to_perc=False,value2count=None):
    #group_on, quello che vuoi sulle righe
    #hue, quello che vuoi sulle colonne, 1 colonna per ciascun valore della colonna hue
    df_cols=list(df.columns)
    if group_on in df_cols and hue in df_cols:
        hues=list(df[hue].unique())
        df_p=pd.DataFrame(columns=[group_on]+hues)
        for group in df[group_on].unique():
            cdict={group_on:group,}
            counts=[df[(df[group_on]==group) & (df[hue]==h)][value2count].sum() for h in hues] if value2count else [df[(df[group_on]==group) & (df[hue]==h)].shape[0] for h in hues]
            for (h,cn) in zip(hues,counts):
                cdict[h]=cn*100/np.sum(counts) if to_perc else cn
            df_p=df_p.append(cdict,ignore_index=True)      
        return df_p
    else:
        print('Wrong columns name...Exit')
        exit()

def duration_to_float(d):
    d=d.replace(',','.')
    d=d.replace(' seconds','')
    return float(d)

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def copy_all_files(src, dest,ext):
    #for file_path in glob.glob(os.path.join(src, '**', '*.%s'%ext), recursive=True):
    #    new_path = os.path.join(dest, os.path.basename(file_path))
    #    shutil.copy(file_path, new_path)
    for file in glob.glob(src+'/**/*%s'%ext, recursive = True): 
        new_path = os.path.join(dest, os.path.basename(file))
        print(file)
        print(new_path)
        shutil.copyfile(file, new_path)
        

def delete_biflows_by_len(ds,field,min_packets=1, verbose=False):


    count_row_before = ds.shape[0]
    #index = [i for i, v in ds[[field]].iterrows() if len(v[0]) < min_packets]
    #ds.drop(index, inplace=True)
    ds['BF_len']=ds['packet_dir'].apply(lambda x: len(x))
    ds=ds[ds['BF_len']>0]
    del ds['BF_len']
    count_row_after = ds.shape[0]

    if verbose:
        print('Deleting of all biflows with less than %s packets, deleted %s/%s Biflows (%.2f%%)' % (min_packets,
            str(count_row_before - count_row_after),
            count_row_before,
            (100 * (count_row_before - count_row_after) / count_row_before))
            )    
    return ds,(count_row_before - count_row_after)


def get_ps(headers,pls,dirs,has_dir=False):
    ps=[]
    #print(headers)
    #print(pls)
    for i,(h,p,d) in enumerate(zip(headers,pls,dirs)):
        #print(i,h,p)
        if has_dir:
            #ps.append(np.sign(p)*(h+abs(p)))
            ps.append((1 if d==1 else -1)*(h+p))
        else:
            ps.append(h+p)
    return ps

#*******************************************************************************************************************
#Per-trace windowed metrics
#*******************************************************************************************************************
def get_slots_value(slots_values, verbose=False):
    
    #sommo sulle colonne. time_slots= NBF x NTimeSlot
    #print(len(slots_values))
    y=np.array([np.array(xi) for xi in slots_values])
    bfactive=np.sum(y, axis=0)
    
    if verbose:
        print('*****get_slots_values IN: [%s]\n%s'%(np.shape(y),y))
        print('*****get_slots_values OUT: [%s]\n%s'%(np.shape(bfactive),bfactive))
        #input()
    #print('*****get_slots_values OUT <list>: %s',len(bfactive))
    return list(bfactive)


def get_slots_rate(values,edges,w,verbose=False):
    slots_rate=[]
    
    #print('#############',len(values),len(edges))
    for s,v in enumerate(values):
        if(edges[s+1]-edges[s])!=w:
            print('Warning, slot width mismatch')
        slots_rate.append(v/(edges[s+1]-edges[s]))
        
    if verbose:
        print('*****get_slots_values IN: [%s]\n%s',np.shape(values),values)
        print('*****get_slots_rate_values OUT: [%s]\n%s',np.shape(slots_rate),slots_rate)
        #input()
    return slots_rate


def get_mean_slots_value(slot_values):
    maxslots=np.max([len(s) for s in slot_values])
    tot_slot_values=np.zeros(maxslots)
    count_slot_values=np.zeros(maxslots,int)
    
    for s in slot_values:
        for i,sv in enumerate(s):
            tot_slot_values[i]+=sv
            count_slot_values[i]+=1
        
    return tot_slot_values/count_slot_values

def get_rhod(dvalues,uvalues):
    
    return [d/(d+u) if (d+u)>0 else 0 for (d,u) in zip(np.asarray(dvalues,np.float64),np.asarray(uvalues,np.float64))]



def copy_from_mirage(ts,cpath,dest):
    files_path={
        'pickle':'mirage2020dataset_biflows_all_packets.pickle',
        'csv':'mirage2020dataset_labeled_biflows_metadata.csv',
        'json':'mirage2020dataset_labeled_biflows_all_packets_encryption_metadata.json'
    }
    
    for ext,file in files_path.items():
        for infile in glob.glob(cpath+'%s_mirage2020dataset_biflows_all_packets/*%s'%(ts,file), recursive = True): 
            new_path = os.path.join(dest[ext], os.path.basename(infile))
            #print(infile)
            #print(new_path)
            #print('Copying %s to %s'%(infile,new_path))
            shutil.copyfile(infile, new_path)
            
#*******************************************************************************************************************
#LOAD/STORE Dataset
#*******************************************************************************************************************
def load_dataset(ds_path, sep=',',delimiter=None, header='infer',index_col=None,decimal='.', subset=None):
    if Path(ds_path).is_file():
        if ds_path.endswith('.parquet') or 'snappy' in os.path.basename(ds_path):
            print('Dataset\'s extension: parquet/snappy')
            if subset is not None:
                ds=pd.read_parquet(ds_path,columns=subset)
            else:
                ds=pd.read_parquet(ds_path)
        elif ds_path.endswith('.csv'):
            print('Dataset\'s extension: csv')
            ds=pd.read_csv(ds_path, sep=sep,delimiter=delimiter, header=header,index_col=index_col,decimal=decimal)
        elif ds_path.endswith('.pickle'):
            print('Dataset\'s extension: pickle')
            with open(ds_path, 'rb') as f_df:
                ds = pk.load(f_df)
        elif ds_path.endswith('.xlsx'):
                data=pd.ExcelFile(ds_path)
                ds=data.parse(data.sheet_names[0]) 
                ds.dropna(how='all')
    if Path(ds_path).is_dir():
        files=glob.glob(ds_path+'/*.parquet')
        if len(files)>0:     
            if subset is not None:
                ds=pd.concat([pd.read_parquet(tmp_pq, columns=subset) for tmp_pq in files],copy=False) 
            else:
                ds=pd.concat([pd.read_parquet(tmp_pq) for tmp_pq in files],copy=False) 
        ds=ds.reset_index()
    return ds



def save_dataset(ds,ds_path, fmt='parquet',sep=',',delimiter=None, header='infer',index=False,decimal='.'):
    if fmt=='parquet':
        print('Dataset\'s extension: parquet/snappy')
        ds.to_parquet(ds_path+'.%s'%fmt)
    elif fmt=='csv':
        print('Dataset\'s extension: csv')
        ds.to_csv(ds_path+'.%s'%fmt, sep=sep,delimiter=delimiter, header=header,index=index,decimal=decimal)
    elif fmt=='pickle':
        print('Dataset\'s extension: pickle')
        pk.dump(ds, open(ds_path+'.%s'%fmt, 'wb'), pk.HIGHEST_PROTOCOL)
        
    print('Dataset saved in: %s'%(ds_path+'.%s'%fmt))
    
    
    
    
#*******************************************************************************************************************   
#Deep Learning Framework
#*******************************************************************************************************************
def get_trainable_params(rootres,arch):
    fmodel=os.path.join(rootres,'%s_model_summary.txt'%arch)
    with open(fmodel, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            if 'Trainable params: ' in line:
                params=int(line.replace('Trainable params: ','').replace(',',''))
                break
    return params

def load_performances(rootres,arch,nfolds=10):
    fmetrics=os.path.join(rootres,'performance','%s_performance.dat'%arch)
            
    data=np.zeros(shape=(nfolds,4),dtype=float)
    fi=0
    with open(fmetrics, 'r') as read_obj:
        for line in read_obj:
            if 'Fold' in line:
                print(line)
                fline=line.split()[1:]
                fline=np.asarray([float(a.replace('%','')) for a in fline],dtype=float)
                data[fi]=fline
                fi+=1
    return data      

def load_predictions_fbyf(rootres,fold):
    path_predf=os.path.join(rootres,'predictions','fold_%s'%fold)
    file=glob.glob(path_predf+'/*fold_%s_predictions.dat'%fold)
    if len(file)!=1:
        print('Error in predictions file')
        exit()
    else:
        file=file[0]
    
    datapred=np.loadtxt(file,dtype=str)[1:]    
    return datapred

import json
def load_labels_map(rootres):
    lmap_path=os.path.join(rootres,'labels_map.json')

    with open(lmap_path,mode='r') as f:
        lmap=json.load(f)    
    return lmap


def load_soft_values_fbyf(rootres,fold):
    path_predf=os.path.join(rootres,'soft_values','fold_%s'%fold)
    file=glob.glob(path_predf+'/*fold_%s_soft_values.dat'%fold)
    if len(file)!=1:
        print('Error in predictions file')
        exit()
    else:
        file=file[0]
    
    softs=np.loadtxt(file,dtype=str)[1:]
    softs_out=np.asarray((np.shape(softs)[0],3))
    for si,sf in enumerate(softs):
        softs_out[si][0]=int(sf[0])
        softs_out[si][1]=np.argmax(np.asarray(str(sf[1:]).replace('[','').replace(']','').replace('\'','').split(','),dtype=float))
        softs_out[si][2]=np.max(np.asarray(str(sf[1:]).replace('[','').replace(']','').replace('\'','').split(','),dtype=float))

    return softs_out
    
    
    
def get_root_results_path(root_path,classifier,label_type,npackets=None,nbytes=None):
    if classifier=='wang' and nbytes:
        rootres=os.path.join(root_path,'%s_%sB_%s'%(classifier.upper(),nbytes,label_type))
    elif classifier=='lopez' and npackets:
        rootres=os.path.join(root_path,'%s_%sP_%s'%(classifier.upper(),npackets,label_type))
    elif classifier=='mimetic' and npackets and nbytes:
        rootres=os.path.join(root_path,'%s_%sB_%sP_%s'%(classifier.upper(),nbytes,npackets,label_type))
    elif classifier=='mimeticenh' and npackets and nbytes:
        rootres=os.path.join(root_path,'%s_%sB_%sP_%s'%(classifier.upper(),nbytes,npackets,label_type))
    else:
        print('get_root_results_path(): passed incomplete input')
        exit()
    if not os.path.exists(rootres):
        print('DIR <%s> not found'%rootres)
    return rootres



def reset_5fields(quintuple):
    #<IPs,Ps,IPd,Pd,Proto>
    qfields=quintuple.split(',')
    return qfields[0],qfields[2],qfields[3],qfields[4]


def check_from_beginning(row):
    from_beg=True
    if row['proto']=='6':
        
        hs_flags=row['HS_TCP_flags']
        hs_dir=row['HS_packet_dir']
        bf_flags=row['BF_TCP_flags']
        bf_dir=row['packet_dir']
        #print(hs_flags)
        if len(hs_flags)>1:
            csyn=False
            ssyn=False
            cack=False
            icsyn=None
            issyn=None
            icack=None
            
            for ics,(f,d) in enumerate(zip(hs_flags,hs_dir)):
                if f=='S' and d==0:
                    csyn=True
                    icsyn=ics
                    break                        
            if not csyn:        
                from_beg=False
            else:
                for iss,(f,d) in enumerate(zip(hs_flags,hs_dir)):
                    if f=='SA' and d==1 and iss>icsyn:
                        ssyn=True
                        issyn=iss
                        break
                if not ssyn:
                    from_beg=False
                else:
                    for ica,(f,d) in enumerate(zip(hs_flags,hs_dir)):
                        if 'A' in f and d==0 and ica>issyn:
                            cack=True
                            icack=ica
                            break
                        elif 'A' in bf_flags[0] and bf_dir[0]==0:
                            cack=True
                            icack=len(hs_flags)+1
                            break
                    if not cack:
                        from_beg=False  
        else:
            from_beg=False
    return from_beg


def delete_no_handshake(ds,activity=False,verbose=False):
    ds['ips'],ds['ipd'],ds['pd'],ds['proto']=zip(*ds['BF_quintuple'].apply(reset_5fields))
    ds['3Way']=ds.apply(lambda x: check_from_beginning(x), axis=1)                                             
    count_row_before = ds.shape[0]
    
    if verbose:
        if activity:
            no3way=['_'.join([x['BF_label'],x['BF_activity']]) for i,x in ds.iterrows() if not x['3Way']]
        else:
            no3way=[x['BF_label'] for i,x in ds.iterrows() if not x['3Way']]
    ds=ds[ds['3Way']]
    count_row_after = ds.shape[0]
    
    print('Deleting of all TCP biflows without 3 Way Handshake, deleted %s/%s Biflows (%.2f%%)' % (
            str(count_row_before - count_row_after),
            count_row_before,
            (100 * (count_row_before - count_row_after) / count_row_before))
            )
    
    if verbose:
        labels,counts=np.unique(no3way, return_counts=True)
        no3way_count=count_row_before - count_row_after
        for lab,count in zip(labels,counts):
            print('%s: %s (%.2f%%)'%(lab,count,count/no3way_count*100))
    return ds,(count_row_before - count_row_after)    


# def sanitize(ds,activities=False,less1=False,handshake=False,f_app=None,f_act=None, replace_vconf=False,return_count=False):

#         if f_app:
#             ds=ds[ds['BF_label'].isin(f_app)]
#         if f_act:
#             ds=ds[ds['BF_activity'].isin(f_act)]
            
#         if replace_vconf and activities:
#             ds.loc[ds['BF_activity']=='videoconference','BF_activity']='videocall'     
#         count=0
#         if less1:
#             ds, emptycount=delete_biflows_by_len(ds,'packet_dir',1,True)
#             count+=emptycount
#         if handshake:
#             ds,nohandshake_count=delete_no_handshake(ds,activities)
#             count+=nohandshake_count    
                 
#         if return_count:
#             return ds,count
#         else:
#             return ds
 
def sanitize(ds,activities=False,less1=False,handshake=False,f_app=None,f_act=None, how_join=None,return_count=False):

        if f_app:
            ds=ds[ds['BF_label'].isin(f_app)]
        if f_act:
            ds=ds[ds['BF_activity'].isin(f_act)]
            
        if how_join and activities:
            if how_join=='novconf':
                ds.loc[ds['BF_activity']=='videoconference','BF_activity']='videocall'
            elif how_join=='media':
                ds.loc[ds['BF_activity'].isin(['videoconference','webinar']),'BF_activity']='videocall'     
        count=0
        if less1:
            ds, emptycount=delete_biflows_by_len(ds,'packet_dir',1,True)
            count+=emptycount
        if handshake:
            ds,nohandshake_count=delete_no_handshake(ds,activities)
            count+=nohandshake_count    
                 
        if return_count:
            return ds,count
        else:
            return ds   
from pathlib import Path
#def load_and_sanitize(ds_path,dofi,noempty=True,handshake=False,replace_vconf=False,app_filter=None,activity_filter=None):
def load_and_sanitize(ds_path,dofi,noempty=True,handshake=False,how_join=None,app_filter=None,activity_filter=None, subset=None):    
    if dofi in ['a','m'] and app_filter is None:
        exit()
    if dofi in ['t','m'] and activity_filter is None:
        exit()    
        
    if Path(ds_path).is_file():
            activities=True if 'activities' in os.path.basename(ds_path) else False     
            ds=load_dataset(ds_path,subset=subset)
            md5=os.path.basename(ds_path).split('_')[-1].split('.')[0]
            #ds=sanitize(ds,activities, noempty, handshake, app_filter if dofi in ['a','m'] else None,activity_filter if dofi in ['t','m'] else None,replace_vconf)
            ds=sanitize(ds,activities, noempty, handshake, app_filter if dofi in ['a','m'] else None,activity_filter if dofi in ['t','m'] else None,how_join)
            return ds,md5,activities
                
    elif Path(ds_path).is_dir():
        files=glob.glob(ds_path+'/*.parquet')

        if len(files)>0:
            md5=os.path.basename(files[0]).split('_')[-1].split('.')[0]
            activities=True if 'activities' in os.path.basename(files[0]) else False 
            if subset is not None:
                ds=pd.concat([pd.read_parquet(tmp_pq, columns=subset) for tmp_pq in files],copy=False)     
            else:
                ds=pd.concat([pd.read_parquet(tmp_pq) for tmp_pq in files],copy=False)  
            #ds=load_dataset(files[0])
            
            # for file in files[1:]:  
            #     ds_t=load_dataset(file)
            #     ds=pd.concat([ds,ds_t])
        else:
            print('Wrong path, dataset/s not found')
            exit()     
        #ds=sanitize(ds,activities,noempty, handshake, app_filter if dofi in ['a','m'] else None,activity_filter if dofi in ['t','m'] else None,replace_vconf)
        ds=ds.reset_index()
        ds=sanitize(ds,activities,noempty, handshake, app_filter if dofi in ['a','m'] else None,activity_filter if dofi in ['t','m'] else None,how_join)
        return ds,md5,activities
    else:
        print('Wrong path')
        exit()
        
        
def get_percentile(df, column, percentile, is_temporal=False):
    base_index = 1 if is_temporal else 0
    values = [v for bf in df[column].values for v in bf[base_index:]]
    perc = np.percentile(values, percentile)
    return perc

def saturate(bf, lower_threshold, upper_threshold, is_temporal=False):
    if not len(bf):
        return bf
    bf = [v if lower_threshold <= v <= upper_threshold else lower_threshold if v < lower_threshold else upper_threshold
          for v in bf] if not is_temporal else [0.] + [
        v if lower_threshold <= v <= upper_threshold else lower_threshold if v < lower_threshold else upper_threshold
        for v in bf[1:]]
    return bf


class_type_cmap={
    'app': 'YlOrRd',
    'activity':'PuBuGn',
    'appactivity':'YlGnBu'
}      


label_name_map={
    'a':'APP',
    't':'ACT',
    'm': 'APPACT'
}

on_label_type_map={
    'APP': 'App',
    'ACT':'Activity',
    'APPACT':'App x Activity'
}


labels_map={
                'GotoMeeting': 'Gmg',
                'Skype': 'Sky',
                'Teams': 'Tms',
                'Webex': 'Wbx',
                'Zoom': 'Zom',
                'Meets':'Met',
                'Discord':'Dsd',
                'Signal':'Sgl',
                'JitsiMeet':'Jmt',
                'Whatsapp':'Wap',
                'Workplace':'Wpl',
                'Playstore':'Ps',
                'Messenger':'Msg',
                'Telegram': 'Tel',
                'Slack':'Slk',
                'chat':'Chat',
                'audiocall':'ACall',
                'videocall':'VCall',
                'videoconference':'VConf',
                'webinar':'Webi',
                'video on-demand':'VoD',
                'Unknown':'Unk'
        }
        

label_map={
    'a':'app',
    't':'activity',
    'm': 'appactivity'
}
#multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_test_times.dat

classifiers_map={
    'wang2017endtoend_1DCNN': '1D-CNN',
    'lopez2017network_CNN_RNN_2a':'Hybrid',
    'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU':'MIMETIC',
    'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_enhanced':'MIMETICE',
    'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_3M': 'MIMETICA',
    'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_3M2': 'MIMETICA2',
    'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_EA': 'MIMETICEA',
    'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_EA2': 'MIMETICEA2',
    'multimodal_lopez2017network_GRU_EA': 'MIMETICEALC',
    'multimodal_wang2017endtoend_1DCNN_EA': 'MIMETICEAWC',
    'model_contextual_features_v1': 'confet1',
    'wang2017endtoend_1DCNN_multimodal_branch': 'ewang',
    'lopez2017network_GRU_multimodal_branch': 'elopez'
}  

classifiers_archs={
    'wang':'wang2017endtoend_1DCNN',
    'lopez':'lopez2017network_CNN_RNN_2a',
    'mimetic':'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU',
    'mimetice':'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_enhanced',
    'mimetica':'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_3M',
    'mimetica2':'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_3M2',
    'mimeticea':'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_EA',
    'mimeticea2':'multimodal_wang2017endtoend_lopez2017network_1DCNN_GRU_EA2',
    'mimeticeawc':'multimodal_wang2017endtoend_1DCNN_EA',
    'mimeticealc':'multimodal_lopez2017network_GRU_EA',
    'confet1':'model_contextual_features_v1',
    'ewang':'wang2017endtoend_1DCNN_multimodal_branch',
    'elopez':'lopez2017network_GRU_multimodal_branch'
}