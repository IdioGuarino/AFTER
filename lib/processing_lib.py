# Updateme

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import time
import psutil
import os


def dataset_split(n_samples, y=None, k=10, random_state=0, stratified=False):
    """
    :param n_samples: number of samples compose the dataset
    :param y: labels, for stratified splitting
    :param k: number of splits
    :param random_state: random state for the splitting function
    :return:
    """
    if y is not None and stratified:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
    indexes = [(train_index, test_index) for train_index, test_index in splitter.split(np.zeros((n_samples, 1)), y)]
    return indexes


def timeit(fn, args=None, time_types=None, unit='s'):
    """
    Function for timing functions. Enables the timing w/ multiple types of times and diverse units.
    :param fn: function to be evaluated.
    :param args: arguments of function under evaluation
    :param time_types: type of time one would assess, could be 'time_ns', 'process_time_ns', 'thread_time_ns', or
     'monotonic_ns'.
    :param unit: unit of time, could be ns, us, ms, or s.
    :return: fn return and time dict (one key for each time_types)
    """
    if args is not None and not isinstance(args, tuple) and not isinstance(args, dict):
        raise ValueError('Argument args should be a tuple (or dict) of arguments for fn.')
    supported_time_types = ['time', 'process_time', 'thread_time', 'monotonic']
    unit_factors = {'ns': 1e9, 'us': 1e6, 'ms': 1e3, 's': 1}
    if unit not in unit_factors.keys():
        raise ValueError('Supported units are %s.' % ' '.join(unit_factors.keys()))
    time_dict = dict()
    default = False
    if time_types is None:
        time_types = ['process_time']  # Default
        default = True
    elif isinstance(time_types, str):
        if time_types.lower() == 'all':
            time_types = supported_time_types
        else:
            time_types = [time_types]
    if len([t for t in time_types if t in supported_time_types]) == 0:
        raise ValueError('Supported Time Types are %s.' % ' '.join(supported_time_types), )
    for time_type in time_types:
        time_dict[time_type] = getattr(time, time_type)()
    ret = fn(**args) if isinstance(args, dict) else fn(*args) if args is not None else fn()
    for time_type in time_types:
        time_dict[time_type] = (getattr(time, time_type)() - time_dict[time_type]) * unit_factors[unit]
    if ret is not None:
        return ret, time_dict['process_time'] if default else time_dict
    else:
        return time_dict['process_time'] if default else time_dict


def get_used_ram(process=None, unit='k'):
    unit_factors = {'G': 1e-9, 'M': 1e-6, 'k': 1e-3, 'B': 1}
    process = psutil.Process(os.getpid()) if process is None else process
    memory = process.memory_info().rss * unit_factors[unit]
    return memory


def monitorit(fn, args=None, time_types=None, time_unit='s', memory_unit='k'):
    pre_training_memory = get_used_ram(unit=memory_unit)
    timeit_ret = timeit(fn, args, time_types, time_unit)
    model_memory = get_used_ram(unit=memory_unit) - pre_training_memory
    if isinstance(timeit_ret, tuple):
        return timeit_ret + (model_memory,)
    return timeit_ret, model_memory


def workload():
    print('CPU stressing.')
    for i in range(int(1e7)):
        _ = i ** 2
    print('Enter sleep.')
    time.sleep(3)


def timeit_test(iter=1):
    time_dict_list = []
    for i in range(iter):
        ret, time_dict = timeit(workload, time_types='all', unit='s')
        time_dict_list.append(time_dict)
    time_dict = {}
    for k in time_dict_list[0]:
        time_dict[k] = (np.mean([time_dict_elem[k] for time_dict_elem in time_dict_list]),
                        np.std([time_dict_elem[k] for time_dict_elem in time_dict_list]))
    print(time_dict)