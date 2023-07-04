import os
import pickle
import sys
from os import path, listdir
from pathlib import Path
import numpy as np
import math
import pandas as pd


def load_all_pickles(filename, features_name):

    df_per_feature_dict = dict()
    i = 0
    #with lz4.frame.open(filename, "rb") as f:
    with open(filename, "rb") as f:
        while True:
            try:
                df_per_feature_dict[features_name[i]] = pickle.load(f)
                i += 1
            except EOFError:
                break
    return df_per_feature_dict

def concat_temp_results(path_to_pickles, path_to_res_file, file_out_template, root_final_res, features_list):
    # path_to_pickles = path.join(root_temp_res, app, granularity, aggregation_strategy, models)
    # path_to_pickles = path_to_dir
    n_features = 4
    pickle_in_list = sorted(os.listdir(path=path_to_pickles))
    total_fold = len(pickle_in_list)
    pickle_path = path.join(path_to_pickles, pickle_in_list[0])
    df_per_fold_dict = load_all_pickles(pickle_path, features_list)
    for i in range(1, len(pickle_in_list)):
        pickle_path = path.join(path_to_pickles, pickle_in_list[i])
        df_temp_dict = load_all_pickles(pickle_path, features_list)
        for feature in df_per_fold_dict.keys():
            df_per_fold_dict[feature] = df_per_fold_dict[feature].append(df_temp_dict[feature])

    res_dir = path.join(root_final_res, path_to_res_file)
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    file_output = path.join(res_dir, file_out_template + '.pickle')
    # pickle_out = lz4.frame.open(file_output, "wb")
    pickle_out = open(file_output, "wb")
    for feature in df_per_fold_dict.keys():
        pickle.dump(df_per_fold_dict[feature], pickle_out, pickle.HIGHEST_PROTOCOL)
    pickle_out.close()


def create_dataset_jumping_prediction_series(dataset, look_back=1, max_look_back=60, look_forward=1, pad=0, granularity=100,
                                                   features_last=True, multi_output=True, right_pad=False,
                                                   prediction_time=100, testing=False, rtt=False):
    """
    :param dataset: array-like of size (nbiflows, nfeatures, npackets)
    :param look_back: integer, dataX will have the shape (nsequences, look_back, nfeatures)
    :param look_forward: integer, dataY will have the shape (nsequences, look_forward, nfeatures)
    :param pad: value (or list of values of length = nfeatures) used to pad the first packets under the look_back window size.
    :param features_last: if True, dataX will be (nsequences, look_back, nfeatures).
        If False, dataX will be (nsequences, nfeatures, look_back).
    :param multi_output: if True (default), will return the entire look_forward groud truth, else will return the single
        output groud truth.
    :return: dataX and dataY
    """

    nfeatures = len(dataset[0])
    prediction_win_size = int(prediction_time/granularity) if granularity < prediction_time else 1
    fraction = prediction_win_size if testing else 1
    #memory_size = slide_frac * (look_back - 1) + 1
    try:
        len(pad)  # Check if pad has a len.
    except:
        pad = [pad] * nfeatures  # If not, we built a list of size nfeatures with the same pad values
    assert len(pad) == nfeatures, 'The length of the pad should be the same as the features number.'
    dataX, dataY, dataBF = [], [], []
    right_pad_list = list([[pad[j]] * look_forward for j in range(nfeatures)]) if right_pad else [[]] * nfeatures
    for bf_index, bf in enumerate(dataset):
        vX, vY = [], []
        n_pkt = len(bf[0])
        for i in range(1, min([max_look_back-fraction+1, n_pkt - (prediction_win_size-1)]), fraction):
            initial_history = list([list(feat[0:i+fraction-1]) for feat in bf])
            initial_history = list(
                [[pad[j]] * (max_look_back - len(feat)) + feat + right_pad_list[j]
                 for j, feat in enumerate(initial_history)])
            initial_history = np.array(initial_history).T
            initial_history = initial_history[(max_look_back-look_back):max_look_back]
            initial_oracle = [sum(feat[i+fraction-1:i+prediction_win_size+fraction-1]) for feat in bf]

            """if not multi_output:
                            initial_oracle = [o[-1:] for o in initial_oracle]"""
            initial_oracle = np.array(initial_oracle).T
            initial_oracle = initial_oracle.reshape(1, len(initial_oracle))
            vX.append(initial_history)
            vY.append(initial_oracle)
            dataBF.append(bf_index)
            #end_value = math.ceil((n_pkt - memory_size - slide_frac)/slide_frac) + 1
        for i in range(0, n_pkt - max_look_back - (prediction_win_size-1), fraction):
            history = list([list(feat[i:i + max_look_back]) + right_pad_list[j] for j, feat in enumerate(bf)])
            history = np.array(history).T
            history = history[(max_look_back-look_back):max_look_back]
            """ if granularity != prediction_time:
                oracle = [sum(feat[(i + max_look_back + look_back%2):i + max_look_back + look_back%2 + prediction_win_size]) for feat in bf]
            else:"""
            oracle = [sum(feat[(i + max_look_back):i + max_look_back + prediction_win_size]) for
                          feat in bf]
            """if not multi_output:
                oracle = [o[-1:] for o in oracle]"""
            oracle = np.array(oracle).T
            oracle = oracle.reshape(1, len(oracle))
            vX.append(history)
            vY.append(oracle)
            dataBF.append(bf_index)
        dataX.extend(vX)
        dataY.extend(vY)
    if not features_last:
        dataX = np.asarray([x.T for x in dataX])
    return np.asarray(dataX), np.asarray(dataY), np.asarray(dataBF)


def create_dataset_jumping_prediction_series_rtt(dataset, rtt, row_index, max_look_back, memory_time, look_forward=1, pad=0,
                                                 features_last=True, right_pad=False, prediction_time=100,
                                                 testing=False):
    # Convert RTT to ms
    rtt = rtt/1000
    # Max W is considered to compute bf padding
    # max_look_back = math.floor(memory_time / min(rtt))

    nfeatures = len(dataset[0])
    try:
        len(pad)  # Check if pad has a len.
    except:
        pad = [pad] * nfeatures  # If not, we built a list of size nfeatures with the same pad values
    assert len(pad) == nfeatures, 'The length of the pad should be the same as the features number.'
    dataX, dataY, dataBF = [], [], []
    right_pad_list = list([[pad[j]] * look_forward for j in range(nfeatures)]) if right_pad else [[]] * nfeatures
    for bf_index, bf in enumerate(dataset):
        # RTT values typically vary between 4 and 40 ms
        granularity = rtt[row_index[bf_index]]    # different for each bf
        if granularity < 1:
            continue
        prediction_win_size = math.floor(prediction_time/granularity)
        # Discard bf if granularity > prediction_time
        if not prediction_win_size:
            continue
        fraction = prediction_win_size if testing else 1
        look_back = math.floor(memory_time/granularity)
        vX, vY = [], []
        n_pkt = len(bf[0])
        for i in range(1, min([look_back-fraction+1, n_pkt - (prediction_win_size-1)]), fraction):
            initial_history = list([list(feat[0:i+fraction-1]) for feat in bf])
            initial_history = list(
                [[pad[j]] * (max_look_back - len(feat)) + feat + right_pad_list[j]
                 for j, feat in enumerate(initial_history)])
            initial_history = np.array(initial_history).T
            # debug
            #debug_oracle = [feat[i:i+prediction_win_size] for feat in bf]

            initial_oracle = [sum(feat[i+fraction-1:i+prediction_win_size+fraction-1]) for feat in bf]
            #initial_oracle = [feat[i:i+prediction_win_size] for feat in bf]

            """if not multi_output:
                            initial_oracle = [o[-1:] for o in initial_oracle]"""
            initial_oracle = np.array(initial_oracle).T
            initial_oracle = initial_oracle.reshape(1, len(initial_oracle))
            vX.append(initial_history)
            vY.append(initial_oracle)
            dataBF.append(bf_index)
            #end_value = math.ceil((n_pkt - memory_size - slide_frac)/slide_frac) + 1
        for i in range(0, n_pkt - look_back - (prediction_win_size-1), fraction):
            history = list([list(feat[i:i + look_back]) + right_pad_list[j] for j, feat in enumerate(bf)])
            # Padding added to fulfill memory buffer with @max_look_back samples
            history = list(
                [[pad[j]] * (max_look_back - len(feat)) + feat + right_pad_list[j]
                 for j, feat in enumerate(history)])
            history = np.array(history).T
            #oracle_pkt = (i + look_back + prediction_win_size)
            if granularity != prediction_time:
                oracle = [sum(feat[(i + look_back + look_back%2):i + look_back + look_back%2 + prediction_win_size]) for feat in bf]
            else:
                oracle = [sum(feat[(i + look_back):i + look_back + prediction_win_size]) for
                          feat in bf]
            """if not multi_output:
                oracle = [o[-1:] for o in oracle]"""
            oracle = np.array(oracle).T
            oracle = oracle.reshape(1, len(oracle))
            vX.append(history)
            vY.append(oracle)
            dataBF.append(bf_index)
        dataX.extend(vX)
        dataY.extend(vY)
    if not features_last:
        dataX = np.asarray([x.T for x in dataX])
    return np.asarray(dataX), np.asarray(dataY), np.asarray(dataBF)


def get_memory_parameters(par, gran, fixed_memory_time=False):
    if fixed_memory_time:
        out_par = int(par/gran)
    else:
        out_par = par*gran
    return out_par

def get_interval_string(interval, agg_strategy='temporal', time_unit=True):
    if agg_strategy == 'temporal' or 'rtt':
        if time_unit:
            out_string = str(interval) + 'ms' if int(interval) < 1000 else str(int(int(interval)/1000)) + 's'
        else:
            out_string = str(interval) + 'ms'

    elif agg_strategy == 'spatial':
        out_string = str(interval) + '_pkt'
    else:
        print('ERROR: aggregation not supported')
        sys.exit(-1)
    return out_string

def set_input_values(config_name, dataset_path, app, lookahead, model, interval, agg_strategy, for_strategy, fold,
                     current_fold, fit_model, pred_win_size, pred_win_time, prediction_time, is_fixed_tm, win_type,
                     app_categories, shortened=False, delta=None):
    from configparser import ConfigParser

    cfg = ConfigParser()
    cfg.read(config_name)

    cfg.set('I/O', 'dataset_path', dataset_path)
    cfg.set('BASE', 'app', app)
    cfg.set('BASE', 'lookahead', str(lookahead))
    cfg.set('MODEL', 'models', model)
    cfg.set('OPTIONS', 'granularity', interval)
    cfg.set('OPTIONS', 'aggregation_strategy', agg_strategy)
    cfg.set('OPTIONS', 'forecasting_strategy', for_strategy)
    cfg.set('BASE', 'fold', str(fold))
    cfg.set('BASE', 'current_fold', str(current_fold))
    cfg.set('OPTIONS', 'fit_model', str(fit_model))
    cfg.set('BASE', 'prediction_win_size', str(pred_win_size))
    cfg.set('BASE', 'prediction_win_time', str(pred_win_time))
    cfg.set('BASE', 'prediction_time', str(prediction_time))
    cfg.set('OPTIONS', 'fixed_memory_time', str(is_fixed_tm))
    cfg.set('OPTIONS', 'win_type', win_type)
    cfg.set('OPTIONS', 'shortened', str(shortened))
    cfg.set('OPTIONS', 'app_categories', str(app_categories))
    cfg.set('OPTIONS', 'delta_rtt', str(delta))

    with open(config_name, 'w') as cfgfile:
        cfg.write(cfgfile)

    return cfg

def check_experiment_repeatition(app, model, agg_strategy, win_type, gran, par, prediction_time, is_fixed_tm, dir_path,
                                 shortened=False):
    import shutil

    if is_fixed_tm:
        memory_time = par
        memory_size = get_memory_parameters(int(par), int(gran), is_fixed_tm)
    else:
        memory_size = par
        memory_time = get_memory_parameters(int(par), int(gran))

    gran = str(gran) + 'ms' if int(gran) < 1000 else str(int(int(gran) / 1000)) + 's'
    prediction_time = str(prediction_time) + 'ms' if int(prediction_time) < 1000 else \
        str(int(int(prediction_time) / 1000)) + 's'
    memory_time = str(memory_time)
    dir_to_find = f'W_{memory_size}' if is_fixed_tm else f'Tm_{memory_time}'
    dir_to_place = f'Tm_{memory_time}' if is_fixed_tm else f'W_{memory_size}'
    dir_to_find_string = dir_to_find + 'ms' if not is_fixed_tm else dir_to_find
    dir_to_place_string = dir_to_place + 'ms' if is_fixed_tm else dir_to_place

    if not shortened:
        filename = f'{app}_{model}_{agg_strategy}_agg_{win_type}_win_{gran}_{dir_to_find_string}_' \
                   f'Tp_{prediction_time}.pickle'
    else:
        filename = f'{app}_{model}_{agg_strategy}_agg_{win_type}_win_{gran}_{dir_to_find_string}_' \
                   f'Tp_{prediction_time}_shortened.pickle'
    if not shortened:
        new_filename = f'{app}_{model}_{agg_strategy}_agg_{win_type}_win_{gran}_{dir_to_place_string}_Tp_' \
                                  f'{prediction_time}.pickle'
    else:
        new_filename = f'{app}_{model}_{agg_strategy}_agg_{win_type}_win_{gran}_{dir_to_place_string}_Tp_' \
                       f'{prediction_time}_shortened.pickle'

    file_found = False
    data_file_list = listdir(dir_path)

    i = 0
    while not file_found:
        if i == len(data_file_list):
            #print('File not found')
            #sys.exit(-1)
            break
        #print('File: %s' % data_file_list[i])
        if data_file_list[i].find(dir_to_find) != -1:
            file_path = path.join(dir_path, data_file_list[i])
            file_found = True
            file_res = path.join(file_path, filename)
            new_dir = path.join(dir_path, dir_to_place)
            Path(new_dir).mkdir(parents=True, exist_ok=True)
            print('Experiment found\nCopying %s to %s' % (filename, new_filename))
            path_to_copy = path.join(new_dir, new_filename)
            shutil.copy(file_res, path_to_copy)
        else:
            i += 1

    return file_found


def cut_biflows_by_timestamp(dataset, features=['timestamp', 'L4_payload_bytes_dir', 'L4_payload_bytes'], thresh=100):

    """
    :param dataset:
    :param thresh: time threshold [ms] which cuts biflows by time
    :return: new updated dataframe
    """

    ts_pl_df = dataset[features]
    ts_pl_df = ts_pl_df[ts_pl_df['timestamp'].map(lambda x: isinstance(x, np.ndarray))]
    ts_pl_df["ts_packet"] = ts_pl_df.timestamp.map(lambda x: x-x[0])
    ts_pl_df['ts_packet_cut'] = ts_pl_df.ts_packet.map(lambda x: x[x <= thresh])
    ts_pl_df['length'] = ts_pl_df.ts_packet_cut.map(lambda x: len(x))

    #df = pd.DataFrame()
    for feat in features:
        dataset[feat] = ts_pl_df.apply(lambda x: x.loc[feat][:x.length], axis=1)

    return dataset

def get_dataset(app, dataset_files_list, root_dir_datasets, agg_strategy='temporal', shortened=True):

    if agg_strategy == 'temporal':
        agg_string = 'time' if not shortened else 'time_aggregate_shortened'
    elif agg_strategy == 'rtt':
        agg_string = agg_strategy if not shortened else 'rtt_aggregate_shortened'
    i=0
    while True:
        if i == len(dataset_files_list):
            sys.exit('Dataset not found!!!')
        if dataset_files_list[i].find(app.lower()) != -1 and dataset_files_list[i].find(agg_string) != -1:
            dataset_path = path.join(root_dir_datasets, dataset_files_list[i])
            file_found = True
            return dataset_path
        else:
            i+=1

def create_test_dataset(in_data_path, out_data_path='/home/lorenzo/test_data', percentage=0.1):
    """ data_path = 'C:\\Users\\loren\\Tesi\\local-aggregation\\data\\agg_data\\' \
                'dataset_instagram_video_df_exact_noNullver_no0load_messages_saturated_fa650576_rtt_aggregate_shortened.parquet'"""
    df = pd.read_parquet(in_data_path)
    df = df.sample(frac=percentage)
    test_data_name = 'test_' + str(percentage) + '_' + in_data_path.split('/')[-1]
    output_test_data_path = path.join(out_data_path, test_data_name)
    df.to_parquet(output_test_data_path)
    print('Saving test dataset to %s' % output_test_data_path)
    return output_test_data_path


def get_empty_dataframe():
    df = pd.DataFrame(columns=[
        'Network',
        'Mode',
        'App',
        'Feature',
        'Granularity',
        'Memory_size',
        'Memory_time',
        'Prediction_time',
        'Aggregation_strategy',
        'Window_type',
        'Forecasting_strategy',
        'Lookahead',
        'N_biflows',
        'RMSE',
        'NRMSE',
        'R2',
        'RMSE_abs',
        'RMSE_std',
        'NRMSE_std',
        'R2_abs',
        'G-MEAN',
        'G-MEAN_std'])
    return df


def metric_analysis(res_kf, win_size, win_time, pred_time, metrics_df, granularity, fold=10, feature='PL', mode='biflow', net='LSTM', app='None',
                    agg_strategy='temporal', lookahead=1, forecasting_strategy='None', win_type='jumping'):
    import numpy as np
    from imblearn.metrics import geometric_mean_score
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.metrics import recall_score

    from lib import metrics_lib as metlib

    listRMSE = []
    listRMSE_abs = []
    listR2 = []
    listR2_abs = []
    listGM = []
    listGM_std = []
    listRMSE_std = []
    listNRMSE = []
    listNRMSE_std = []

    # calcolo dei parametri prestazionali
    try:
        res_fold_group = res_kf.groupby(['FOLD'])
        for h in range(lookahead):
            listRMSE_per_fold = []
            listR2_per_fold = []
            listRMSE_abs_per_fold = []
            listGM_per_fold = []
            listR2_abs_per_fold = []
            listNRMSE_per_fold = []

            for j in range(fold):
                res = res_fold_group.get_group(str(j))
                vals_true = [res.iloc[i]['y_true'][0:] for i in range(len(res))]

                if lookahead > 1:
                    vals_pred = [res.iloc[i]['y_pred_lookahead_%s' % str(h + 1)]
                                 [lookahead - h:len(res.iloc[i]['y_pred_lookahead_%s' % str(h + 1)]) - h] for i in
                                 range(len(res))
                                 if len(res.iloc[i]['y_true']) > lookahead]

                else:
                    vals_pred = [res.iloc[i]['y_pred']
                                 [0:len(res.iloc[i]['y_pred']) - h] for i in
                                 range(len(res))]

                if mode == 'biflow':

                    if feature == 'PD':
                        listGM_per_fold.append(np.nanmean([metlib.compute_g_mean(YT, YP) for (YT, YP) in
                                                           zip(vals_true, vals_pred)
                                                           if len(YT) > 1]))
                    else:
                        listRMSE_per_fold.append(np.nanmean([metlib.root_mean_squared_error(YT, YP) for (YT, YP) in
                                                             zip(vals_true, vals_pred)]))
                        listR2_per_fold.append(np.nanmean([r2_score(YT, YP) for (YT, YP) in zip(vals_true, vals_pred)
                                                           if len(YT) > 1]))
                        listRMSE_abs_per_fold.append(
                            np.nanmean([metlib.root_mean_squared_error(abs(YT), abs(YP)) for (YT, YP) in
                                        zip(vals_true, vals_pred)]))
                        listR2_abs_per_fold.append(np.nanmean([r2_score(abs(YT), abs(YP)) for (YT, YP) in
                                                               zip(vals_true, vals_pred) if len(YT) > 1]))

                elif mode == 'packets' or mode == 'aggregate_%s' % granularity:
                    YT = np.asarray([item for subarray in vals_true for item in subarray])
                    YP = np.asarray([item for subarray in vals_pred for item in subarray])

                    if feature == 'PD':
                        listGM_per_fold.append(metlib.compute_g_mean(YT, YP))
                    else:
                        listRMSE_per_fold.append(metlib.root_mean_squared_error(YT, YP))
                        listR2_per_fold.append(r2_score(YT, YP))
                        listRMSE_abs_per_fold.append(metlib.root_mean_squared_error(abs(YT), abs(YP)))
                        listR2_abs_per_fold.append(r2_score(abs(YT), abs(YP)))
                        listNRMSE_per_fold.append(metlib.normalized_root_mean_squared_error(YT, YP))

            listRMSE.append(np.nanmean(listRMSE_per_fold)) if len(listRMSE_per_fold) else []
            listRMSE_abs.append(np.nanmean(listRMSE_abs_per_fold)) if len(listRMSE_abs_per_fold) else []
            listR2.append(np.nanmean(listR2_per_fold)) if len(listR2_per_fold) else []
            listR2_abs.append(np.nanmean(listR2_abs_per_fold)) if len(listR2_abs_per_fold) else []
            listRMSE_std.append(np.nanstd(listRMSE_abs_per_fold)) if len(listRMSE_abs_per_fold) else []
            listGM.append(np.nanmean(listGM_per_fold)) if len(listGM_per_fold) else []
            listGM_std.append(np.nanstd(listGM_per_fold)) if len(listGM_per_fold) else []
            listNRMSE.append(np.nanmean(listNRMSE_per_fold)) if len(listNRMSE_per_fold) else []
            listNRMSE_std.append(np.nanstd(listNRMSE_per_fold)) if len(listNRMSE_per_fold) else []
            n_biflows = len(res_kf)

        metrics_df = metrics_df.append({
            'Network': net,
            'Mode': mode,
            'App': app,
            'Feature': feature,
            'Granularity': granularity,
            'Memory_size': win_size,
            'Memory_time': win_time,
            'Prediction_time': pred_time,
            'Aggregation_strategy': agg_strategy,
            'Window_type': win_type,
            'Forecasting_strategy': forecasting_strategy,
            'Lookahead': lookahead,
            'N_biflows': n_biflows,
            'RMSE': listRMSE,
            'NRMSE': listNRMSE,
            'R2': listR2,
            'RMSE_abs': listRMSE_abs,
            'RMSE_std': listRMSE_std,
            'NRMSE_std': listNRMSE_std,
            'R2_abs': listR2_abs,
            'G-MEAN': listGM,
            'G-MEAN_std': listGM_std,
        }, ignore_index=True)

        print(metrics_df)

    except (ValueError, RuntimeWarning):
        print('Errore nel calcolo RMSE')
    return metrics_df