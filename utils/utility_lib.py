import os
import pickle
import sys
from os import path, listdir
from pathlib import Path
import lz4.frame
from sklearn.utils import resample
from operator import itemgetter
import numpy as np
import csv
from pathlib import Path


def fromlatexcsv_to_csv_comnet2021(filename):
    """This function deletes tag latex from a csv file and save a cleaned csv. ONLY FOR COMNET 2021
    The input file must have this shape:
    APP,HMM,MC,ML,DL,DIFF
    DiscoveryVR,$17.69(\pm13.46)$,$79.02(\pm2.81)$,$63.59(\pm2.98)$,$86.58(\pm2.82)$,$+389.49\%$"""
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rows_list = []
        bad_chars_list = ['$', '(', ')']
        for i, row in enumerate(csv_reader):
            # From current row the last value concerning the percentage change is ignored
            if i:
                rows_list.append([''.join(ch for ch in elem if ch not in bad_chars_list).split('\\')[0]
                                  for elem in row])
            else:
                row[-1] = row[-1] + '%'
                rows_list.append(row)

        fname_head, fname_tail = path.split(filename)
        cleaned_filename = Path(fname_tail).stem + '_cleaned.csv'
        cleaned_filename = path.join(fname_head, cleaned_filename)
        with open(cleaned_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows_list)

        return cleaned_filename


def retrieve_info_from_filename(df_filename):
    filename = df_filename.split('/')[-1]
    filename_token = filename.split('_')
    model = filename_token[0]
    app = filename_token[1]
    forecasting_strategy = filename_token[5]
    granularity = filename_token[8]
    return model, app, forecasting_strategy, granularity


def read_multiple_df_per_pred_strategy(path_dataset, return_features=False):
    """This function organizes multiple dataframes into a dictionary where each key indicates a forecasting strategy"""
    if path.isdir(path_dataset):
        results_df_name_list = listdir(path_dataset)
        results_per_strategy_dict = {}
        for i, df_filename in enumerate(results_df_name_list):
            model, app, forecasting_strategy, granularity = retrieve_info_from_filename(df_filename)
            df_path = path.join(path_dataset, df_filename)
            results_per_strategy_dict[forecasting_strategy] = load_all_pickles(df_path, granularity)
            if i == 0:
                features_list = list(results_per_strategy_dict[forecasting_strategy].keys())
    else:
        sys.exit('First parameter must be a root directory containing dataframes')

    if not return_features:
        return results_per_strategy_dict
    else:
        return results_per_strategy_dict, features_list


def resampling_pred_matrices(x, y, indices, n_samples, random_state=0):
    x_y_indexed = [(idx, matrix, g_truth) for (idx, matrix, g_truth) in zip(indices, x, y)]
    x_y_indexed_resampled = sorted(resample(x_y_indexed, n_samples=n_samples, replace=False, random_state=random_state),
                                   key=itemgetter(0))
    ids_resampled = np.array([x_y_indexed_resampled[i][0] for i in range(len(x_y_indexed_resampled))])
    x_resampled = np.array([x_y_indexed_resampled[i][1] for i in range(len(x_y_indexed_resampled))])
    y_resampled = np.array([x_y_indexed_resampled[i][2] for i in range(len(x_y_indexed_resampled))])
    return x_resampled, y_resampled, ids_resampled


def assert_lookahead(lookahead):
    assert (isinstance(lookahead, tuple) or isinstance(lookahead, list)) and len(lookahead) == 2, \
        "Lookahead parameter must be a tuple or list of length equal to 2. Example: (1,15)"
    lookahead_start = lookahead[0]
    lookahead_stop = lookahead[1]
    assert isinstance(lookahead_start, int) and isinstance(lookahead_stop, int), \
        "The lower boundary and the upper boundary of the lookahead must be integers"
    assert 0 < lookahead_start <= lookahead_stop, \
        "The lower boundary of the lookahead must be less or equal than the upper boundary and both must be positive"


def align_true_predict(y_true, y_pred, forward_step):
    for j in range(len(y_true)):
        y_true[j] = y_true[j][forward_step:]
        y_pred[j] = y_pred[j][:-forward_step]
    return y_true, y_pred


def load_all_pickles(filename, granularity):
    features_name = ['PL', 'IAT', 'PD'] if granularity == 'packets' else ['UF_volume_%s' % granularity,
                                                                          'DF_volume_%s' % granularity,
                                                                          'UF_N_packets_%s' % granularity,
                                                                          'DF_N_packets_%s' % granularity]
    df_per_feature_dict = dict()
    i = 0
    with lz4.frame.open(filename, "rb") as f:
        while True:
            try:
                df_per_feature_dict[features_name[i]] = pickle.load(f)
                df_per_feature_dict[features_name[i]].reset_index(drop=True, inplace=True)
                i += 1
            except EOFError:
                break
    return df_per_feature_dict


def concat_temp_results(model, app, forecasting_strategy, granularity, n_features, root_temp_res=os.getcwd(),
                        root_final_res=os.getcwd(), order=30):
    path_to_pickles = path.join(root_temp_res, app, granularity, forecasting_strategy, model)
    pickle_in_list = sorted(os.listdir(path=path_to_pickles))
    total_fold = len(pickle_in_list)
    pickle_path = path.join(path_to_pickles, pickle_in_list[0])
    df_per_fold_dict = load_all_pickles(pickle_path, granularity)
    for i in range(1, len(pickle_in_list)):
        pickle_path = path.join(path_to_pickles, pickle_in_list[i])
        df_temp_dict = load_all_pickles(pickle_path, granularity)
        for feature in df_per_fold_dict.keys():
            df_per_fold_dict[feature] = df_per_fold_dict[feature].append(df_temp_dict[feature])
            df_per_fold_dict[feature].reset_index(drop=True, inplace=True)

    res_dir = path.join(root_final_res, app, granularity, model)
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    file_output = path.join(res_dir, model + '_' + app + '_results_' + str(total_fold) + '_Fold_' +
                            forecasting_strategy + '_W_%s_' % order + granularity + '_' + str(n_features) +
                            '_features.pickle')
    pickle_out = lz4.frame.open(file_output, "wb")
    for feature in df_per_fold_dict.keys():
        pickle.dump(df_per_fold_dict[feature], pickle_out, pickle.HIGHEST_PROTOCOL)
    pickle_out.close()


# From https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/gridspec_multicolumn.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-multicolumn-py
def format_axes(fig):
    """This function shows how axes are formatted on the current figure"""
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


if __name__ == "__main__":
    app = sys.argv[1]
    forecasting_strategy = sys.argv[2]
    model = sys.argv[3]
    granularity = sys.argv[4] if len(sys.argv) >= 5 else 'packets'
    n_features = int(sys.argv[5]) if len(sys.argv) == 6 else 3
    concat_temp_results(model, app, forecasting_strategy, granularity, n_features)
