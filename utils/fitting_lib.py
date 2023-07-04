import keras
from os import path
from lib import custom_callbacks as cbacks
from lib import generic_lookahead_predictor as glp
from lib import model_library
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
import psutil
import os
import numpy as np
import random as python_random
import utility_lib as ulib


def get_used_ram():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def fit_model(x_train, y_train, model_name, model_map, prediction_window, n_features, epochs, config, cp_dir,
              output_log_keras, idx=0, features_name=None, granularity='packets'):
    print('Used RAM before training:', get_used_ram())
    # MODEL
    experiment_model = model_library.Models(config[model_name], prediction_window, n_features, features_name)

    # CALLBACKS -> EARLY STOPPING
    es = EarlyStopping(monitor='loss',
                       min_delta=float(config['MODEL']['min_delta']),
                       mode='min',
                       patience=int(config['MODEL']['patience']),
                       verbose=1,
                       restore_best_weights=True)

    # CALLBACKS -> CSV LOGGER
    csv_logger = CSVLogger(output_log_keras + '_model_step_' + str(idx + 1) + '_keras_log.csv', append=True,
                           separator=';')

    # CALLBACK -> SAVE CHECK POINT
    path_to_save_cp = path.join(cp_dir, 'model_step_%s' % (str(idx + 1)))
    save_cp = ModelCheckpoint(path_to_save_cp, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False)

    # CALLBACK -> TIME HISTORY
    time_callback = cbacks.TimeEpochs()

    # MODEL________________
    model = getattr(experiment_model, model_map[config['MODEL']['model']])()

    config_loss_list = [config['COMPILE']['PL_loss'], config['COMPILE']['IAT_loss'], config['COMPILE']['PD_loss']]
    config_loss_weights_list = [config['COMPILE'].getfloat('pl_weight'), config['COMPILE'].getfloat('iat_weight'),
                                config['COMPILE'].getfloat('pd_weight')]

    if granularity != 'packets':
        config_loss_list = [config['COMPILE']['n_bytes_UF_loss'], config['COMPILE']['n_bytes_DF_loss']]
        config_loss_weights_list = [config['COMPILE'].getfloat('n_bytes_UF_weight'),
                                    config['COMPILE'].getfloat('n_bytes_DF_weight')]

        if n_features == 4:
            config_loss_list += [config['COMPILE']['n_agg_UF_loss'], config['COMPILE']['n_agg_DF_loss']]
            config_loss_weights_list += [config['COMPILE'].getfloat('n_agg_UF_weight'),
                                         config['COMPILE'].getfloat('n_agg_DF_weight')]

    loss = dict((k, v) for (k, v) in zip(model.output_names, config_loss_list))
    loss_weights = dict((k, v) for (k, v) in zip(model.output_names, config_loss_weights_list))
    metrics_list = n_features * ['accuracy']
    metrics = dict((k, v) for (k, v) in zip(model.output_names, metrics_list))
    model.compile(loss=loss, loss_weights=loss_weights, optimizer='adam', metrics=metrics)

    """if fold == 0 and config['OPTIONS'].getboolean('print_graph'):
                model_library.model_plot_arch(model,
                                              model_name='Model',
                                              summary=
                                              config['OPTIONS'][
                                                  'print_summary'])"""

    print(model.summary())

    if model_name == 'STN':
        x_train = np.reshape((-1, prediction_window, 1, 1, n_features))

    np.random.seed(0)
    python_random.seed(0)
    tf.random.set_seed(0)
    y_train = [y_train[:, i] for i in range(y_train.shape[1])]
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=1,
              callbacks=[es, time_callback, csv_logger, save_cp])

    print('Used RAM after training:', get_used_ram())
    del model
    keras.backend.clear_session()
    print('Used RAM after clear session:', get_used_ram())


def fit_multiple_models(x_train, y_train, bf_index_train, lookahead, model_name, model_map, prediction_window,
                        n_features, epochs, config, cp_dir, output_log_keras, forecasting_strategy, features_name=None,
                        granularity='packets', n_samples=0):
    num_trained_model = sum((os.path.isdir(os.path.join(cp_dir, mod)) for mod in os.listdir(cp_dir)))
    lookahead_start = lookahead[0] - 1 if forecasting_strategy not in ['hybrid', 'ouroboros'] else 0
    lookahead_stop = lookahead[1]
    for i in range(lookahead_start, lookahead_stop):
        if i:
            x_indices = []
            y_indices = []
            offset = 0
            for j in np.unique(bf_index_train):
                bf_len = (bf_index_train == j).sum().astype(np.int32)
                x_indices.extend([i for i in range(offset, offset + bf_len - 1)])
                y_indices.extend([i for i in range(offset + 1, offset + bf_len)])
                offset += bf_len
            x_train = np.take(x_train, x_indices, axis=0)
            y_train = np.take(y_train, y_indices, axis=0)
            bf_index_train = np.take(bf_index_train, x_indices, axis=0)

        if n_samples and x_train.shape[0] > n_samples:
            x_train, y_train, bf_index_train = ulib.resampling_pred_matrices(x_train, y_train, bf_index_train, n_samples)

        if i >= num_trained_model:
            prediction_window = x_train.shape[1]
            fit_model(x_train, y_train, model_name, model_map, prediction_window, n_features, epochs, config,
                      cp_dir, output_log_keras, i, features_name, granularity)

        if forecasting_strategy in ['hybrid', 'ouroboros']:
            loaded_model = [keras.models.load_model(path.join(cp_dir, 'model_step_%s' % str(i + 1)))]
            predictor_lookahead = glp.PredictorLookahead(lookahead, n_features, features_name, forecasting_strategy)
            x_train = predictor_lookahead.predictor_nn(loaded_model, x_train, fit_next_model=True)
