#import keras
from os import path
from lib import custom_callbacks as cbacks
from lib import model_library
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
import psutil
import os
import numpy as np
import random as python_random
import utility_lib as ulib
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import joblib
supported_models = {
    'RFR': RandomForestRegressor,
    'LR': LinearRegression,
    'KNR': KNeighborsRegressor
}

def get_used_ram():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def fit_model_wrapper(x_train, y_train, bf_index_train, model_name, model_map, prediction_window,
                        in_n_features, epochs, config, cp_dir, output_log_keras, features_name=None,
                        granularity=None , n_samples=0, 
                        validation=False, x_val=None, y_val=None):
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
            x_train, y_train, bf_index_train =ulib.resampling_pred_matrices(x_train, y_train, bf_index_train, n_samples)

        prediction_window = x_train.shape[1]
        
        model=fit_model_v2( x_train, y_train, model_name, model_map, prediction_window, 
                            in_n_features, epochs, config,
                            cp_dir, output_log_keras, 0, 
                            features_name, granularity, 
                            validation=validation,x_val=x_val,y_val=y_val)
        return model
    
    
    
def fit_model_v2(   x_train, y_train, model_name, model_map, prediction_window, 
                    in_n_features, epochs, config, cp_dir,
                    output_log_keras, idx=0, features_name=None, 
                    granularity=None, validation=False,
                    x_val=None, y_val=None):
    
    out_n_features = len(features_name)
    print('Used RAM before training:', get_used_ram())
    print(features_name)
    if model_name in supported_models.keys():
        np.shape(y_train)[1]==1, 'ML models work only with one feature at time'
        ml_model = supported_models[model_name]
        model = ml_model(n_jobs=4)

        x_train=np.reshape(x_train,(-1,in_n_features*prediction_window))
        y_train=np.squeeze(y_train, axis=1)
        print('Training starts.')

        path_to_save_cp = path.join(cp_dir, 'model_step_%s.sav' % (str(idx + 1)))
        model.fit(x_train, y_train)
        joblib.dump(model, path_to_save_cp,compress=1)
    else:
        # MODEL
        experiment_model = model_library.Models(config[model_name], 
                                                prediction_window, in_n_features, features_name)

        # CALLBACKS -> EARLY STOPPING
        
        if validation:
            es = EarlyStopping(monitor='val_loss',
                            min_delta=float(config['MODEL']['min_delta']),
                            mode='min',
                            patience=int(config['MODEL']['patience']),
                            verbose=1,
                            restore_best_weights=True)
        else:
            es = EarlyStopping(monitor='loss',
                            min_delta=float(config['MODEL']['min_delta']),
                            mode='min',
                            patience=int(config['MODEL']['patience']),
                            verbose=1,
                            restore_best_weights=True)
        # CALLBACKS -> CSV LOGGER
        csv_logger = CSVLogger(output_log_keras + 
                               '_model_step_' + str(idx + 1) + '_keras_log.csv', append=True,
                            separator=';')

        # CALLBACK -> SAVE CHECK POINT
        path_to_save_cp = path.join(cp_dir, 'model_step_%s' % (str(idx + 1)))
        if validation:
            save_cp = ModelCheckpoint(path_to_save_cp, monitor='val_loss', 
                                      verbose=1, save_best_only=False, save_weights_only=False)
        else:
            save_cp = ModelCheckpoint(path_to_save_cp, monitor='loss', verbose=0, 
                                      save_best_only=False, save_weights_only=False)
        # CALLBACK -> TIME HISTORY
        time_callback = cbacks.TimeEpochs()

        # MODEL________________
        model = getattr(experiment_model, model_map[config['MODEL']['model']])()

        config_loss_list = [config['COMPILE']['PL_loss'], config['COMPILE']['IAT_loss'], config['COMPILE']['PD_loss']]
        
        config_loss_weights_list = [config['COMPILE'].getfloat('pl_weight'), config['COMPILE'].getfloat('iat_weight'),
                                config['COMPILE'].getfloat('pd_weight')]
        
            
        loss = dict((k, v) for (k, v) in zip(model.output_names, config_loss_list))
        loss_weights = dict((k, v) for (k, v) in zip(model.output_names, config_loss_weights_list))
        
        metrics_list = out_n_features * ['accuracy']
        if config['I/O'].getboolean('classification',False):
            metrics_list += ['accuracy']
        metrics = dict((k, v) for (k, v) in zip(model.output_names, metrics_list))
        model.compile(loss=loss, loss_weights=loss_weights, optimizer='adam', metrics=metrics)

        print(model.summary())

        if model_name == 'STN':
            x_train = np.reshape((-1, prediction_window, 1, 1, in_n_features))

        np.random.seed(0)
        python_random.seed(0)
        tf.random.set_seed(0)
        if config['I/O'].getboolean('add_pkt_indices'):
            y_train = [y_train[:, i] for i in range(y_train.shape[1]-1)]
            if y_val is not None:
                y_val = [y_val[:, i] for i in range(y_val.shape[1]-1)]
        else:
            y_train = [y_train[:, i] for i in range(y_train.shape[1])]
            if y_val is not None:
                y_val = [y_val[:, i] for i in range(y_val.shape[1])]

        if validation:
            model.fit({"Input_pred": x_train}, 
                      y_train, validation_data=({"Input_pred": x_val},y_val),epochs=epochs, 
                      batch_size=32, verbose=2, callbacks=[es, time_callback, csv_logger, save_cp])
        else:
            model.fit({"Input_pred": x_train}, 
                      y_train,epochs=epochs, batch_size=32, verbose=1, 
                      callbacks=[es, time_callback, csv_logger, save_cp])
        print('Used RAM after training:', get_used_ram())
        print('Used RAM after clear session:', get_used_ram())
    return model