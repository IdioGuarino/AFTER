"""
-Functional API keras
-MinMax Scaler: utilizzando questo scaler si effettua lo scaling anche sulle
direzioni (PD). L'insieme delle direzioni è:
    # -1 UPSTREAM
    # +1 DOWNSTREAM
    #  0 PADDING (comune a PL e IAT)
A seguito dello scaling si avranno i valori: {0, .5, 1} e di conseguenza si
puà applicare la sigmoide come funzione di attivazione per {PL, IAT, PD}.
"""
import configparser
import logging
import pickle
import json
import sys
import os, gc
from os import path
from pathlib import Path
import random
import lz4.frame
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import utility_lib as ulib
from lib import fitting_lib as flib
from lib import generic_predictor as gp
from lib import model_library, processing_lib
from lib import nn_lib
from sklearn.model_selection import train_test_split
from utility_lib import plot_training_infos, granularity_map, filter_map, model_map
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def main(configin, gpu_id=None):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    global config
    global directory_output
    global exp_dir
    global model_name
    global granularity
    global in_n_features
    global out_n_features
    config=configin
    print('Pre Analysis Operations....')

    file_input = config['I/O']['app_from_dataset']
    
    dataset_in_df=ulib.load_dataset(file_input)
    try:
        del dataset_in_df['BF_L4_raw_payload']
    except:
        print('NO PAY in dataset')
        
    dataset_in_df=dataset_in_df[dataset_in_df['BF_label'].isin(['Zoom','Teams','Skype','Webex'])]
    dataset_in_df['joint_label']=dataset_in_df.apply(lambda x: '-'.join([x['BF_label'],x['BF_activity']]), axis=1)
    
    # 1. Dataset read
    samples = config['OPTIONS'].getboolean('sampling',False)
    granularity = config['I/O']['granularity']
    if samples:
        dwns=0.1
        print('Warning Dataset Downsampling at %s'%dwns)
        dataset_in_df,_,=train_test_split(dataset_in_df,test_size=None, train_size=dwns,random_state=0,stratify=dataset_in_df['joint_label'].values)
        unique,counts=np.unique(dataset_in_df['joint_label'].values,return_counts=True)
        print('Labels in Dataset: ',dict(zip(unique,counts)))
        dataset_in_df=dataset_in_df.reset_index()
        slabels=dataset_in_df['joint_label'].values
    else:
        #dataset_in_df=dataset_in_df.reset_index()
        slabels=dataset_in_df['joint_label'].values
    
    print('Loading data')
    # 2. Extracting features
    
    returned_df, padding_list, n_features, in_features_list, out_features_list = ulib.extract_feature_from_df(dataset_in_df, granularity,config)
    in_n_features=n_features
    out_n_features=len(out_features_list)
    notsame=False
    if in_n_features!=out_n_features:
        notsame=True
        out_features_indices=[in_features_list.index(feat) for feat in out_features_list]
        
    print('INFO: Predicting %s'%out_features_list)
    print('Directories and log configuration...')

    #BASE
    win_size = config['BASE'].getint('win_size')
    app = config['BASE']['app']
    total_fold = config['BASE'].getint('fold')
    n_samples = config['BASE'].getint('n_samples') if config['BASE'].getint('n_samples') > 0 else 0
    
    #MODEL
    model_name = config['MODEL']['model']
    epochs = config['MODEL'].getint('epochs')
    
    #I/O
    directory_output = config['I/O']['output_path']
    
    #OPTIONS
    validation = config['OPTIONS'].getboolean('validation_set',False)
    print('Validating: %s'%validation)
    
    filtering=config['OPTIONS'].get('filtering','X')
    use_training_set = config['OPTIONS'].getboolean('use_training_set')
    fit_model = config['OPTIONS'].getboolean('fit_model')
    assert filtering in filter_map.keys(), 'Wrong filter mode'
    proto=None
    if 'proto' in filtering:
        proto=config['BASE'].get('proto','TCP')
        print('Filtering on %s protocol'%proto)
    main_exp_dir = path.join(directory_output,
                             '_'.join([model_name,
                                       granularity_map[granularity], 
                                       '%sW'%str(win_size), 
                                       '%sF'%str(out_n_features),
                                        'X', 
                                        'X', 
                                        'X', 
                                        filter_map[filtering][0]
                                                      ]))
    Path(main_exp_dir).mkdir(parents=True, exist_ok=True)
    

    protomap={
        'TCP':'6',
        'UDP':'17'
    }    

    scaled_features_list=in_features_list.copy()

    applabels = dataset_in_df['BF_label'].values
    activitylabels=dataset_in_df['BF_activity'].values if 'BF_activity' in dataset_in_df.columns else None
    protolabels=dataset_in_df['BF_quintuple'].apply(lambda x: x.split(',')[-1]).values
    del dataset_in_df
    if model_name not in model_map:
        sys.exit(
            'Not supported metod. Try: [CNN, LSTM, GRU, GLOBAL_CNN_RNN, SERIES_NET, DSANet, STN] or visit README')

    # 3. K-Fold

    sel_dataset_df =returned_df
    kf = processing_lib.dataset_split(sel_dataset_df.shape[0], y=slabels, k=total_fold, random_state=0)

    for fold in range(0,total_fold):
        train, test = kf[fold][0], kf[fold][1]
        print('_______________________________________________________________')
        print('______________________ FOLD %d / %d ___________________________' % (fold + 1, total_fold))
        print('_______________________________________________________________')

        key='ALL'
        if filtering not in ['X','proto']:
            #TODO: fix activity filtering
            key=app
            if proto is not None:
                train=[it for (it,lb, plb) in zip(train, np.take(slabels,train, axis=0), np.take(protolabels,train, axis=0)) if app in lb and plb==protomap[proto]]
                test=[it for (it,lb, plb) in zip(test, np.take(slabels,test, axis=0), np.take(protolabels,test, axis=0)) if app in lb and plb==protomap[proto]]
            else:
                train=[it for (it,lb) in zip(train,np.take(slabels,train, axis=0)) if app in lb]
                test=[it for (it,lb) in zip(test,np.take(slabels,test, axis=0)) if app in lb]
            train_df = np.take(sel_dataset_df, train, axis=0)
            test_df = np.take(sel_dataset_df, test, axis=0)
        else:
            if proto is not None:
                key=proto
                train=[it for (it,lb) in zip(train,np.take(protolabels,train, axis=0)) if lb==protomap[proto]]
                test=[it for (it,lb) in zip(test,np.take(protolabels,test, axis=0)) if lb==protomap[proto]]
    
            train_df = np.take(sel_dataset_df, train, axis=0)
            test_df = np.take(sel_dataset_df, test, axis=0)
            
        exp_dir=os.path.join(main_exp_dir,key)
        Path(exp_dir).mkdir(parents=True, exist_ok=True)
        res_dir=os.path.join(exp_dir,'results')
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        print('General Info initialization...')

        print('(BF) TRAIN/TEST:',np.shape(train),np.shape(test))

        train_indexes = train_df.index.tolist()
        test_indexes = test_df.index.tolist()

        if validation:
            kfv=processing_lib.dataset_split(len(train_df), y=np.take(slabels,train) , k=5, random_state=0)
            train, sval=kfv[0][0], kfv[0][1]
            
            print('(BF) VALIDATION:',np.shape(train),np.shape(sval))
            val_df=np.take(train_df, sval, axis=0)
            train_df=np.take(train_df, train, axis=0)
            
            val_indexes = val_df.index.tolist()
            train_indexes = train_df.index.tolist()
        # 4. Create prediction matrix for training and test-set
        print('Loading prediction matrices and ground-truth')
        
        file_output = path.join(res_dir,'first_values_by_fold')+'.pickle'
        
        if os.path.exists(file_output):
            print('A.%s'%fold)
            with lz4.frame.open(file_output, "rb") as f:
                firstv=pickle.load(f)
            firstv[fold]={}
            for feature in out_features_list:
                firstv[fold][feature]={} 
                firstv[fold][feature]['BF']=test_indexes
                firstv[fold][feature]['y_true_0']=[x[0] for x in test_df[feature].values]
                firstv[fold][feature]['lv0']=applabels[test_indexes]
                firstv[fold][feature]['lv1']=activitylabels[test_indexes]
                firstv[fold][feature]['lv2']=protolabels[test_indexes]
                
            with lz4.frame.open(file_output, "wb") as f:
                pickle.dump(firstv, f )
        else:
            print('B.%s'%fold)
            firstv={}
            firstv[fold]={}
            for feature in out_features_list:
                firstv[fold][feature]={} 
                firstv[fold][feature]['BF']=test_indexes
                firstv[fold][feature]['y_true_0']=[x[0] for x in test_df[feature].values]
                firstv[fold][feature]['lv0']=applabels[test_indexes]
                firstv[fold][feature]['lv1']=activitylabels[test_indexes]
                firstv[fold][feature]['lv2']=protolabels[test_indexes]
            with lz4.frame.open(file_output, "wb") as f:
                pickle.dump(firstv, f )

        x_train, y_train, BF_train = nn_lib.create_dataset(train_df.values, look_back=win_size,
                                                    pad=padding_list,
                                                    features_last=True, multi_output=False,verbose=True)
        print('Train-Set---> OK')
        x_val=None 
        y_val=None 
        BF_val=None
        if validation:
            x_val, y_val, BF_val = nn_lib.create_dataset(val_df.values, look_back=win_size,
                                                                        pad=padding_list,
                                                                        features_last=True, multi_output=False, verbose=True)
            print('Validation-Set---> OK')
        x_test, y_test, BF_test = nn_lib.create_dataset(test_df.values, look_back=win_size,
                                                    pad=padding_list,
                                                    features_last=True, multi_output=False, verbose=True)
        print('Test-Set---> OK')
        
        ds=pd.DataFrame(columns=['Key','Fold','Set','Level','W','Labels', 'indices'])
        ds['Labels']=ds['Labels'].astype(object)
        ds=ulib.save_labels_df(ds,BF_train,train_indexes,applabels,'Train',key,fold,win_size,'App', train)
        ds=ulib.save_labels_df(ds,BF_test,test_indexes,applabels,'Test',key,fold,win_size,'App', test)
        ds=ulib.save_labels_df(ds,BF_train,train_indexes,activitylabels,'Train',key,fold,win_size,'Activity',train)
        ds=ulib.save_labels_df(ds,BF_test,test_indexes,activitylabels,'Test',key,fold,win_size,'Activity',test)
        ds=ulib.save_labels_df(ds,BF_train,train_indexes,protolabels,'Train',key,fold,win_size,'Proto',train)
        ds=ulib.save_labels_df(ds,BF_test,test_indexes,protolabels,'Test',key,fold,win_size,'Proto',test)
        ds.to_parquet(os.path.join(exp_dir,'%sF_labels.parquet'%fold))
        
        del ds, train

        # 5-a. Pre-processing: data scaling
        y_train = y_train.reshape((-1, in_n_features))
        y_test = y_test.reshape((-1, in_n_features))

        if validation:
            y_val=y_val.reshape((-1, in_n_features))
            
        if not config['OPTIONS'].getboolean('multi_scale'):
            x_train, y_train, scaler = model_library. \
                transform_data(x_train=x_train, y_train=y_train, n_features=in_n_features, prediction_window=win_size,
                        multi_scale=False, multi_scale_features=config['OPTIONS'].getint('multi_scale_features')) 
                
        else:
            x_train, y_train, scaler, scaler_2 = \
            model_library.transform_data(x_train=x_train, y_train=y_train, n_features=in_n_features,
                                        prediction_window=win_size, multi_scale=True,
                                        multi_scale_features=config['OPTIONS'].getint('multi_scale_features'))
        
        if validation:
            x_val, y_val = model_library. \
                            transform_data(x_test=x_val, y_test=y_val, n_features=in_n_features, 
                                            prediction_window=win_size,
                                            multi_scale=False, multi_scale_features=config['OPTIONS'].getint('multi_scale_features'),
                                            scaler=scaler)
        else:
            x_val=None
            y_val=None                    
        x_test, y_test = model_library. \
        transform_data(x_test=x_test, y_test=y_test, n_features=in_n_features, prediction_window=win_size,
                    multi_scale=False, multi_scale_features=config['OPTIONS'].getint('multi_scale_features'),
                    scaler=scaler)

        
        if notsame:
            y_train=y_train[:,out_features_indices]
            y_test=y_test[:,out_features_indices]
            if validation:
                y_val=y_val[:,out_features_indices]
            print('INFO (notsame):',np.shape(y_train))
                


        logging.info('[FOLD] : %s' % str(fold))

        logging.info('[FOLD] : %s' % str(fold))
        cp_dir=os.path.join(exp_dir,'models','_'.join(['checkpoint',str(fold)]))
        Path(cp_dir).mkdir(parents=True, exist_ok=True)
        print('CPDIR--> %s'%cp_dir)
        output_log_keras = os.path.join(exp_dir,'models')
        take_exe_time = config['OPTIONS'].getboolean('take_exe_time')
        
        # 6. Fit models and store it

        if fit_model:
            if take_exe_time:
                args = (x_train, y_train, BF_train, model_name, model_map,
                        win_size, in_n_features, epochs, config, cp_dir,
                        output_log_keras, out_features_list, granularity, n_samples, 
                        validation,x_val,y_val)
                model, train_exe_time = processing_lib.timeit(flib.fit_model_wrapper, args)
            else:
                model=flib.fit_model_wrapper(x_train, y_train, BF_train, model_name, model_map,
                                        win_size, in_n_features, epochs, config, cp_dir, output_log_keras,
                                        out_features_list, granularity, 
                                        n_samples, validation=validation,x_val=x_val,y_val=y_val)
        # 7. Predict
        predictor= gp.Predictor(out_n_features, out_features_list)
        
        #TODO: fix prediction when in_n_features!=out_n_features
        trainPredict = None
        train_cls_predict=None
        if use_training_set:
            # Forecasting on training set
            print('Train-set predicting...')
            train_predicted_dict = predictor.predictor_nn(model, x_train)
            trainPredict = np.zeros(shape=(y_train.shape[0], n_features))
        del x_train
        
        
        # Forecasting on test set
        print('Test-set predicting...')
        if take_exe_time:
            args = (model, x_test)
            test_predicted_dict, test_exe_time = processing_lib.timeit(predictor.predictor_nn, args)
            logging.info('Test predict time: %s' % str(test_exe_time))
        else:
            test_predicted_dict = predictor.predictor_nn(model, x_test)
        
        # Inverse Scaling
        if set(scaled_features_list).issubset(out_features_list):
            print('FEATURES INPUT: ', n_features)
            print('MULTI SCALING: ', config['OPTIONS']['multi_scale'])
            
            testPredict=np.zeros(shape=(y_test.shape[0], len(scaled_features_list)))
            for j,feature in enumerate(scaled_features_list):
                    if use_training_set:
                        trainPredict[:, j] = train_predicted_dict[feature][:, 0]
                    testPredict[:, j] = test_predicted_dict[feature][:, 0]

            if granularity == 'packets' and n_features > 3 and not config['OPTIONS'].getboolean('multi_scale'):
                # Concatenation of y because there are some feature that only serve as support
                if use_training_set:
                    trainPredict[:, 3:] = y_train[:, 3:]

            if use_training_set:
                logging.warning('Predictions on training set are not saved into output table')
                y_train = scaler.inverse_transform(y_train )
            else:
                del y_train

            y_test = scaler.inverse_transform(y_test)
            if use_training_set:
                trainPredict = scaler.inverse_transform(trainPredict)
                
            testPredict = scaler.inverse_transform(testPredict)
            if granularity == 'packets' and 'packet_dir' in out_features_list:
                    dir_index = out_features_list.index('packet_dir')
                    testPredict[:, dir_index] = np.where(testPredict[:, dir_index] < 0.5, 0,
                                                        np.where(testPredict[:, dir_index] > 0.5, 1, random.choice([0, 1])))
            results_pred_dict = dict()
            results_true_dict = dict()

            for j,feature in enumerate(scaled_features_list):
                results_true_dict[feature] = y_test[:, j]
                results_pred_dict[feature]= testPredict[:, j]

                
            del testPredict, trainPredict, y_test
            
            logging.warning('Predictions on training set are not saved into output table')
        else:
            #TODO: continue here
            results_pred_dict = dict()
            results_true_dict = dict()

            for j,feature in enumerate(scaled_features_list):
                if feature in out_features_list:
                    print('WARNING: assuming MinMaxScaler in inverse transformation')
                    fscaler=MinMaxScaler()
                    fscaler.min_, fscaler.scale_, fscaler.data_min_,fscaler.data_max_,fscaler.data_range_=scaler.min_[j], scaler.scale_[j], scaler.data_min_[j],scaler.data_max_[j],scaler.data_range_[j]
                    out_f_index=out_features_list.index(feature)
                    
                    results_true_dict[feature]=np.squeeze(fscaler.inverse_transform(np.expand_dims(y_test[:,out_f_index],1)),1)
                    results_pred_dict[feature]=np.squeeze(fscaler.inverse_transform(np.expand_dims(test_predicted_dict[feature][:, 0],1)),1)
                    
                    if granularity == 'packets' and feature=='packet_dir':
                        results_true_dict[feature]=np.where(results_true_dict[feature] < 0.5, 0,
                                                        np.where(results_true_dict[feature] > 0.5, 1, random.choice([0, 1])))
                        results_pred_dict[feature]=np.where(results_pred_dict[feature] < 0.5, 0,
                                                        np.where(results_pred_dict[feature] > 0.5, 1, random.choice([0, 1])))
            del y_test
        gc.collect()
            
        print(results_true_dict.keys())
        id_columns = ['BF', 'FOLD', 'y_true', 'y_pred']
        # Saving results
        results_dict = dict((k, v) for (k, v) in zip(out_features_list, [pd.DataFrame(columns=id_columns)
                                                                    for _ in out_features_list]))

        file_output = path.join(res_dir,'_'.join(['temp_res_fold',str(fold)]))+'.pickle'

        pickle_out = lz4.frame.open(file_output, "wb")
        for feature in out_features_list:            
            results_dict[feature] = ulib.save_data_v2(results_dict[feature], results_true_dict[feature], results_pred_dict[feature],
                                            test, BF_test, fold)
            pickle.dump(results_dict[feature], pickle_out, )
            
        pickle_out.close()

        print('Temporary results saved in ---> %s'%file_output)
        pickle_out.close()
        
        K.clear_session()
        
        my_config_parser_dict = {s:dict(config.items(s)) for s in config.sections()}
        cfg_json=os.path.join(exp_dir,'config.json')
        with open(cfg_json, 'w') as f:
            json.dump(my_config_parser_dict,f)

        if fold == total_fold - 1:
            ulib.concat_temp_results_2(key,granularity,res_dir,root_final_res=config['I/O']['final_res_root'],out_features=out_features_list)
        print('DONE')

        fold+=1

    automatic_shutdown = config['OPTIONS'].getboolean('automatic_shutdown')
    if automatic_shutdown:
        os.system('shutdown -s')

if __name__ == "__main__":
    cfg_file=sys.argv[1]
    config = configparser.ConfigParser()
    config.read(cfg_file)
    gpu_id=sys.argv[2]
    main(config,gpu_id)