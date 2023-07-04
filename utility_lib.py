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
from tensorflow.keras.models import load_model

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pk   
from tensorflow.keras.utils import to_categorical
from matplotlib.ticker import MaxNLocator, LogLocator
import glob  
from itertools import zip_longest
#####################################
#########_MODEL INPUT MAP_###########
#####################################

model_map = {
    'CNN': 'create_CNN_WANG_model',
    'LSTM': 'create_LSTM_model',
    'GRU': 'create_GRU_model',
    'GLOBAL_CNN_RNN': 'create_CNN_RNN_model',
    'SERIES_NET': 'create_SeriesNET',
    'DSANET': 'create_DSANet',
    'STN': 'create_STN',
    'NAIVE':'NAIVE',
    'RFR': 'RFR',
    
}

filter_map={
    'app':['APP','BF_activity'],
    'activity':['ACT','BF_label'],
    'appactivity':['APPACT'],
    'X':['X','BF_label'],
    'appproto':['APPPROTO','BF_activity'],
    'activityproto':['ACTPROTO','BF_label'],
    'appactivityproto':['APPACTPROTO'],
    'proto':['proto','BF_label'],
}

classification_map={
    'app':['APP','BF_label'],
    'activity':['ACT','BF_activity'],
    'appactivity':['APPACT','joint_label']
}

out_map={
    'L4_payload_bytes':'PL',
    'iat_micros':'IAT',
    'packet_dir':'DIR',
    'APP':'APP',
    'ACT':'ACT',
    'VOL_dw':'VOL_dw',
    'PKTS_dw':'PKTS_dw',
    'VOL_up':'VOL_up',
    'PKTS_up':'PKTS_up'
    
}

in_map={
    'PL':'L4_payload_bytes',
    'IAT':'iat_micros',
    'DIR':'packet_dir',  
}
granularity_map={
    'packets':'PKT',
    'aggregated':'AGG',
    'hybrid': 'HYB'
}


#OK
def plot_training_infos(log_p, outputs, losses, scales=['linear', 'linear'],
                        palette='tab20', figsize=(8,4), 
                        ms=4, lw=1, lbs=18, legend_kw=None, 
                        acros=['pl','iat','dir']):
    
    u_losses=list(np.unique(losses))
    log=pd.read_csv(log_p, sep=';')
    log['epoch']=log['epoch'].astype(int)
    metrics=['loss','acc']
    log_cols=list(log.columns)
    
    for m,met in enumerate(metrics):
        for o,out in enumerate(outputs):
            if '%s_output_%s'%(out,met) not in log_cols and met=='acc':
                metrics[m]='accuracy'
                #assert '%s_output_%s'%(out,met) in log_cols, '%s_output_%s not logged\n%s'%(out,met,log_cols)
    
    val=True if 'val_%s_output_loss'%(outputs[0]) in log_cols else False
    colors=sns.color_palette(palette,n_colors=len(outputs)+1)
    #f,ax=plt.subplots(1,2, figsize=(12, 3.5))
    epochs=np.unique(log['epoch'].values)
    glog=log.groupby('epoch')
    y_lb=['Loss','Accuracy[%]']
    for i,met in enumerate(metrics):
        f,ax=plt.subplots(1, figsize=figsize)
        handles=[]
        labels=[]
        if len(outputs)>1:
            for j,out in enumerate(outputs):
                e_mean=glog['%s_output_%s'%(out,met)].mean()
                e_std=glog['%s_output_%s'%(out,met)].std()
                
                e_mean=e_mean*100 if met=='acc' else e_mean
                e_std=e_std*100 if met=='acc' else e_std
                    
                ax.plot(epochs+1,e_mean, ms=ms,lw=lw, marker='o',ls='-',label=acros[j], color=colors[j])
                ax.fill_between(epochs+1,e_mean-e_std, e_mean+e_std, alpha=0.3, color=colors[j])
                
                if val:
                    e_vmean=glog['val_%s_output_%s'%(out,met)].mean()
                    e_vstd=glog['val_%s_output_%s'%(out,met)].std()
                    
                    e_vmean=e_vmean*100 if met=='acc' else e_vmean
                    e_vstd=e_vstd*100 if met=='acc' else e_vstd
                
                    ax.plot(epochs+1,e_vmean, ms=ms,lw=lw, marker='o', ls='--', label='v_'+acros[j], color=colors[j])
                    ax.fill_between(epochs+1,e_vmean-e_vstd, e_vmean+e_vstd, alpha=0.3, color=colors[j])
                handles, labels =ax.get_legend_handles_labels()

        if met=='loss':
            e_mean=glog[met].mean()
            e_std=glog[met].std()
        
            ax.plot(epochs+1,e_mean, ms=ms,lw=lw, marker='o', ls='-',label='global', color=colors[-1])
            ax.fill_between(epochs+1,e_mean-e_std, e_mean+e_std, alpha=0.3, color=colors[-1])
            ax.set_ylabel(y_lb[i],fontsize=lbs)
            if val:
                e_vmean=glog['val_%s'%met].mean()
                e_vstd=glog['val_%s'%met].std()
                ax.plot(epochs+1,e_vmean, ms=ms,lw=lw, marker='o', ls='--', label='v_global', color=colors[-1])
                ax.fill_between(epochs+1,e_vmean-e_vstd, e_vmean+e_vstd, alpha=0.3, color=colors[-1])
            ax.set_yscale(scales[0])
            if scales[0] == 'log':
                ax.xaxis.set_major_locator(LogLocator(10))
                (ym, yM)=ax.get_ylim()
                nym, nyM = None,None
                for exp in np.arange(-9, 2,1):
                    if np.float_power(10, exp) <= ym:
                        nym=np.float_power(10, exp)
                    else:
                        break
                for exp in np.arange(-9, 2,1):
                    if np.float_power(10, exp) >= yM:
                        nyM=np.float_power(10, exp)
                        break
                ax.set_ylim(nym, nyM)    
            ax.legend(**legend_kw)
        else:
            ax.legend(**legend_kw)
            ax.set_yscale(scales[1])
                
        ax.grid(which='major',axis='y',alpha=0.5,ls='--')
        ax.set_xlabel('Epoch',fontsize=18)
        
        ax.tick_params(axis='x',labelsize=lbs, which='both')
        ax.tick_params(axis='y',labelsize=lbs, which='both')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.5, zorder=1, axis='x', which='major')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.5, zorder=1, axis='y', which='major')
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.3, zorder=3, axis='y', which='minor')
        
        savefig=os.path.join(os.path.dirname(log_p),'plot_epoch_infos_%s.pdf'%met)
        plt.tight_layout()
        plt.savefig(savefig, format='pdf', bbox_inches='tight', pad_inches=0.0)
        plt.close()
            


def retrieve_info_from_filename(df_filename):
    filename = df_filename.split('/')[-1]
    filename_token = filename.split('_')
    model = filename_token[0]
    #app = filename_token[1]
    forecasting_strategy = filename_token[5]
    granularity = filename_token[8]
    return model, forecasting_strategy, granularity


def resampling_pred_matrices(x, y, indices, n_samples, random_state=0):
    x_y_indexed = [(idx, matrix, g_truth) for (idx, matrix, g_truth) in zip(indices, x, y)]
    x_y_indexed_resampled = sorted(resample(x_y_indexed, n_samples=n_samples, replace=False, random_state=random_state),
                                   key=itemgetter(0))
    ids_resampled = np.array([x_y_indexed_resampled[i][0] for i in range(len(x_y_indexed_resampled))])
    x_resampled = np.array([x_y_indexed_resampled[i][1] for i in range(len(x_y_indexed_resampled))])
    y_resampled = np.array([x_y_indexed_resampled[i][2] for i in range(len(x_y_indexed_resampled))])
    return x_resampled, y_resampled, ids_resampled

def align_true_predict(y_true, y_pred, forward_step):
    for j in range(len(y_true)):
        y_true[j] = y_true[j][forward_step:]
        y_pred[j] = y_pred[j][:-forward_step]
    return y_true, y_pred


def load_all_pickles(filename,features):
    df_per_feature_dict = dict()
    i = 0
    with lz4.frame.open(filename, "rb") as f:
        while True:
            try:
                # df_per_feature_dict[features_name[i]] = pickle.load(f)
                # df_per_feature_dict[features_name[i]].reset_index(drop=True, inplace=True)
                
                df_per_feature_dict[out_map[features[i]]] = pickle.load(f)
                df_per_feature_dict[out_map[features[i]]].reset_index(drop=True, inplace=True)
                i += 1
            except EOFError:
                break
    return df_per_feature_dict

def get_results_path(model,key,forecasting_strategy, granularity, n_features, win_size,activity_filter=False,app_filter=False,activity_app_filtering=False, root_final_res=os.getcwd(), total_fold=10):
    import glob
    if activity_filter:
        res_dir = path.join(root_final_res,forecasting_strategy, granularity, 'Activity_filter', '_W_' + str(win_size) + '_*', key)
    elif app_filter:
        res_dir = path.join(root_final_res,forecasting_strategy,model, granularity, 'Application_filter', '_W_' + str(win_size) + '_*', key)
    elif activity_app_filtering:
        res_dir = path.join(root_final_res,forecasting_strategy,model, granularity, 'Activity_app_filter', '_W_' + str(win_size) + '_*', key)
    else:
        res_dir = path.join(root_final_res,forecasting_strategy, model,granularity, 'No_filter', '_W_' + str(win_size) + '_*', key)

    res_path=glob.glob(res_dir)
    if len(res_path)==1:
        res_path=res_path[0]
        print(res_path)
        file_output = path.join(res_path, 
                                model + '_' + '_results_' + str(total_fold) + '_Fold_' +forecasting_strategy + '_W_%s_' % str(win_size) + granularity + '_' + str(n_features) +'_features.pickle')

        if os.path.exists(file_output):
            print('2-Results Found: %s'%file_output)
            return file_output
        else:
            print('2-Results not Found: %s'%file_output)
            exit()
    else:
        print('1-Results not found: %s'%res_dir)
        exit()
        

#OK  
def concat_temp_results_2(key,granularity,root_temp_res=os.getcwd(),root_final_res=os.getcwd(),
                          out_features=['L4_payload_bytes','iat_micros','packet_dir'],  query=''):
    file_output=os.path.join(root_final_res,'_'.join([os.path.dirname(root_temp_res).split('/')[-2],key,query]))+'.pickle'
    t_pickles=glob.glob(root_temp_res+'/temp_res_fold*.pickle')
    t_pickles=[t for t in t_pickles if 'aggr' not in t]
    df_per_fold_dict = load_all_pickles(t_pickles[0], granularity,out_features)
    for tp in t_pickles[1:]:
        df_temp_dict = load_all_pickles(tp, granularity,out_features)
        for feature in df_per_fold_dict.keys():
            df_per_fold_dict[feature] = df_per_fold_dict[feature].append(df_temp_dict[feature])
            df_per_fold_dict[feature].reset_index(drop=True, inplace=True)
    
    Path(root_final_res).mkdir(parents=True, exist_ok=True)
    pickle_out = lz4.frame.open(file_output, "wb")
    for feature in df_per_fold_dict.keys():
        #TODO: this save entire dictionary not only the dataframe
        pickle.dump(df_per_fold_dict[feature], pickle_out, )
    pickle_out.close()
    print('Final results saved in ---> %s'%file_output)
    
def concat_temp_results(model, timestamp,key,forecasting_strategy, granularity, n_features, win_size,activity_filter=False,app_filter=False,activity_app_filtering=False, root_temp_res=os.getcwd(),
                        root_final_res=os.getcwd(), order=30):  
    if activity_app_filtering:
        path_to_pickles = path.join(root_temp_res, forecasting_strategy, model,granularity, 'Activity_app_filter', '_W_' + str(win_size) + '_' + str(timestamp),key)
    elif activity_filter:
        path_to_pickles = path.join(root_temp_res, forecasting_strategy, model,granularity, 'Activity_filter', '_W_' + str(win_size) + '_' + str(timestamp),key)
    elif app_filter:
        path_to_pickles = path.join(root_temp_res, forecasting_strategy, model,granularity, 'Application_filter', '_W_' + str(win_size) + '_' + str(timestamp),key)
    else:
        path_to_pickles = path.join(root_temp_res, forecasting_strategy, model,granularity, 'No_filter', '_W_' + str(win_size) + '_' + str(timestamp),key)
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

    if activity_filter:
        res_dir = path.join(root_final_res,forecasting_strategy,model, granularity, 'Activity_filter', '_W_' + str(win_size) + '_' + str(timestamp), key)
    elif app_filter:
        res_dir = path.join(root_final_res,forecasting_strategy,model, granularity, 'Application_filter', '_W_' + str(win_size) + '_' + str(timestamp), key)
    elif activity_app_filtering:
        res_dir = path.join(root_final_res,forecasting_strategy,model, granularity, 'Activity_app_filter', '_W_' + str(win_size) + '_' + str(timestamp), key)
    else:
        res_dir = path.join(root_final_res,forecasting_strategy, model,granularity, 'No_filter', '_W_' + str(win_size) + '_' + str(timestamp), key)
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    file_output = path.join(res_dir, model + '_' + '_results_' + str(total_fold) + '_Fold_' +
                            forecasting_strategy + '_W_%s_' % str(win_size) + granularity + '_' + str(n_features) +
                            '_features.pickle')
    pickle_out = lz4.frame.open(file_output, "wb")
    for feature in df_per_fold_dict.keys():
        pickle.dump(df_per_fold_dict[feature], pickle_out, )
    pickle_out.close()
    print('Final results saved in ---> %s'%file_output)

###############################################################################################################
#Load and Pre-process
def load_results2(results_path,nfeat=3, fine=True):
        from collections import defaultdict
        assert fine, "this function works only with fine-grained results"
        print(results_path)
        dfs= defaultdict(list)
        pin=lz4.frame.open(results_path,'rb')
        for findex in range(nfeat):
            #print(nf)
            dft=pickle.load(pin)
            
            if isinstance(dft, dict):
                k=dft.keys()
                assert len(k)==1, "Error in loading results (dict)"
                dfs[k].append(dft[k])
            else:
                #Inferring feature
                if np.count_nonzero([yt[0] for yt in dft['y_true'].values]) == len(dft['y_true'].values):
                    ft='PL'
                else:
                    ft='DIR' if np.isin(np.concatenate(dft['y_true'].values).astype(int), [0, 1]).all() else 'IAT'
                dfs[ft].append(dft)
        pin.close()
        return dfs, list(dfs.keys())

def load_saved_model_v2(cp_dir,root_models,n_samples, model_name,forecasting_strategy, win_size,granularity,fold,n_features,key, filter=None):
    one_shot=False
    #model_path=path.join(cp_dir, 'model_step_1')
    model_path=path.join(cp_dir, 'model_step_1')
    if os.path.exists(model_path):
        print('loading--> %s'%model_path)
        model=load_model(model_path)
        print('model found and loaded...')
        one_shot=True
        return model, model_path, one_shot
    else:
        print('First failed attempt-->%s'%model_path)
        label_n_samples = '_samples_%s' % n_samples if n_samples > 0 else '_all_samples_'
        if filter=='activity':
            model_path = path.join(root_models, forecasting_strategy , model_name + '_W_' +
                        str(win_size) + label_n_samples, granularity, 'Activity_filter','checkpoint_' + str(fold) + '_'
                        + str(n_features) + '_features' + '_' + key,'model_step_1')
        elif filter=='app':
            model_path = path.join(root_models, forecasting_strategy , model_name + '_W_' +
                        str(win_size) + label_n_samples, granularity, 'Application_filter','checkpoint_' + str(fold) + '_'
                        + str(n_features) + '_features' + '_' + key,'model_step_1')
        elif filter=='appactivity':
            model_path = path.join(root_models, forecasting_strategy , model_name + '_W_' +
                        str(win_size) + label_n_samples, granularity, 'Activity_app_filter','checkpoint_' + str(fold) + '_'
                        + str(n_features) + '_features' + '_' + key,'model_step_1')
        else:
            model_path = path.join(root_models, forecasting_strategy, model_name+ '_W_' +
                        str(win_size) + label_n_samples, granularity, 'No_filter','checkpoint_' + str(fold) + '_'
                        + str(n_features) + '_features_ALL','model_step_1')
            
        if os.path.exists(model_path):
            print('loading--> %s'%model_path)
            model=load_model(model_path)
            print('model found and loaded...')
            return model, model_path,one_shot
        else:
            print('Model not found(%s)...Exit'%model_path)
            return None, None,one_shot

def delete_biflows_by_len(ds,field='packet_dir',min_packets=1, verbose=False):


    count_row_before = ds.shape[0]
    #index = [i for i, v in ds[[field]].iterrows() if len(v[0]) < min_packets]
    #ds.drop(index, inplace=True)
    ds['BF_len']=ds[field].apply(lambda x: len(x))
    ds=ds[ds['BF_len']>(min_packets-1)]
    del ds['BF_len']
    count_row_after = ds.shape[0]

    if verbose:
        print('Deleting of all biflows with less than %s packets, deleted %s/%s Biflows (%.2f%%)' % (min_packets,
            str(count_row_before - count_row_after),
            count_row_before,
            (100 * (count_row_before - count_row_after) / count_row_before))
            )    
    return ds,(count_row_before - count_row_after)


#OK
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
        #ds=ds.reset_index()
    return ds


#OK
def extract_feature_from_df(df, granularity, config):

    features_list = config['BASE']['fine_features'].split(',')
    #padding_list = [0, 0, 0.5]
    padding_list=[]
    for ft in features_list:
        padding_list.append(0.5 if ft=='packet_dir' else 0)
    n_features = len(features_list)
    out_features_list = config['BASE'].get('out_fine_features',config['BASE']['fine_features']).split(',')
    sel_dataset_df = df[features_list]

    return sel_dataset_df, padding_list, n_features, features_list, out_features_list

def get_windowed_labels(BF_indices_windowed,set_indices,labels,encoder=None):
    unique, counts = np.unique(BF_indices_windowed, return_counts=True)
    dT=dict(zip(unique, counts))
    label_index = []
    for it,ti in enumerate(set_indices):
        if it in dT.keys():
            label_index+=[ti]*dT[it]
    labels_windowed=np.take(labels,label_index)
    labels_encoded=None
    if encoder is not None:
        labels_encoded=to_categorical(encoder.transform(labels_windowed))
    
    return labels_encoded,labels_windowed

###############################################################################################################
#Output
def save_labels(train, test, directory_output, timestamp, val=None, kind='labels'):
    # Salvataggio index biflussi utilizzati per il train e test
    if val is None:
        with open(directory_output + str(timestamp) + '_train_w%s.csv'%kind, 'w', encoding="ISO-8859-1", newline='') as csv_vet:
            writer = csv.writer(csv_vet)  # , lineterminator='\n')
            writer.writerow([['FOLD_' + str(r)] for r in range(10)])
            train = zip_longest(*train, fillvalue='')
            writer.writerows(train)

        csv_vet.close()
    else:
        print('Train Windowed Labels exported in -->%s'%(directory_output + str(timestamp) + '_train_w%s.csv'%kind))
        with open(directory_output + str(timestamp) + '_subtrain_w%s.csv'%kind, 'w', encoding="ISO-8859-1", newline='') as csv_vet:
            writer = csv.writer(csv_vet)  # , lineterminator='\n')
            writer.writerow([['FOLD_' + str(r)] for r in range(10)])
            train = zip_longest(*train, fillvalue='')
            writer.writerows(train)
        csv_vet.close()
        with open(directory_output + str(timestamp) + '_validation_w%s.csv'%kind, 'w', encoding="ISO-8859-1", newline='') as csv_vet:
            writer = csv.writer(csv_vet)  # , lineterminator='\n')
            writer.writerow([['FOLD_' + str(r)] for r in range(10)])
            val = zip_longest(*val, fillvalue='')
            writer.writerows(val)
        csv_vet.close()
    with open(directory_output + str(timestamp) + '_test_w%s.csv'%kind, 'w', encoding="ISO-8859-1", newline='') as csv_vet:
        writer = csv.writer(csv_vet)
        writer.writerow([['FOLD_' + str(r)] for r in range(10)])
        test = zip_longest(*test, fillvalue='')
        writer.writerows(test)

    csv_vet.close()

def save_index(train, test, directory_output, timestamp, val=None):
    # Salvataggio index biflussi utilizzati per il train e test
    if val is None:
        with open(directory_output + str(timestamp) + '_train.csv', 'w', encoding="ISO-8859-1", newline='') as csv_vet:
            writer = csv.writer(csv_vet)  # , lineterminator='\n')
            writer.writerow([['FOLD_' + str(r)] for r in range(10)])
            train = zip_longest(*train, fillvalue='')
            writer.writerows(train)
        csv_vet.close()
    else:
        with open(directory_output + str(timestamp) + '_subtrain.csv', 'w', encoding="ISO-8859-1", newline='') as csv_vet:
            writer = csv.writer(csv_vet)  # , lineterminator='\n')
            writer.writerow([['FOLD_' + str(r)] for r in range(10)])
            train = zip_longest(*train, fillvalue='')
            writer.writerows(train)
        csv_vet.close()
        print('Train Index exported in -->%s'%(directory_output + str(timestamp) + '_subtrain.csv'))
        with open(directory_output + str(timestamp) + '_validation.csv', 'w', encoding="ISO-8859-1", newline='') as csv_vet:
            writer = csv.writer(csv_vet)  # , lineterminator='\n')
            writer.writerow([['FOLD_' + str(r)] for r in range(10)])
            val = zip_longest(*val, fillvalue='')
            writer.writerows(val)
        csv_vet.close()
    with open(directory_output + str(timestamp) + '_test.csv', 'w', encoding="ISO-8859-1", newline='') as csv_vet:
        writer = csv.writer(csv_vet)
        writer.writerow([['FOLD_' + str(r)] for r in range(10)])
        test = zip_longest(*test, fillvalue='')
        writer.writerows(test)

    csv_vet.close()

def save_data(res, y_true, y_pred, test, BF_index, fold):
    try:
        id_columns_results = ['y_pred_' + key for key in y_pred.keys()]
    except:
        print(y_pred)
    offset = 0
    try:
        for i in np.unique(BF_index):
            BF_len = (BF_index == i).sum().astype(np.int32)
            indices = [i for i in range(offset, offset + BF_len)]
            y_true_per_BF_dict = {'BF': test[i], 'FOLD': str(fold), 'y_true': y_true.take(indices)}
            y_pred_per_BF_list = [y_pred[key].take(indices) for key in y_pred.keys()]
            y_pred_per_BF_dict = dict((key, value) for (key, value) in zip(id_columns_results, y_pred_per_BF_list))
            final_res_per_BF_dict = {**y_true_per_BF_dict, **y_pred_per_BF_dict}
            res = res.append(final_res_per_BF_dict, ignore_index=True)
            offset += BF_len

        if len(y_pred) == 1:
            res.rename({'y_pred_lookahead_1': 'y_pred'}, axis=1, inplace=True)

    except ValueError:
        logging.error("ValueError in save_data()")

    return res

def save_data_v2(res, y_true, y_pred, test, BF_index, fold):
    try:
        id_columns_results = ['y_pred']
    except:
        print(y_pred)
    offset = 0
    try:
        for i in np.unique(BF_index):
            BF_len = (BF_index == i).sum().astype(np.int32)
            indices = [i for i in range(offset, offset + BF_len)]
            y_true_per_BF_dict = {'BF': test[i], 'FOLD': str(fold), 'y_true': y_true.take(indices)}
            y_pred_per_BF_list = [y_pred.take(indices)]
            y_pred_per_BF_dict = dict((key, value) for (key, value) in zip(id_columns_results, y_pred_per_BF_list))
            final_res_per_BF_dict = {**y_true_per_BF_dict, **y_pred_per_BF_dict}
            res = res.append(final_res_per_BF_dict, ignore_index=True)
            offset += BF_len

        # if len(y_pred) == 1:
        #     res.rename({'y_pred_lookahead_1': 'y_pred'}, axis=1, inplace=True)

    except ValueError:
        logging.error("ValueError in save_data()")

    return res

def save_data_1step(res, y_true, y_pred, test, BF_index, fold, delta=None, full_delta=None):
    offset = 0
    try:
        for i in np.unique(BF_index):
            BF_len = (BF_index == i).sum().astype(np.int32)
            indices = [i for i in range(offset, offset + BF_len)]
            y_per_BF_dict = {'BF': test[i], 'FOLD': str(fold), 'y_true': y_true.take(indices), 'y_pred': y_pred.take(indices)}
            if delta is not None:
                y_per_BF_dict['delta']=str(delta)
            if full_delta is not None:
                y_per_BF_dict['full_delta']=full_delta.take(indices)
            final_res_per_BF_dict = {**y_per_BF_dict}
            res = res.append(final_res_per_BF_dict, ignore_index=True)
            offset += BF_len

    except ValueError:
        logging.error("ValueError in save_data()")

    return res
#OK
def save_labels_df(ds, BF,indexes,labels,typeset,key, fold, w, level, indices=None):
    _,labelsWf=get_windowed_labels(BF,indexes,labels,None)
    ds=ds.append({
            'Key':key,
            'Fold':fold,
            'Set':typeset,
            'Level':level,
            'W':w,
            'Labels':labelsWf,
            'indices':indices
    }, ignore_index=True)
    
    return ds  


##################################################################################################################