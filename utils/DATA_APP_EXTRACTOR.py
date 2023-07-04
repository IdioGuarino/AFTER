# -*- coding: utf-8 -*-
import configparser
import gc
import logging
import pickle
import lz4.frame
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow


def x_matrix_rows_generator(data, w, lookahead, forecasting_strategy):
    padding_positions = lookahead if forecasting_strategy == 'hybrid' else 0
    for x in range(int(len(data) - w)):
        yield np.pad(data[x:w], [(0, padding_positions), (0, 0)], 'constant')
        w = w + 1


def compute_payload(k, data, save_mode='parquet'):
    PL = len(data['L4_payload_bytes'])
    RP = np.true_divide(data['BF_L4_raw_payload'], 255)

    for p in range(len(RP)):
        RP[p] = np.pad(RP[p], (0, 1460 - len(RP[p])), constant_values=(0, 0))
    RP = np.stack(RP)
    if PL != len(RP):
        y = np.zeros((PL, 1460))
        y[:RP.shape[0]] = RP
        RP = y.T
        del y
    else:
        RP = RP.T

    return {'BF': k, 'Data': list(RP)} if save_mode == 'parquet' else {'BF': k, 'Data': RP}


flag = ['C', 'E', 'U', 'A', 'P', 'R', 'S', 'F']


def check_flag(code):
    return [code.find(flag[i]) for i in range(len(flag))]


def compute_tcp_flag(k, data, w):
    TCPF = np.array([check_flag(data['BF_TCP_flags'][i]) for i in range(len(data['BF_TCP_flags']))])
    TCPF = np.where(np.array(TCPF) != -1, 1, 0).T
    # r = np.full((len(data)+w,8), .5)
    # r[-len(TCPF):,:]=TCPF

    return {'BF': k, 'Data': TCPF}


if __name__ == "__main__":

    ## WINDOW_SIZE={5,10,30,60},
    ## PICKLE_NAME_OUT (nome output senza .pickle) es: Dropbox
    ## PICKLE_NAME_IN (nome input) es: dataset_df_MESSAGE_FINAL_SATURATED.pickle
    ## PAD_TYPE: {constant, edge, linear_ramp, maximum, mean, median, minimum, reflect, reflect_type = 'odd', symmetric, wrap, empty}

    # CONFIG
    config = configparser.ConfigParser()
    config.read('config.ini')

    # LOG FILE
    logging.basicConfig(filename='log_extracted_app.txt', level=logging.DEBUG)

    # INIZIALIZZAZIONI
    MD = dict()  # DIZIONARIO PER I METADATI, INDICIZZATO TRAMITE FLOW_INDEX
    DATA = dict()  # DIZIONARIO PER LA MATRICE DI PREDIZIONE E INDEX : {L4_PAYLOAD_BYTES_INPUT, L4_PAYLOAD_BYTES_OUTPUT, IAT_INPUT, IAT_OUTPUT, PD_INPUT, PD_OUTPUT, FLOW_INDEXES}
    INFO = dict()  # DIZIONARIO INFO GENERICHE SUL FILE: {DATASET, APPLICAZIONE, MAT_SIZE}

    forecasting_strategy = config['OPTIONS']['forecasting_strategy']
    if forecasting_strategy not in ['direct', 'recursive', 'hybrid']:
        forecasting_strategy = 'recursive'
    INFO['APP'] = config['BASE']['app']
    INFO['PAD_TYPE'] = int(config['BASE']['pad_type'])
    lookahead = config['OPTIONS'].getint('lookahead')
    lookahead = 1 if lookahead <= 0 else lookahead
    lookahead_keys = ['lookahead_%s' % i for i in range(1, lookahead + 1)]

    opt_strategy = '_hybrid_%d_step' % lookahead if forecasting_strategy == 'hybrid' else ''

    filename = INFO['APP'] + opt_strategy + '.pickle'  # pickle file name
    logging.info('pickle file name : %s' % filename)

    # INIZIALIZZAZIONE DIZIONARIO 'DATA'
    DATA = {'L4_PAYLOAD_BYTES_INPUT': [],
            'L4_PAYLOAD_BYTES_OUTPUT': [],
            'IAT_INPUT': [],
            'IAT_OUTPUT': [],
            'PD_INPUT': [],
            'PD_OUTPUT': [],
            'TWIN_INPUT': [],
            'TWIN_OUTPUT': [],
            'FLOW_INDEX': [],
            'PACKET_ID': []}

    # STRUTTURE PER ELABORAZIONE TEMPORANEA
    L4B_temp = []
    IAT_temp = []
    B_temp = []
    S_temp = []
    DIRECTION_temp = []
    TCP_WIN_temp = []

    # LISTA BIFLUSSI NULLI O PARI A 1
    NULL_BF = list()

    # INDEX PER BIFLUSSO NEL DIZIONARIO
    B_INDEX = 0

    # FINESTRA DI PREDIZIONE
    INFO['PRED_WIN'] = int(config['BASE']['win_size'])

    # 60 valore di default per la finestra di predizione
    default_win = 60

    if INFO['PRED_WIN'] not in [5, 10, 30, 60]:
        INFO['PRED_WIN'] = default_win
        logging.warning('FINESTRA DI PREDIZIONE IMPOSTATA AL VALORE DI DEFAULT: %d' % default_win)

    # Accesso al dataset
    INFO['DATASET'] = config['I/O']['dataset_path']
    no_support = False
    extract_payload = False

    if INFO['DATASET'].find('pickle') != -1:
        data = pd.read_pickle(INFO['DATASET'])
        INFO['DATASET_encoded'] = INFO['DATASET'][INFO['DATASET'].find('pickle') - 9:INFO['DATASET'].find('pickle') - 1]
    elif INFO['DATASET'].find('parquet') != -1:
        data = pd.read_parquet(INFO['DATASET'])
        INFO['DATASET_encoded'] = INFO['DATASET'][
                                  INFO['DATASET'].find('parquet') - 9:INFO['DATASET'].find('parquet') - 1]
    else:
        no_support = True

    if config['BASE'].getboolean('payload') and INFO['DATASET'].find('payload') == -1:
        no_support = True
    elif config['BASE'].getboolean('payload') and INFO['DATASET'].find('payload') != -1:
        payload = pd.DataFrame()
        extract_payload = True

    assert no_support is False, logging.error('Data NOT SUPPORTED, check config file')

    extract_tcp_flags = False

    if config['BASE'].getboolean('tcp_flag'):
        tcp_flag = pd.DataFrame()
        extract_tcp_flags = True

    # Dati dell'applicazione scelta
    data = data.loc[data['BF_label'] == INFO['APP']]

    # Salvo riferimento dataset
    logging.info('dataset : %s' % INFO['DATASET'])
    logging.info('dataset (encoded) : %s' % INFO['DATASET_encoded'])
    # ESTRAZIONE DATI
    for keys in tqdm(range(len(data))):

        B_temp = []
        S_temp = []

        # CARICAMENTO DEI DATI DELLA CHIAVE
        L4B_temp = np.array(data.iloc[keys]['L4_payload_bytes'])

        if len(L4B_temp) > lookahead:
            IAT_temp = np.array(data.iloc[keys]['iat_micros'])
            TCP_WIN_temp = np.array(data.iloc[keys]['BF_TCP_win_size'])
            DIRECTION_temp = np.array(data.iloc[keys]['packet_dir'])
            DIRECTION_temp = np.where(DIRECTION_temp == 0, -1, DIRECTION_temp)  # [UP=-1, DW=1]
            DIRECTION_temp = DIRECTION_temp.astype(int)

            if config['OPTIONS'].getboolean('apply_TCP_win_scale'):
                DF_scale = data.iloc[keys]['DF_WS']
                UF_scale = data.iloc[keys]['UF_WS']

                if all([~np.isnan(DF_scale), ~np.isnan(UF_scale)]):
                    win_scale = np.where(TCP_WIN_temp < 0, int(UF_scale), int(DF_scale))
                    TCP_WIN_temp = np.multiply(abs(TCP_WIN_temp), win_scale)
                else:
                    TCP_WIN_temp = np.multiply(abs(TCP_WIN_temp), 0)
            duration = (np.sum(IAT_temp)) * 1e-6

            # LEFT PADDING
            # -----> PAD_TYPE: {constant, edge, linear_ramp, maximum, mean, median, minimum, reflect, reflect_type = 'odd', symmetric, wrap, empty}

            nrows = L4B_temp.size
            L4B_temp = np.pad(L4B_temp, (INFO['PRED_WIN'] - 1, 0), constant_values=(INFO['PAD_TYPE'], 0))
            IAT_temp = np.pad(IAT_temp, (INFO['PRED_WIN'] - 1, 0), constant_values=(INFO['PAD_TYPE'], 0))
            DIRECTION_temp = np.pad(DIRECTION_temp, (INFO['PRED_WIN'] - 1, 0),
                                    constant_values=(INFO['PAD_TYPE'], 0))
            TCP_WIN_temp = np.pad(TCP_WIN_temp, (INFO['PRED_WIN'] - 1, 0), constant_values=(INFO['PAD_TYPE'], 0))

            L4B_temp = np.reshape(L4B_temp, (-1, 1))
            IAT_temp = np.reshape(IAT_temp, (-1, 1))
            DIRECTION_temp = np.reshape(DIRECTION_temp, (-1, 1))
            TCP_WIN_temp = np.reshape(TCP_WIN_temp, (-1, 1))

            # GENERAZIONE [<------------INPUT------------>][<-OUTPUT->]

            X_B_L4B = np.stack([x for x in x_matrix_rows_generator(L4B_temp, INFO['PRED_WIN'], lookahead,
                                                                   forecasting_strategy)])
            X_B_IAT = np.stack([x for x in x_matrix_rows_generator(IAT_temp, INFO['PRED_WIN'], lookahead,
                                                                   forecasting_strategy)])
            X_B_PD = np.stack([x for x in x_matrix_rows_generator(DIRECTION_temp, INFO['PRED_WIN'], lookahead,
                                                                  forecasting_strategy)])
            X_B_TWIN = np.stack([x for x in x_matrix_rows_generator(TCP_WIN_temp, INFO['PRED_WIN'], lookahead,
                                                                    forecasting_strategy)])

            Y_B_L4B = L4B_temp[INFO['PRED_WIN']:len(L4B_temp)]
            Y_B_IAT = IAT_temp[INFO['PRED_WIN']:len(L4B_temp)]
            Y_B_PD = DIRECTION_temp[INFO['PRED_WIN']:len(L4B_temp)]
            Y_B_TWIN = TCP_WIN_temp[INFO['PRED_WIN']:len(L4B_temp)]

            L = len(X_B_L4B)

            # GENERAZIONE INDICI CHE IDENTIFICANO IL BIFLUSSO
            B_temp = np.full(L, B_INDEX)

            # GENERAZIONE ARRAY CHE IDENTIFICANO I PACCHETTI IN OGNI RIGA
            '''
            ES. Lunghezza i-esimo Biflusso Bi = 21
                La notazione è [PLeft : PRight]
                    |-> Es. [4-13] sono presenti i pacchetti
                    |che vanno da 4 a 13 nella riga
                [     P_Left    ]
                [ 60  30  10  5  P_Right]
                [ 1,  1,  1,  1,  1]
                [ 1,  1,  1,  1,  2]
                [ 1,  1,  1,  1,  3]
                [ 1,  1,  1,  1,  4]
                [ 1,  1,  1,  1,  5]
                [ 1,  1,  1,  2,  6]
                [ 1,  1,  1,  3,  7]
                [ 1,  1,  1,  4,  8]
                [ 1,  1,  1,  5,  9]
                [ 1,  1,  1,  6, 10]
                [ 1,  1,  2,  7, 11]
                [ 1,  1,  3,  8, 12]
                [ 1,  1,  4,  9, 13]
                [ 1,  1,  5, 10, 14]
                [ 1,  1,  6, 11, 15]
                [ 1,  1,  7, 12, 16]
                [ 1,  1,  8, 13, 17]
                [ 1,  1,  9, 14, 18]
                [ 1,  1, 10, 15, 19]
                [ 1,  1, 11, 16, 20]
                [ 1,  1, 12, 17, 21]
                *Al variare della dimensione della prediction_window variano anche
                i pacchetti contenuti
            '''
            S_temp = np.reshape(np.linspace(1, L, L, dtype=int), (-1, 1))
            one_temp = np.ones((L, 1), dtype=int)

            for win in [5, 10, 30, 60]:
                if L > win:
                    S_temp_left = np.linspace(2, L - win + 1, L - win, dtype=int)
                    S_temp_left = np.pad(S_temp_left, (win, 0), constant_values=(1, 0))
                    S_temp_left = np.reshape(S_temp_left, (-1, 1))
                    S_temp = np.concatenate((S_temp_left, S_temp), axis=1)
                else:
                    S_temp = np.concatenate((one_temp, S_temp), axis=1)

            # if len(B_temp) != 0:
            # SALVA METADATA

            # salvo le dimensioni dei pacchetti in downstream e le relative occorrenze
            # es. 1384 : 9 (ci sono 9 pacchetti da 1384 bytes)
            unique_D, counts_D = np.unique(data.iloc[keys]['DF_L4_payload_bytes'], return_counts=True)
            unique_U, counts_U = np.unique(data.iloc[keys]['UF_L4_payload_bytes'], return_counts=True)

            MD[B_INDEX] = {
                "KEYS": keys,
                "BF_NUM_PACKETS": len(data.iloc[keys]['L4_payload_bytes']),
                "UP_NUM_PACKETS": len(data.iloc[keys]['UF_L4_payload_bytes']),
                "DW_NUM_PACKETS": len(data.iloc[keys]['DF_L4_payload_bytes']),
                "DW_BYTES_DIST": dict(zip(unique_D, counts_D)),
                "UP_BYTES_DIST": dict(zip(unique_U, counts_U)),
                "BF_DURATION": duration
            }

            # AGGIORNA "MATRICE" PREDIZIONI SALVATA NEL DIZIONARIO
            for i in range(L):
                DATA['L4_PAYLOAD_BYTES_INPUT'].append(X_B_L4B[i])
                DATA['L4_PAYLOAD_BYTES_OUTPUT'].append(Y_B_L4B[i])
                DATA['IAT_INPUT'].append(X_B_IAT[i])
                DATA['IAT_OUTPUT'].append(Y_B_IAT[i])
                DATA['PD_INPUT'].append(X_B_PD[i])
                DATA['PD_OUTPUT'].append(Y_B_PD[i])
                DATA['TWIN_INPUT'].append(X_B_TWIN[i])
                DATA['TWIN_OUTPUT'].append(Y_B_TWIN[i])

            DATA['FLOW_INDEX'].append(B_temp)
            DATA['PACKET_ID'].append(S_temp)

            del L4B_temp, IAT_temp, DIRECTION_temp, TCP_WIN_temp
            del B_temp, S_temp

            del X_B_L4B, Y_B_L4B
            del X_B_IAT, Y_B_IAT
            del X_B_PD, Y_B_PD
            del X_B_TWIN, Y_B_TWIN

            if extract_payload:
                payload = payload.append(compute_payload(keys, data.iloc[keys], save_mode='pickle'),
                                         ignore_index=True)
            if extract_tcp_flags:
                tcp_flag = tcp_flag.append(compute_tcp_flag(keys, data.iloc[keys], INFO['PRED_WIN']),
                                           ignore_index=True)

            B_INDEX += 1
        else:
            NULL_BF.append(keys)
        gc.collect()

    logging.info('Numero di biflussi: %d' % B_INDEX)
    [logging.info('Chiave biflusso scartato: [%d]' % NULL_BF[i]) for i in range(len(NULL_BF))]

    # EXPORT TO PICKLE

    # pickle_out = bz2.BZ2File(filename,"wb")
    pickle_out = lz4.frame.open(filename, "wb")

    # CARICA INFO
    INFO['NULL_BF'] = NULL_BF
    pickle.dump(INFO, pickle_out, pickle.HIGHEST_PROTOCOL)

    # CARICA METADATA
    pickle.dump(MD, pickle_out, pickle.HIGHEST_PROTOCOL)

    # CARICA INDICI BIFLUSSI
    pickle.dump(DATA['FLOW_INDEX'], pickle_out, pickle.HIGHEST_PROTOCOL)
    del DATA['FLOW_INDEX']

    # CARICA INFO INDEX
    pickle.dump(DATA['PACKET_ID'], pickle_out, pickle.HIGHEST_PROTOCOL)
    del DATA['PACKET_ID']

    print("METADATI --> SAVED")

    # CARICA MATRICE DI PREDIZIONE
    pickle.dump(DATA['L4_PAYLOAD_BYTES_INPUT'], pickle_out, pickle.HIGHEST_PROTOCOL)
    del DATA['L4_PAYLOAD_BYTES_INPUT']
    print("L4_PAYLOAD_BYTES_INPUT --> SAVED")

    pickle.dump(DATA['L4_PAYLOAD_BYTES_OUTPUT'], pickle_out, pickle.HIGHEST_PROTOCOL)
    del DATA['L4_PAYLOAD_BYTES_OUTPUT']
    print("L4_PAYLOAD_BYTES_OUTPUT --> SAVED")

    pickle.dump(DATA['IAT_INPUT'], pickle_out, pickle.HIGHEST_PROTOCOL)
    del DATA['IAT_INPUT']
    print("IAT_INPUT --> SAVED")

    pickle.dump(DATA['IAT_OUTPUT'], pickle_out, pickle.HIGHEST_PROTOCOL)
    del DATA['IAT_OUTPUT']
    print("IAT_OUTPUT --> SAVED")

    pickle.dump(DATA['PD_INPUT'], pickle_out, pickle.HIGHEST_PROTOCOL)
    del DATA['PD_INPUT']
    print("PD_INPUT --> SAVED")

    pickle.dump(DATA['PD_OUTPUT'], pickle_out, pickle.HIGHEST_PROTOCOL)
    del DATA['PD_OUTPUT']
    print("PD_OUTPUT --> SAVED")

    pickle.dump(DATA['TWIN_INPUT'], pickle_out, pickle.HIGHEST_PROTOCOL)
    del DATA['TWIN_INPUT']
    print("TWIN_INPUT --> SAVED")

    pickle.dump(DATA['TWIN_OUTPUT'], pickle_out, pickle.HIGHEST_PROTOCOL)
    del DATA['TWIN_OUTPUT']
    print("TWIN_OUTPUT --> SAVED")

    pickle_out.close()

    logging.info('Successfully pickle_dump : DATA, METADATA')
    logging.info('Pickle_protocol : %d' % pickle.HIGHEST_PROTOCOL)

    if extract_payload:
        payload.to_pickle(filename + '_payload.pickle', compression='gzip', protocol=pickle.HIGHEST_PROTOCOL)
        # payload.to_parquet('./'+INFO['APP']+'_payload.parquet', engine='auto', compression='snappy')

        print("PAYLOAD --> SAVED")

        logging.info('Successfully pickle_dump : PAYLOAD')
        logging.info('Pickle_protocol : %d' % pickle.HIGHEST_PROTOCOL)
    if extract_tcp_flags:
        tcp_flag.to_pickle(filename + '_tcp_flags.pickle', compression='gzip', protocol=pickle.HIGHEST_PROTOCOL)

        print("TCP FLAGS --> SAVED")

        logging.info('Successfully pickle_dump : PAYLOAD')
        logging.info('Pickle_protocol : %d' % pickle.HIGHEST_PROTOCOL)
