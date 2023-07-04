import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.autonotebook import tqdm


def create_dataset(dataset, look_back=1, look_forward=1, pad=0, features_last=True, 
                   multi_output=True, right_pad=False,verbose=False
                   ):
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
    #print(np.shape(dataset))
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
        check=False

        for i in range(1, min([look_back, n_pkt - look_forward + 1])):
            #if check:
            #    print('IN 1',bf_index)
            initial_history = list([list(feat[:i]) for feat in bf])
            initial_history = list(
                [[pad[j]] * (look_back - i) + feat + right_pad_list[j] for j, feat in enumerate(initial_history)])
            initial_history = np.array(initial_history).T
            initial_oracle = [feat[i:(i + look_forward)] for feat in bf]
            if not multi_output:
                initial_oracle = [o[-1:] for o in initial_oracle]
            initial_oracle = np.array(initial_oracle).T
            vX.append(initial_history)
            vY.append(initial_oracle)
            dataBF.append(bf_index)
                
        for i in range(n_pkt - look_back - look_forward + 1):
            #if check:
            #    print('IN 2',bf_index)
            history = list([list(feat[i:(i + look_back)]) + right_pad_list[j] for j, feat in enumerate(bf)])
            history = np.array(history).T
            oracle = [feat[(i + look_back):(i + look_back + look_forward)] for feat in bf]
            if not multi_output:
                oracle = [o[-1:] for o in oracle]
            oracle = np.array(oracle).T
            vX.append(history)
            vY.append(oracle)
            dataBF.append(bf_index)
        dataX.extend(vX)
        dataY.extend(vY)
    if not features_last:
        dataX = np.asarray([x.T for x in dataX])
    return np.asarray(dataX), np.asarray(dataY), np.asarray(dataBF)


def GEN_CRNN(core, rnn_units, n_steps, n_fields, n_noise, output_activation=tf.keras.activations.sigmoid,
             name='gen_clstm'):
    """
    Conditional LSTM generator. Predictor.
    :param output_activation:
    :param rnn_units:
    :param n_steps:
    :param n_fields:
    :param n_noise:
    :param name:
    :return:
    """
    RNN = tf.keras.layers.LSTM if core == 'LSTM' else \
        tf.keras.layers.GRU if core == 'GRU' else \
            tf.keras.layers.RNN

    history = tf.keras.layers.Input(shape=(n_steps, n_fields), name='history_g')
    noise = tf.keras.layers.Input(shape=(n_noise,), name='noise')

    rnn = RNN(rnn_units, activation=tf.keras.activations.sigmoid, name='%s_g' % core)(history)
    x = tf.keras.layers.concatenate([rnn, noise], name='salted_embedding')
    x = tf.keras.layers.Dense(n_steps + n_noise, activation=tf.keras.layers.LeakyReLU(),
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros', name='fully_connected')(x)

    fake = tf.keras.layers.Dense(n_fields, activation=output_activation, kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros', name='prediction_fake_g')(x)

    model = tf.keras.models.Model([history, noise], fake, name=name)
    return model


def DIS_CRNN(core, rnn_units, n_steps, n_fields, output_activation=tf.keras.activations.sigmoid, name='dis_clstm'):
    """
    Conditional LSTM discriminator. Classifier.
    :param output_activation:
    :param rnn_units:
    :param n_steps:
    :param n_fields:
    :param name:
    :return:
    """
    RNN = tf.keras.layers.LSTM if core == 'LSTM' else \
        tf.keras.layers.GRU if core == 'GRU' else \
            tf.keras.layers.RNN

    history = tf.keras.layers.Input(shape=(n_steps, n_fields), name='history_d')
    preds = tf.keras.layers.Input(shape=(n_fields,), name='prediction_fake_d')

    rs = tf.keras.layers.Reshape((1, n_fields))(preds)
    x = tf.keras.layers.concatenate([history, rs], axis=1, name='full_sequence')

    rnn = RNN(rnn_units, activation=tf.keras.activations.sigmoid, name='%s_d' % core)(x)

    validity = tf.keras.layers.Dense(1, activation=output_activation, kernel_initializer='glorot_uniform',
                                     bias_initializer='zeros', name='validity')(rnn)

    model = tf.keras.models.Model([history, preds], validity, name=name)
    return model
