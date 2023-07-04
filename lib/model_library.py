from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal, RandomUniform
from tensorflow.keras.layers import Concatenate, Add, Reshape, ConvLSTM2D, Conv3D
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout, concatenate
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from transformer import EncoderLayer


class Models:

    def __init__(self, par, prediction_window, in_n_features, features_name=None):

        self.par = par
        self.prediction_window = prediction_window
        self.in_n_features = in_n_features
        self.features_name = features_name

#######################################################################################################################à

    def create_LSTM_model(self):  # units, prediction_window, n_features, bidirectional = False):

        i = Input((self.prediction_window, self.in_n_features), name='Input_pred')

        if self.par.getboolean('bidirectional'):
            lstm = LSTM(int(self.par['units_rnn']), activation=self.par['activation_rnn'], return_sequences=False)
            y = Bidirectional(lstm, name='BI_LSTM')(i)
        else:
            y = LSTM(int(self.par['units_rnn']), activation=self.par['activation_rnn'], return_sequences=False,
                    name='LSTM')(i)
            
        O = []
        for feature_name in self.features_name:
            O.append(Dense(1, activation='sigmoid', name='%s_output' % feature_name)(y))
            

        model = Model(i, O,name='LSTM')

        return model

    ################################################################################################################################################################################################################################################################

    def create_GRU_model(self):  # units, prediction_window, n_features, bidirectional = False):

        i = Input((self.prediction_window, self.in_n_features), name="Input_pred")

        if self.par.getboolean('bidirectional'):
            gru = GRU(int(self.par['units_rnn']), activation=self.par['activation_rnn'], return_sequences=False)
            y = Bidirectional(gru, name='BI_GRU')(i)
        else:
            y = GRU(int(self.par['units_rnn']), activation=self.par['activation_rnn'], return_sequences=False)(i)

        O = []
        for feature_name in self.features_name:
            O.append(Dense(1, activation='sigmoid', name='%s_output' % feature_name)(y))

        model = Model(i, O,name='GRU')

        return model

    ################################################################################################################################################################################################################################################################

    def create_CNN_WANG_model(self):
        i = Input((self.prediction_window, self.in_n_features), name="Input_pred")
        LV1 = Conv1D(filters=int(self.par['filters_lv1']), kernel_size=int(self.par['kernel_size']), strides=1,
                    padding=self.par['padding'], activation=self.par['activation'])(i)
        LV1 = MaxPooling1D(pool_size=int(self.par['pool_size']), strides=None, padding=self.par['padding'])(LV1)
        LV2 = Conv1D(filters=int(self.par['filters_lv2']), kernel_size=int(self.par['kernel_size']), strides=1,
                     padding=self.par['padding'], activation=self.par['activation'])(LV1)
        LV2 = MaxPooling1D(pool_size=int(self.par['pool_size']), strides=None, padding=self.par['padding'])(LV2)
        if self.par.getboolean('extend'):
            if self.par['extender'] == 'LSTM':
                x_net = LSTM(int(self.par['units_rnn']), activation='sigmoid', return_sequences=False)
            else:
                x_net = GRU(int(self.par['units_rnn']), activation='sigmoid', return_sequences=False)

            if self.par.getboolean('bidirectional'):
                x = Bidirectional(x_net)(LV2)
            else:
                x = x_net(LV2)
        else:
            x = Flatten()(LV2)
        y = Dense(int(self.par['units_dense']), activation='relu', name='FC_CNN')(x)
        O = []
        for feature_name in self.features_name:
            O.append(Dense(1, activation='sigmoid', name='%s_output' % feature_name)(y))
        model = Model(i, O,name='CNN')

        return model

    ################################################################################################################################################################################################################################################################

    def create_CNN_RNN_model(self):  # units, self.prediction_window, n_features, rnn='GRU', filters = 32, ks = 4, padding = 'valid', activation = 'relu'):
        i = Input((self.prediction_window, self.in_n_features))

        LV_CNN = Conv1D(filters=int(self.par['filters']), kernel_size=int(self.par['kernel_size']),
                        strides=int(self.par['strides']), padding=self.par['padding'],
                        activation=self.par['activation'])(i)

        if self.par['rnn'] == 'LSTM':
            RNN_1 = LSTM(int(self.par['units_rnn']), activation=self.par['activation'], return_sequences=True,
                         name='LSTM_LV1')
            RNN_2 = LSTM(int(self.par['units_rnn']), activation=self.par['activation'], return_sequences=True,
                         name='LSTM_LV2')
        else:
            RNN_1 = GRU(int(self.par['units_rnn']), activation=self.par['activation'], return_sequences=True,
                        name='GRU_LV1')
            RNN_2 = GRU(int(self.par['units_rnn']), activation=self.par['activation'], return_sequences=True,
                        name='GRU_LV2')

        LV_RNN_LV1 = RNN_1(LV_CNN)
        LV_RNN_LV2 = RNN_2(LV_RNN_LV1)

        y = TimeDistributed(Dense(int(self.par['units_dense']), activation=self.par['activation']))(LV_RNN_LV2)
        y = Flatten()(y)
              
        O = []
        for feature_name in self.features_name:
            O.append(Dense(1, activation='sigmoid', name='%s_output' % feature_name)(y))

        model = Model(i, O,name='CNN-RNN')

        return model

    ################################################################################################################################################################################################################################################################

    def DC_CNN_Block(self, nb_filter, filter_length, dilation, l2_layer_reg):
        def f(input_):
            residual = input_

            layer_out = Conv1D(filters=nb_filter, kernel_size=filter_length,
                               dilation_rate=dilation,
                               activation='linear', padding='causal', use_bias=False,
                               kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                                  seed=42), kernel_regularizer=l2(l2_layer_reg))(input_)

            layer_out = Activation('selu')(layer_out)

            skip_out = Conv1D(1, 1, activation='linear', use_bias=False,
                              kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                                 seed=42), kernel_regularizer=l2(l2_layer_reg))(
                layer_out)

            network_in = Conv1D(1, 1, activation='linear', use_bias=False,
                                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05,
                                                                   seed=42), kernel_regularizer=l2(l2_layer_reg))(
                layer_out)

            network_out = Add()([residual, network_in])

            return network_out, skip_out

        return f

    ################################################################################################################################################################################################################################################################

    def create_SeriesNET(self):  # units, self.prediction_window, n_features, activation = 'relu', extend = False, extended_units = 200):

        i = Input((self.prediction_window, self.in_n_features), name="Input_pred")
        l1a, l1b = self.DC_CNN_Block(32, 2, 1, 0.001)(i)
        l2a, l2b = self.DC_CNN_Block(32, 2, 2, 0.001)(l1a)
        l3a, l3b = self.DC_CNN_Block(32, 2, 4, 0.001)(l2a)
        l4a, l4b = self.DC_CNN_Block(32, 2, 8, 0.001)(l3a)
        l5a, l5b = self.DC_CNN_Block(32, 2, 16, 0.001)(l4a)
        l6a, l6b = self.DC_CNN_Block(32, 2, 32, 0.001)(l5a)
        l6b = Dropout(0.8)(l6b)  # dropout used to limit influence of earlier data
        l7a, l7b = self.DC_CNN_Block(32, 2, 64, 0.001)(l6a)
        l7b = Dropout(0.8)(l7b)  # dropout used to limit influence of earlier data

        l8 = Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])

        l9 = Activation('relu')(l8)

        l21 = Conv1D(1, 1, activation='linear', use_bias=False,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
                     kernel_regularizer=l2(0.001))(l9)
        l21 = Flatten()(l21)

        if self.par.getboolean('extend'):

            l21 = Dense(int(self.par['units_dense']), activation=self.par['activation'], name='FC_CNN')(l21)

            lstm = LSTM(int(self.par['units_rnn']), return_sequences=True, name='LSTM')(i)
            lstm = LSTM(int(self.par['units_rnn']), return_sequences=False, name='LSTM_2')(lstm)
            lstm = Dense(int(self.par['units_dense']), activation='relu', name='FC_LSTM')(lstm)

            y = Add()([l21, lstm])
            y = Dense(int(self.par['units_merge']), activation='relu')(y)
        else:
            y = Dense(int(self.par['units_dense']), activation=self.par['activation'], name='FC_CNN')(l21)

        O = []
        for feature_name in self.features_name:
            O.append(Dense(1, activation='sigmoid', name='%s_output' % feature_name)(y))

        model = Model(i, O,name='SeriesNET')

        return model

    ################################################################################################################################################################################################################################################################

    def GlobalTemporalConvolution(self, nb_filter, ks, units, act='relu', act_FF='relu', pad='same'):
        def f(input_):
            GTC = Conv1D(filters=nb_filter, kernel_size=ks, padding=pad, activation=act)(input_)
            print('GTC shape', GTC.shape)

            GTC = Dropout(0.1)(GTC)

            GTC = Dense(nb_filter)(GTC)
            print('GTC shape', GTC.shape)

            enc_out, att_out = EncoderLayer(30, 30, 3, dropout=0.1)(GTC)

            for i in range(3):
                enc_out, att_out = EncoderLayer(30, 30, 3, dropout=0.1)(enc_out)

            network_out = Dense(units)(enc_out)

            return network_out

        return f
      
    def create_DSANet(self):  # units, self.prediction_window, n_features):
        
        i = Input((self.prediction_window, self.in_n_features), name="Input_pred")
        i_conv = Reshape((i.shape[2], i.shape[1]))(i)

        # GLOBAL TEMPORAL CONVOLUTION
        # GTC = self.GlobalTemporalConvolution(int(self.par['GTC_filter']), int(self.par['GTC_kernel_size']),
        #                                      int(self.par['GTC_units']), act='relu', act_FF='relu', pad='same')(i_conv)

        # LOCAL TEMPORAL CONVOLUTION
        # LTC = self.LocalTemporalConvolution(int(self.par['LTC_filter']), int(self.par['LTC_kernel_size']),
        #                                     int(self.par['LTC_pool_size']), int(self.par['LTC_units']), act='relu',
        #                                     act_FF='relu', pad='same')(i_conv)

        # LTC = LocalTemporalConvolution(self.ltc_filter, self.ltc_kernel_size,
        #                                     self.ltc_pool_size, self.ltc_units, act='relu',
        #                                     act_FF='relu', pad='same')(i_conv)
        
        class LocalTemporalConvolution(tf.keras.layers.Layer):
            def __init__(self, ltc_filters, ltc_kernel_size, ltc_pool_size, units, act='relu', pad='same'):
                super(LocalTemporalConvolution, self).__init__()
                self.ltc_filters=ltc_filters
                self.ltc_kernel_size=ltc_kernel_size
                self.ltc_pool_size=ltc_pool_size
                self.units = units
                self.act=act
                self.pad=pad
                
            def get_config(self):
                config = super().get_config().copy()
                config.update({
                    'ltc_filters': self.ltc_filters,
                    'ltc_kernel_size': self.ltc_kernel_size,
                    'ltc_pool_size':self.ltc_pool_size,
                    'units':self.units,
                    'act':self.act,
                    'pad':self.pad
                })
                return config

            def call(self, input):
                LTC = Conv1D(filters=self.ltc_filters, kernel_size=self.ltc_kernel_size, padding=self.pad, activation=self.act)(input)
                LTC = MaxPooling1D(pool_size=self.ltc_pool_size, strides=1, padding=self.pad)(LTC)
                #print('LTC shape', LTC.shape)

                LTC = Dropout(0.1)(LTC)

                LTC = Dense(self.ltc_filters)(LTC)
                #print('LTC shape', LTC.shape)

                enc_out, att_out = EncoderLayer(7, 4, 3, dropout=0.1)(LTC)

                for i in range(3):
                    enc_out, att_out = EncoderLayer(7, 4, 3, dropout=0.1)(enc_out)

                network_out = Dense(self.units)(enc_out)

                return network_out
            
        class GlobalTemporalConvolution(tf.keras.layers.Layer):
            def __init__(self, gtc_filters, gtc_kernel_size, units, act='relu', pad='same'):
                super(GlobalTemporalConvolution, self).__init__()
                self.gtc_filters=gtc_filters
                self.gtc_kernel_size=gtc_kernel_size
                self.units = units
                self.act=act
                self.pad=pad
                
            def get_config(self):
                config = super().get_config().copy()
                config.update({
                    'gtc_filters': self.gtc_filters,
                    'gtc_kernel_size': self.gtc_kernel_size,
                    'units':self.units,
                    'act':self.act,
                    'pad':self.pad
                })
                return config

            def call(self, input):
                GTC = Conv1D( filters=self.gtc_filters, kernel_size=self.gtc_kernel_size, padding=self.pad, activation=self.act )(input)
                print('GTC shape', GTC.shape)

                GTC = Dropout(0.1)(GTC)

                GTC = Dense(self.gtc_filters)(GTC)
                print('GTC shape', GTC.shape)

                enc_out, att_out = EncoderLayer(30, 30, 3, dropout=0.1)(GTC)

                for i in range(3):
                    enc_out, att_out = EncoderLayer(30, 30, 3, dropout=0.1)(enc_out)

                network_out = Dense(self.units)(enc_out)

                return network_out

        class AR_component(tf.keras.layers.Layer):
            def __init__(self, units):
                super(AR_component, self).__init__()
                self.units = units

            def get_config(self):
                config = super().get_config().copy()
                config.update({
                    'units': self.units
                })
                return config

            def call(self, input):
                AR = Dense(self.units, activation='relu')(input)
                return AR   
        
        GTC = GlobalTemporalConvolution(int(self.par['GTC_filter']), int(self.par['GTC_kernel_size']),
                                               int(self.par['GTC_units']), act='relu', pad='same')(i_conv)
        
        
        LTC = LocalTemporalConvolution(int(self.par['LTC_filter']), int(self.par['LTC_kernel_size']),
                                            int(self.par['LTC_pool_size']), int(self.par['LTC_units']), 
                                            act='relu', pad='same')(i_conv)
        

        # MERGE LOCAL AND GLOBAL CONVOLUTION
        TC = Concatenate(axis=-1)([GTC, LTC])
        #TC = LTC
        TC = Flatten()(TC)
        # TC = Dense(int(self.par['units']), activation='relu')(TC)
        TC = Dense(int(self.par['units']), activation='relu')(TC)
        # AR
        # AR = self.AR_component(int(self.par['units']))(i_conv)
        AR = AR_component(int(self.par['units']))(i_conv)
        
        # SUM CNN and AR CONTRIBUTE
        out = Add()([TC, AR])
        out = Flatten()(out)
        # PREDICTION
        O = []
        for feature_name in self.features_name:
            O.append(Dense(1, activation='sigmoid', name='%s_output' % feature_name)(out))

        model = Model(i, O,name='DSANet')

        return model

    ################################################################################################################################################################################################################################################################

    def create_STN(self):

        i = Input((self.prediction_window, 1, 1, self.in_n_features))
        # -------------First level-------------------------------------------------------------
        kernel_size_lv1 = eval(self.par['kernel_size_lv1'])
        x_cnn_1 = Conv3D(filters=int(self.par['filters_lv11']), kernel_size=kernel_size_lv1,
                         padding=self.par['padding'], activation='relu', use_bias=True)(i)
        x_cnn_1 = Conv3D(filters=int(self.par['filters_lv12']), kernel_size=kernel_size_lv1,
                         padding=self.par['padding'], activation='relu', use_bias=True)(x_cnn_1)
        x_cnn_1 = Conv3D(filters=int(self.par['filters_lv13']), kernel_size=kernel_size_lv1,
                         padding=self.par['padding'], activation='relu', use_bias=True)(x_cnn_1)

        kernel_size_convlstm_lv1 = tuple([kernel_size_lv1[0], kernel_size_lv1[1]])
        x_conv_lstm_1 = ConvLSTM2D(filters=int(self.par['filters_lv13']), kernel_size=kernel_size_convlstm_lv1,
                                   padding='same',
                                   return_sequences=True, use_bias=True)(i)
        fusion_1 = Add()([x_cnn_1, x_conv_lstm_1])

        # -------------Second level-------------------------------------------------------------
        kernel_size_lv2 = eval(self.par['kernel_size_lv2'])
        x_cnn_2 = Conv3D(filters=int(self.par['filters_lv21']), kernel_size=kernel_size_lv2,
                         padding=self.par['padding'],
                         activation='relu',
                         use_bias=True)(fusion_1)
        x_cnn_2 = Conv3D(filters=int(self.par['filters_lv22']), kernel_size=kernel_size_lv2,
                         padding=self.par['padding'], activation='relu',
                         use_bias=True)(x_cnn_2)
        x_cnn_2 = Conv3D(filters=int(self.par['filters_lv23']), kernel_size=kernel_size_lv2,
                         padding=self.par['padding'], activation='relu',
                         use_bias=True)(x_cnn_2)

        kernel_size_convlstm_lv2 = tuple([kernel_size_lv2[0], kernel_size_lv2[1]])
        x_conv_lstm_2 = ConvLSTM2D(filters=int(self.par['filters_lv23']), kernel_size=kernel_size_convlstm_lv2,
                                   return_sequences=True, padding=self.par['padding'], use_bias=True)(fusion_1)
        fusion_2 = Add()([x_cnn_2, x_conv_lstm_2])

        # -------------FC layer & output-------------------------------------------------------------
        to_fc = Flatten()(fusion_2)
        y = Dense(self.par['units_dense'], activation=self.par['activation'], name='FC_STN')(to_fc)
        O = []
        for feature_name in self.features_name:
            O.append(Dense(1, activation='sigmoid', name='%s_output' % feature_name)(y))

        model = Model(i, O)
        return model

    ################################################################################################################################################################################################################################################################

    def model_plot_arch(model, model_name, summary='False'):

        print('Saving model structure..')
        plot_model(model, './' + model_name + '.eps')

        if summary == 'True':
            with open('modelsummary.txt', 'w') as f:
                with redirect_stdout(f):
                    model.summary()

        print('DONE')


################################################################################################################################################################################################################################################################

scaler_inf_range = {
    'MinMax': 0,
    'QuantileTransformer': 0,
    'RobustScaler': -2
}

scaler_sup_range = {
    'MinMax': 1,
    'QuantileTransformer': 1,
    'RobustScaler': 3
}


def transform_data(x_train=None, x_test=None, y_train=None, y_test=None, n_features=3, prediction_window=30,
                   scaler_type='MinMax', multi_scale=False, multi_scale_features=0, scaler=None, scaler_2=None):

    assert x_train is not None and y_train is not None or x_test is not None and y_test is not None, 'Train and test data should not be None'

    # reshape per input allo scaler
    if x_train is not None:
        x_train = np.reshape(x_train, (-1, n_features))
    if x_test is not None:
        x_test = np.reshape(x_test, (-1, n_features))

    new_scaler = False
    if scaler is None and scaler_2 is None:
        if scaler_type == 'MinMax':
            scaler = MinMaxScaler()
            scaler_2 = QuantileTransformer()
        elif scaler_type == 'QuantileTransformer':
            scaler = QuantileTransformer()
            scaler_2 = MinMaxScaler()
        new_scaler = True
        # else:
        #    scaler_type = RobustScaler()

    # SCALING (fit su X e transform su X e Y)

    if multi_scale and multi_scale_features != 0:
        print('MULTI SCALING MODE')
        w = n_features - multi_scale_features
        if new_scaler:
            scaler.fit(x_train[:, 0:w])
            scaler_2.fit(x_train[:, w:])

        # transform X e Y - train e test
            x_train[:, 0:w] = scaler.transform(x_train[:, 0:w])
            y_train[:, 0:w] = scaler.transform(y_train[:, 0:w])
            x_train[:, w:] = scaler_2.transform(x_train[:, w:])
            y_train[:, w:] = scaler_2.transform(y_train[:, w:])
        if x_test is not None and y_test is not None:
            x_test[:, 0:w] = scaler.transform(x_test[:, 0:w])
            y_test[:, 0:w] = scaler.transform(y_test[:, 0:w])
            x_test[:, w:] = scaler_2.transform(x_test[:, w:])
            y_test[:, w:] = scaler_2.transform(y_test[:, w:])
    else:
        if new_scaler:
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            y_train = scaler.transform(y_train)
        if x_test is not None and y_test is not None:
            x_test = scaler.transform(x_test)
            y_test = scaler.transform(y_test)

    # Check MinMax Scaler per valori fuori [0,1] - train e test
    if x_train is not None and y_train is not None:
        x_train = np.where(x_train < scaler_inf_range[scaler_type], scaler_inf_range[scaler_type],
                           (np.where(x_train > scaler_sup_range[scaler_type], scaler_sup_range[scaler_type], x_train)))
        y_train = np.where(y_train < scaler_inf_range[scaler_type], scaler_inf_range[scaler_type],
                           (np.where(y_train > scaler_sup_range[scaler_type], scaler_sup_range[scaler_type], y_train)))
    if x_test is not None and y_test is not None:
        x_test = np.where(x_test < scaler_inf_range[scaler_type], scaler_inf_range[scaler_type],
                          (np.where(x_test > scaler_sup_range[scaler_type], scaler_sup_range[scaler_type], x_test)))

        y_test = np.where(y_test < scaler_inf_range[scaler_type], scaler_inf_range[scaler_type],
                          (np.where(y_test > scaler_sup_range[scaler_type], scaler_sup_range[scaler_type], y_test)))

    # reshape per riportare la forma originaria pre-scaling
    if x_train is not None:
        x_train = np.reshape(x_train, (-1, prediction_window, n_features))
    if x_test is not None:
        x_test = np.reshape(x_test, (-1, prediction_window, n_features))

    if x_train is not None and y_train is not None:
        if x_test is not None and y_test is not None:
            if multi_scale:
                return x_train, y_train, x_test, y_test, scaler, scaler_2
            else:
                return x_train, y_train, x_test, y_test, scaler
        if multi_scale:
            return x_train, y_train, scaler, scaler_2
        else:
            return x_train, y_train, scaler
    else:
        return x_test, y_test

######################################################################################################################################################################################################################

def transform_data_v2(x_train=None, x_test=None, y_train=None, y_test=None, n_features=3, prediction_window=30,
                   scaler_type='MinMax', multi_scale=False, multi_scale_features=0, scaler=None, scaler_2=None):

    assert x_train is not None and y_train is not None or x_test is not None and y_test is not None, 'Train and test data should not be None'

    # reshape per input allo scaler
    if x_train is not None:
        x_train = np.reshape(x_train, (-1, n_features))
    if x_test is not None:
        x_test = np.reshape(x_test, (-1, n_features))

    new_scaler = False
    if scaler is None and scaler_2 is None:
        if scaler_type == 'MinMax':
            scaler = MinMaxScaler()
            scaler_2 = QuantileTransformer()
        elif scaler_type == 'QuantileTransformer':
            scaler = QuantileTransformer()
            scaler_2 = MinMaxScaler()
        new_scaler = True
        # else:
        #    scaler_type = RobustScaler()

    # SCALING (fit su X e transform su X e Y)

    if multi_scale and multi_scale_features != 0:
        print('MULTI SCALING MODE')
        w = n_features - multi_scale_features
        if new_scaler:
            scaler.fit(x_train[:, 0:w])
            scaler_2.fit(x_train[:, w:])

        # transform X e Y - train e test
            x_train[:, 0:w] = scaler.transform(x_train[:, 0:w])
            y_train[:, 0:w] = scaler.transform(y_train[:, 0:w])
            x_train[:, w:] = scaler_2.transform(x_train[:, w:])
            y_train[:, w:] = scaler_2.transform(y_train[:, w:])
        if x_test is not None and y_test is not None:
            x_test[:, 0:w] = scaler.transform(x_test[:, 0:w])
            y_test[:, 0:w] = scaler.transform(y_test[:, 0:w])
            x_test[:, w:] = scaler_2.transform(x_test[:, w:])
            y_test[:, w:] = scaler_2.transform(y_test[:, w:])
    else:
        if new_scaler:
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            y_train = scaler.transform(y_train)
        if x_test is not None and y_test is not None:
            x_test = scaler.transform(x_test)
            y_test = scaler.transform(y_test)

    # Check MinMax Scaler per valori fuori [0,1] - train e test
    if x_train is not None and y_train is not None:
        x_train = np.where(x_train < scaler_inf_range[scaler_type], scaler_inf_range[scaler_type],
                           (np.where(x_train > scaler_sup_range[scaler_type], scaler_sup_range[scaler_type], x_train)))
        y_train = np.where(y_train < scaler_inf_range[scaler_type], scaler_inf_range[scaler_type],
                           (np.where(y_train > scaler_sup_range[scaler_type], scaler_sup_range[scaler_type], y_train)))
    if x_test is not None and y_test is not None:
        x_test = np.where(x_test < scaler_inf_range[scaler_type], scaler_inf_range[scaler_type],
                          (np.where(x_test > scaler_sup_range[scaler_type], scaler_sup_range[scaler_type], x_test)))

        y_test = np.where(y_test < scaler_inf_range[scaler_type], scaler_inf_range[scaler_type],
                          (np.where(y_test > scaler_sup_range[scaler_type], scaler_sup_range[scaler_type], y_test)))

    # reshape per riportare la forma originaria pre-scaling
    if x_train is not None:
        x_train = np.reshape(x_train, (-1, prediction_window, n_features))
    if x_test is not None:
        x_test = np.reshape(x_test, (-1, prediction_window, n_features))

    if x_train is not None and y_train is not None:
        if x_test is not None and y_test is not None:
            if multi_scale:
                return x_train, y_train, x_test, y_test, scaler, scaler_2
            else:
                return x_train, y_train, x_test, y_test, scaler
        if multi_scale:
            return x_train, y_train, scaler, scaler_2
        else:
            return x_train, y_train, scaler
    else:
        return x_test, y_test