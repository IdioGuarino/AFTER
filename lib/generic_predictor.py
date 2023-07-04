from copy import deepcopy
import sys

from pkg_resources import empty_provider

try:
    from lib import markovian_models_mod as mm
except ImportError:
    mm = None
try:
    from lib import my_library as mylib
except ImportError:
    mylib = None
import numpy as np
import pandas as pd
import random

prediction_type_list = ['direct', 'recursive', 'ouroboros', 'hybrid']


class Predictor:

    def __init__(self, out_n_features, features_name):
        self.out_n_features = out_n_features
        self.features_name = features_name
        
    def predictor_nn(self, model, histories, fit_next_model=False):
        
        prediction_steps = 1
            
        predicted_dict = dict((k, v) for (k, v) in zip(self.features_name, 
                                                    [np.zeros(shape=(histories.shape[0],prediction_steps)) 
                                                        for _ in range(self.out_n_features)]))
        predicted_dict={}
        for k in self.features_name:
            predicted_dict[k]=np.zeros(shape=(histories.shape[0],1))
                
        predicted = model.predict(histories, verbose=2)
            
        for j, key in enumerate(predicted_dict.keys()):
            if self.out_n_features>1:
                predicted_dict[key][:, 0] = predicted[j][:, 0]
            else:
                predicted_dict[key][:, 0] = predicted[:, 0]
        return predicted_dict if not fit_next_model else histories
