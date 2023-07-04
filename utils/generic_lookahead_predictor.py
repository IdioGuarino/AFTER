from copy import deepcopy
import sys

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


class PredictorLookahead:

    def __init__(self, lookahead, n_features, features_name, prediction_type, granularity=None):
        self.lookahead = lookahead
        self.n_features = n_features
        self.features_name = features_name
        self.prediction_type = prediction_type if prediction_type in prediction_type_list else 'recursive'

        # def predictor_mc(self, histories, mt_mod, verbose):
        # if mm is None:
        # raise ImportError('You cannot use markovian predictor because <markovian_models> is not imported')
        # iterator = tqdm(
        #    incr_order_histories + fixed_order_histories) if verbose else incr_order_histories + fixed_order_histories
        # del incr_order_histories, fixed_order_histories
        # cprint('Prediction w/ incremental window...', verbose)
        # Uso un dizionario per uniformare l'output dei predittori che leggerà l'interfaccia
        # preds = {'predicted': np.zeros(shape=(self.lookahead, len(histories)), dtype=int)}
        """Le predizioni sia per le finestre incrementali che per le fisse sono effettuate tramite 
         lo stimatore MAP"""
        """for i, history in enumerate(histories):
            for j in range(self.lookahead):
                mt_mod.set_incr_mode(True) if history[0] == mt_mod.escape_symbol else mt_mod.set_incr_mode(False)
                proba = mt_mod.get_distribution(history)
                max_proba = max(proba)
                predicted = mt_mod.symbols[random.choice([i for i, p in enumerate(proba) if p == max_proba])]
                new_history = np.append(history[1:], predicted)
                history = deepcopy(new_history)
                preds['predicted'][j][i] = predicted
        # cprint('Prediction w/ fixed window...', verbose)
        return preds

    def predictor_hmm(self, hmm_mod, histories, mu_PL, mu_IAT):
        if mm is None:
            raise ImportError('You cannot use markovian predictor because <markovian_models> is not imported')
        seqlen = len(histories)  # numero di osservazioni all'interno della sequenza
        feature_keys = ['PL', 'IAT']
        # Creo un dizionario che contiene per ogni chiave "feature" un array di predizioni lookahead x Nsamples
        predicted_dict = dict((k, v) for (k, v) in zip(feature_keys, [np.zeros(shape=(self.lookahead, seqlen))
                                                                      for _ in range(len(feature_keys))]))
        n_states = hmm_mod.node_count() - 2
        delta_matrix = np.zeros((n_states, seqlen), dtype=np.float64)

        # dalla matrice di transizione estraggo soltanto i valori relativi agli stati non silent (cioe' escludo Start-Stop)
        A = (hmm_mod.dense_transition_matrix())[0:n_states, 0:n_states]
        A = A.transpose()

        for i, history in enumerate(histories):
            for j in range(self.lookahead):
                EM = hmm_mod.predict_proba(history)  # matrice di emissione ottenuta con Forward-Backward

                deltaT = EM[-1, :]  # estraggo l'ultima riga della matrice di emissione

                deltaT = deltaT.transpose()
                p = np.matmul(A, deltaT)
                delta_matrix[:, i] = p
                predicted_dict['PL'][j][i] = np.dot(mu_PL, p)
                predicted_dict['IAT'][j][i] = np.dot(mu_IAT, p)
                history_list = list(history)
                history_list.append([predicted_dict['PL'][j][i], predicted_dict['IAT'][j][i]])
                history = deepcopy(np.array(history_list)[1:])

        return predicted_dict, delta_matrix"""

    def predictor_nn(self, nn_models, histories, fit_next_model=False):
        prediction_steps = self.lookahead[1]
        assert isinstance(fit_next_model, bool), "The value of fit_next_model must be boolean"
        if (self.prediction_type == 'hybrid' or self.prediction_type == 'ouroboros') and fit_next_model:
            prediction_steps = 1

        predicted_dict = dict((k, v) for (k, v) in zip(self.features_name, [np.zeros(shape=(histories.shape[0],
                                                                                            prediction_steps))
                                                                            for _ in range(len(self.features_name))]))

        for i in range(len(nn_models)):
            model = nn_models[i]
            predicted = model.predict(histories)
            for j, key in enumerate(predicted_dict.keys()):
                predicted_dict[key][:, i] = predicted[j][:, 0]

            if self.prediction_type == 'hybrid':
                num_elem = histories.shape[0]
                win_size = histories.shape[1] + 1
                temp_histories = np.zeros((num_elem, win_size, self.n_features), dtype=np.float32)
                for j in range(histories.shape[0]):
                    pred_values = np.zeros(self.n_features, dtype=np.float32)
                    current_pred_matrix = list(histories[j])
                    for k in range(self.n_features):
                        pred_values[k] = predicted[k][j]
                    current_pred_matrix.append(pred_values)
                    temp_histories[j] = np.asarray(current_pred_matrix)
                histories = temp_histories
                del temp_histories, current_pred_matrix, pred_values

            if self.prediction_type == 'recursive' or self.prediction_type == 'ouroboros':
                for j in range(histories.shape[0]):
                    for k in range(self.n_features):
                        histories[j][:-1, k] = histories[j][1:, k]
                        histories[j][-1, k] = predicted[k][j]

        return predicted_dict if not fit_next_model else histories

    def predictor_HOMC(self, homc_models, histories, estimator='map', indexes=None, fit_next_model=False, p_field=None,
                       i_field=None, p_bin_ranges=None, i_bin_ranges=None, i_nbins=None):
        if mm is None:
            raise ImportError('You cannot use markovian predictor because markovian_models_mod is not imported')

        prediction_steps = self.lookahead[1]
        assert isinstance(fit_next_model, bool), "The value of fit_next_model must be boolean"
        if (self.prediction_type == 'hybrid' or self.prediction_type == 'ouroboros') and fit_next_model:
            prediction_steps = 1

        if estimator == 'mmse' and not indexes:
            sys.exit('Parameter indexes must be a list when estimator is mmse')

        if estimator == 'mmse' and (p_bin_ranges is None or i_bin_ranges is None or i_nbins is None or p_field is None
                                    or i_field is None):
            sys.exit('If you want to use mmse estimator, you have to specified p_bin_ranges, i_bin_ranges, i_nbina, '
                     'p_field and i_field')

        predicted = [[] for _ in range(prediction_steps)]
        for i in range(len(homc_models)):
            model = homc_models[i]
            predicted[i] = model.predict_on_sequences(histories) if estimator == 'map' else \
                model.predict_proba_on_sequences(histories)

            if estimator == 'mmse':
                indexes_unique_list = list(set(indexes))
                res_per_bf = []
                offset = 0
                for j in indexes_unique_list:
                    bf_len = indexes.count(j)
                    res_per_bf.append([predicted[i][j] for j in range(offset, offset + bf_len)])
                    offset += bf_len

                results = [mylib.from_soft_bin2_joint(pb, p_bin_ranges, i_bin_ranges, True) for pb in res_per_bf]
                pred_dir = [[1 if res[0][i] > 0 else 0 for i in range(len(res[0]))] for res in results]
                pred_size = [results[i][0] for i in range(len(results))]
                pred_i_time = [results[i][1] for i in range(len(results))]
                df = pd.DataFrame({p_field + '_dir': pred_size, i_field: pred_i_time, 'pred_dir': pred_dir})
                size_binned_value = df.apply(mylib.to_bin_zero_aware, args=(p_bin_ranges, p_field + '_dir', 'pred_dir',
                                                                            True), axis=1)
                i_time_binned = df.apply(mylib.to_bin_zero_aware, args=(i_bin_ranges, i_field, 'pred_dir', False),
                                         axis=1)
                predicted[i] = [val for pb in mm.generate_joint_binning(i_time_binned, size_binned_value, i_nbins)
                                for val in pb]
                del results, pred_dir, pred_size, pred_i_time, df, size_binned_value, i_time_binned, res_per_bf

            if self.prediction_type == 'hybrid':
                for j in range(len(predicted[i])):
                    histories[j].append(predicted[i][j])

            if self.prediction_type == 'recursive' or self.prediction_type == 'ouroboros':
                for j in range(len(predicted[i])):
                    if len(histories[j]) < model.order:
                        histories[j].append(predicted[i][j])
                    else:
                        histories[j][:-1] = histories[j][1:]
                        histories[j][-1] = predicted[i][j]

        return predicted if not fit_next_model else histories
