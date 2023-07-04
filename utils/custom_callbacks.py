import keras
import time

#Prova2

class TimeEpochs(keras.callbacks.Callback):
    """Callback used to calculate per-epoch time"""

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        logs['time'] = (time.time() - self.epoch_time_start)
