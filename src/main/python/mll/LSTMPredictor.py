from typing import Union

import pandas as pd
import numpy as np
import tensorflow as tf
# Code below fixes bug: https://github.com/tensorflow/tensorflow/issues/36508
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from utils import lazy_property


class LSTMPredictor:
    def __init__(self, time_series: pd.DataFrame, feature_to_predict: Union[str, int], train_split: int
                 , past_history: int, future_target: int, step: int = 1, batch_size: int = 256,
                 buffer_size: int = 10000, evaluation_interval: int = 200, epochs: int = 10, seed: int = None) -> None:
        super().__init__()
        self.feature_to_predict = time_series.columns.get_loc(feature_to_predict) if isinstance(feature_to_predict, str) else feature_to_predict
        self.time_series = time_series
        self.train_split = train_split
        self.future_target = future_target
        self.past_history = past_history
        self.step = step
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.evaluation_interval = evaluation_interval
        self.epochs = epochs
        self.seed = seed

    @lazy_property
    def dataset(self):
        dataset = self.time_series.values
        train_ds = dataset[:self.train_split]
        return (dataset - train_ds.mean(axis=0)) / train_ds.std(axis=0)

    # todo refactor this method
    @lazy_property
    def multivariate_data(self):
        """
        :param dataset: Timeseries of explanatory variables.
        :param target: Timeseries of explained variable.
        :return: Tuple with:
            - Cube of explanatory variables sets per each step and
            - Set of explained variable sets per each step.
        """
        dataset = self.dataset
        target_size = self.future_target
        history_size = self.past_history
        target = self.dataset[:, self.feature_to_predict]
        step = self.step
        data = []
        labels = []

        for i in range(history_size, len(dataset) - target_size):
            indices = range(i - history_size, i, step)
            data.append(dataset[indices])
            labels.append(target[i:i + target_size])

        return np.array(data), np.array(labels)

    @lazy_property
    def training_dataset(self):
        x, y = self.multivariate_data
        boundary = self.train_split - self.past_history
        self.x_train_multi, y_train_multi = x[0:boundary], y[0:boundary]
        train_data_multi = tf.data.Dataset.from_tensor_slices((self.x_train_multi, y_train_multi))
        if self.seed:
            tf.random.set_seed(self.seed)
        return train_data_multi.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()

    @lazy_property
    def validation_dataset(self):
        x, y = self.multivariate_data
        x_val_multi, y_val_multi = x[self.train_split:], y[self.train_split:]
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
        validation_dataset = validation_dataset.batch(self.batch_size).repeat()
        return validation_dataset

    @lazy_property
    def multi_step_model(self):
        validation_dataset = self.validation_dataset
        multi_step_model = tf.keras.models.Sequential()
        multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=self.x_train_multi.shape[-2:]))
        multi_step_model.add(tf.keras.layers.LSTM(16))
        multi_step_model.add(tf.keras.layers.Dense(self.future_target))

        multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        self.multi_step_history = multi_step_model.fit(self.training_dataset, epochs=self.epochs,
                                                       steps_per_epoch=self.evaluation_interval,
                                                       validation_data=validation_dataset,
                                                       validation_steps=50)
        return multi_step_model
