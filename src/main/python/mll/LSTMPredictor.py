from typing import Union

import pandas as pd
import numpy as np
import tensorflow as tf

from utils import lazy_property


# todo refactor this method
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


class LSTMPredictor:
    def __init__(self, time_series: pd.DataFrame, feature_to_predict: Union[str, int], train_split: int
                 , past_history: int, future_target: int, step: int = 1, batch_size: int = 256,
                 buffer_size: int = 10000, evaluation_interval: int = 200, epochs: int = 10) -> None:
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

    @lazy_property
    def dataset(self):
        dataset = self.time_series.values
        train_ds = dataset[:self.train_split]
        return (dataset - train_ds.mean(axis=0)) / train_ds.std(axis=0)

    @lazy_property
    def validation_dataset(self):
        self.x_train_multi, y_train_multi = multivariate_data(self.dataset, self.dataset[:, self.feature_to_predict], 0
                            , self.train_split, self.past_history, self.future_target, self.step)
        x_val_multi, y_val_multi = multivariate_data(self.dataset, self.dataset[:, self.feature_to_predict]
                            , self.train_split, None, self.past_history, self.future_target, self.step)
        train_data_multi = tf.data.Dataset.from_tensor_slices((self.x_train_multi, y_train_multi))
        self.train_data_multi = train_data_multi.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
        validation_dataset = validation_dataset.batch(self.batch_size).repeat()
        return validation_dataset

    @lazy_property
    def multi_step_model(self):
        validation_dataset = self.validation_dataset
        multi_step_model = tf.keras.models.Sequential()
        multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=self.x_train_multi.shape[-2:]))
        multi_step_model.add(tf.keras.layers.LSTM(16))
        # multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
        multi_step_model.add(tf.keras.layers.Dense(self.future_target))

        multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        self.multi_step_history = multi_step_model.fit(self.train_data_multi, epochs=self.epochs,
                                                  steps_per_epoch=self.evaluation_interval,
                                                  validation_data=validation_dataset,
                                                  validation_steps=50)
        return multi_step_model
