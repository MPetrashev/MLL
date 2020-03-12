"""
Example: https://www.tensorflow.org/tutorials/structured_data/time_series
"""
from utils import lazy_property
import tensorflow as tf
import numpy as np
import pandas as pd
import os

# todo: https://www.tensorflow.org/tutorials/structured_data/time_series
#  You could also use a tf.keras.utils.normalize method that rescales the values into a range of [0,1].


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    """
    !The mean and standard deviation should only be computed using the training data.!
    :param dataset:
    :param start_index:
    :param end_index:
    :param history_size:
    :param target_size:
    :return:
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


class Tutorial:

    @lazy_property
    def df(self):
        zip_path = tf.keras.utils.get_file(
            origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
            fname='jena_climate_2009_2016.csv.zip',
            extract=True)
        csv_path, _ = os.path.splitext(zip_path)
        return pd.read_csv(csv_path)

    @lazy_property
    def simple_lstm_model(self):
        tf.random.set_seed(13)
        TRAIN_SPLIT = 300000
        univariate_past_history = 20
        univariate_future_target = 0
        uni_data = self.df['T (degC)']
        uni_data.index = self.df['Date Time']

        uni_data = uni_data.values
        uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
        uni_train_std = uni_data[:TRAIN_SPLIT].std()
        uni_data = (uni_data - uni_train_mean) / uni_train_std
        x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                                   univariate_past_history,
                                                   univariate_future_target)
        x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                               univariate_past_history,
                                               univariate_future_target)
        EVALUATION_INTERVAL = 200
        EPOCHS = 10
        simple_lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
            tf.keras.layers.Dense(1)
        ])

        simple_lstm_model.compile(optimizer='adam', loss='mae')

        BATCH_SIZE = 256
        BUFFER_SIZE = 10000

        train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
        train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        self.val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
        self.val_univariate = self.val_univariate.batch(BATCH_SIZE).repeat()

        simple_lstm_model.fit(train_univariate, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL,
                              validation_data=self.val_univariate,
                              validation_steps=50)
        return simple_lstm_model

    def predict(self, index=1):
        model = self.simple_lstm_model
        return [model.predict(x) for x, _ in self.val_univariate.take(index)]

