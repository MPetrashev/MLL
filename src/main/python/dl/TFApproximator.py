from typing import Tuple

from tensorflow import keras
import tensorflow as tf
import pandas as pd

from utils import as_ndarray

# to fix tensorflow.python.framework.errors_impl.InternalError:  Blas GEMM launch failed see
# https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class TFApproximator:
    """
    Runs Deep Learning training against 100,000 samples to approximate N-dimensional (possibly with a 't' dimension and
    trade attribute parameters) PV function against PV values.
    """
    def __init__(self, seed: int = 314) -> None:
        tf.random.set_seed(seed)

    # todo swap test and validation names
    def train(self, states, pvs, n_epochs=6000, pct_test=0.2  # Portion for test set
              , pct_validation=0.1  # Portion for validation set
              , n_hidden: int = 1500, n_layers: int = 4, lr=0.01, verbose=0) -> Tuple[tf.keras.Model, pd.DataFrame]:
        """
        :param states:
        :param pvs:
        :param n_epochs:
        :param pct_test:
        :param pct_validation:
        :param n_hidden:
        :param n_layers:
        :param lr:
        :param verbose:
        :return: trained model and history DataFrame[loss,accuracy,val_loss,val_accuracy]
        """
        pvs, states = as_ndarray(pvs), as_ndarray(states)
        n_rows, n_features = states.shape
        split_idx = int(round(n_rows * (1. - pct_validation)))
        validation_split = pct_test / (1. - pct_validation)

        Y_train, Y_test = pvs[:split_idx], pvs[split_idx:]
        X_train, X_test = states[:split_idx, :], states[split_idx:, :]

        n_features = states.shape[1]       # number of columns
        layers = [keras.layers.Dense(n_hidden, input_shape=(n_features,), activation='relu')] \
                 + [keras.layers.Dense(n_hidden, activation='relu') for i in range(n_layers)] \
                 + [keras.layers.Dense(1)]  # + [keras.layers.Dense(1, activation='softmax')]
        model = tf.keras.models.Sequential(layers)
        batch_size = round(split_idx*(1-validation_split))
        # todo https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6
        # Note that when you wrap your model with tf.function(), you cannot use several model functions like model.compile() and model.fit() because they already try to build a graph automatically. But we will cover those examples in a different and more advanced level post of this series.

        # run_eagerly explained here: https://www.tensorflow.org/guide/intro_to_graphs. If it is False, the graph 1st
        # time would be optimized. Otherwise python operations would be called again and again
        model.compile(run_eagerly=False, optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mse', metrics=['accuracy'])
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epochs, validation_split=validation_split
                            , verbose=verbose)
        return model, pd.DataFrame.from_dict(history.history)
