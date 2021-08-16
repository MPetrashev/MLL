import logging
logger = logging.getLogger(__file__)

from utils import split_on_condition, swap_rows

import numpy as np
import pandas as pd

has_gpu = True

if has_gpu:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info('GPU is initialized')
else:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1' # switch off GPU https://datascience.stackexchange.com/questions/58845/how-to-disable-gpu-with-tensorflow
    logger.info('There is no GPU on your machine')

from dl.TrainingData import TrainingData
from dl.NormalizedData import NormalizedData
from dl.TorchApproximator import TorchApproximator
from dl.TFApproximator import TFApproximator
# from dl.PlotlyCallback import plotly_callback


# todo remove this method
def shift_pvs(pvs: np.ndarray, expected_min: float = 100.) -> np.ndarray:
    shift = expected_min - pvs.min()
    return pvs + shift


def move_extremes_to_train(df: pd.DataFrame, n_train: int) -> pd.DataFrame:
    """
    Moves all rows with min or max values by either column into training set and scales all dataset into a [0,1] range.
    :return:
    """
    candidates = sorted({df[c].argmin() for c in df.columns}.union({df[c].argmax() for c in df.columns}))
    inside, rows_to_swap = split_on_condition(candidates, lambda row: row < n_train)
    step = int(n_train / (len(rows_to_swap) + 1))
    for i, row2 in enumerate(rows_to_swap):
        row1 = next(j for j in range((i + 1) * step, n_train) if j not in inside)
        swap_rows(df, row1, row2)
    return df


