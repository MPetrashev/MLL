import logging
import numpy as np
logger = logging.getLogger(__file__)

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
from dl.TorchApproximator import TorchApproximator
from dl.TFApproximator import TFApproximator
# from dl.PlotlyCallback import plotly_callback


# todo remove this method
def shift_pvs(pvs: np.ndarray, expected_min: float = 100.) -> np.ndarray:
    shift = expected_min - pvs.min()
    return pvs + shift