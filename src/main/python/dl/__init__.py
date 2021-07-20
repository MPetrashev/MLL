import sys
import logging
logger = logging.getLogger(__file__)

has_gpu = False

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

from dl.TorchApproximator import TorchApproximator
# from dl.TFApproximator import TFApproximator
# from dl.PlotlyCallback import plotly_callback


def log_epoch(epoch, logs) -> None:
    """
    Callback function which does log epoch's losses ad hoc. May be passes to model.fit method like it:
    callbacks=[LambdaCallback(on_epoch_end=log_epoch)]
    :param epoc: 
    :param logs: 
    :return: 
    """
    if not has_gpu or epoch % 10 == 0:
        logger.info('Epoch: %s,\t Loss: %.3f', epoch, logs['loss'])


class LogEpoch(object):
    def __init__(self):
        self.best_loss = sys.float_info.max
        self.best_epoch = -1

    def __call__(self, epoch, logs):
        loss = logs['loss']
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
        logger.info('Epoch: %s,\t Best loss: %.3f\t Loss: %.3f', self.best_loss, epoch, loss)
