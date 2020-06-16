import os

import sys
import matplotlib.pyplot as plt
import numpy as np


def init():
    global_vars = globals()['__builtins__']
    sys.path.extend([os.path.abspath('../python')])

    from mll import nb_logging_init
    nb_logging_init()


def multi_step_plot(history, true_future, prediction, step):
    def create_time_steps(length):
        return list(range(-length, 0))

    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out) / step, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out) / step, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()