import os
import traceback
from typing import List, Union, Callable

import pandas as pd
import numpy as np
from utils.lazy_property import lazy_property
from utils.Timer import Timer
import datetime as dt

excel_t0_date = dt.datetime(1899, 12, 30)  # Note, not 31st Dec but 30th!


def xl_date(date, format_str='%d/%m/%Y') -> int:
    """
    :param date:
    :param format_str:
    :return: the excel date representation of a string date. For example: xl_date('14/01/2019') returns 43479
    """
    date = dt.datetime.strptime(date, format_str)
    delta = date - excel_t0_date
    return int(delta.days)


def nb_logging_init():
    """
    Does initialize logging framework for Jupyter Notebook (StreamHandler with output to cell output).
    :return:
    """
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.addHandler(logging.StreamHandler())
    return logger


def init_env(work_dir:str) -> None:
    logger = nb_logging_init()
    global_vars = globals()['__builtins__']

    global_vars['logger'] = logger

    global_vars['os'] = os

    import json
    global_vars['json'] = json

    from utils import Timer
    global_vars['Timer'] = Timer

    import utils
    global_vars['utils'] = utils

    global_vars['pd'] = pd

    import numpy as np
    global_vars['np'] = np

    import plotly.offline as py
    py.init_notebook_mode(connected=True) #todo sometimes doesn't work by some reason. Needs explicit call.
    global_vars['py'] = py
    print('Work dir: {}, process id: {}'.format(work_dir, os.getpid()))


def stack_trace_as_string(e: Exception) -> str:
    """
    :param e:
    :return: Stack trace of exception as a string.
    """
    stack = traceback.extract_stack()[:-3] + traceback.extract_tb(e.__traceback__)  # add limit=??
    pretty = traceback.format_list(stack)
    return ''.join(pretty) + '\n  {} {}'.format(e.__class__, e)


def matrix_to_df(data: List[List[float]], columns=None, pattern='Scenario {}', transposed=False, index_name: str =None):
    """
    Converts epe_profile to pd.DataFrame
    :param data:
    :param columns:
    :param pattern:
    :param transposed:
    :return:
    """
    if transposed:
        data = list(map(list, zip(*data)))
    if not columns:
        columns = [pattern.format(i + 1) for i in range(len(data[0]))]
    result = pd.DataFrame(data, columns=columns)
    if index_name:
        result.index.name = index_name
    return result


def as_ndarray(dataset: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    :param dataset:
    :return: Converts, if necessary, pd.DataFrame or pd.Series into np.ndarray
    """
    return dataset.to_numpy() if isinstance(dataset, (pd.DataFrame,pd.Series)) else dataset


def vrange(start, stop=None, step=1, extra_info: Callable[[int], str] = None):
    """
    Verbose version of range() generator which does output % of completion on each iteration
    :param start:
    :param stop:
    :param step:
    :return:
    """
    if stop is None:
        stop = start
        start = 0
    N = (stop - start) / step
    for idx in range(start, stop, step):
        yield idx
        completed = (idx+1) * 100 / N
        msg = '' if extra_info is None else '. ' + extra_info(idx)
        print(f'\r{completed:5.2f}% completed{msg}', end='', flush=True)


def venumerate(sequence, start=0):
    """
    Verbose version of enumerate() generator which does output % of completion on each iteration
    :param sequence:
    :param start:
    :return:
    """
    N = len(sequence)
    print(f'Maxim:{N}', flush=True)
    for e in enumerate(sequence, start):
        yield e
        completed = int( (e+1) * 100 / N )
        print(f'{completed:2d}% completed', end='\r', flush=True)


def less_than_1pc_exceeds_1pc_diff(y, y_prediction):
    y = y.squeeze()
    y_prediction = y_prediction.detach().cpu().numpy().squeeze()
    arr = np.isclose(y, y_prediction, rtol=0.001) # 10bps remove y.squeeze() and reuse original array
    n = np.count_nonzero(arr)
    return n / y.shape[0] > 0.99

# def less_than_1pc_exceeds_1pc_diff(y, y_prediction):
#     y = y.squeeze() + 100.
#     y_prediction = y_prediction.detach().cpu().numpy().squeeze() + 100.
#     arr = np.isclose(y, y_prediction, rtol=0.001) # 10bps remove y.squeeze() and reuse original array
#     n = np.count_nonzero(arr)
#     return n / y.shape[0] > 0.99
