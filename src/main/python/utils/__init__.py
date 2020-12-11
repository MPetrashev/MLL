import os
import sys
import traceback
from typing import List

import pandas as pd
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
    logging.getLogger("caf").setLevel(logging.WARNING)

    logger.addHandler(logging.StreamHandler())
    return logger


def init_env(work_dir:str) -> None:
    logger = nb_logging_init()
    from epe.CAFInvoker import caf_configure
    caf_configure()
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
