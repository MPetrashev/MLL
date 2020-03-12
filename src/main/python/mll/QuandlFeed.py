from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict
import logging

import pandas as pd
import quandl
quandl.ApiConfig.api_key = "-GhfHUAdug3HZ6zuAhzx"
logger = logging.getLogger('mll.quandl')


def get_data(datasets: Dict[str,str], data_column: str, max_n: int = 5, **kwargs) -> Dict[str,pd.DataFrame]:
    """
    :param datasets: Dict of datasets where keys are quandl dataset id and values are name of column on which map data_column.
    :param data_column:
    :param max_n: Max thread pool executor size
    :return: List of dataframes with data loaded from quandl.com for each datasets element.
    """
    def job(dataset: str, column_name: str) -> pd.DataFrame:
        logger.info('Loading %s dataset...', dataset)
        df = quandl.get(dataset, **kwargs)
        return df.rename(columns={data_column: column_name})[[column_name]]

    with ThreadPoolExecutor(max_workers=min(len(datasets), max_n)) as e:
        futures = {dataset : e.submit(job, dataset, column_name) for dataset, column_name in datasets.items()}
    result = {k: f.result() for k, f in futures.items()}
    df = None
    for k in datasets.keys():
        if df is None:
            df = result[k]
        else:
            df = df.join(result[k], how='outer')
    return df

# def join_to_df(pds: Dict[str, pd.DataFrame], columns_mapper: Callable[[str], Dict[str, str]]) -> pd.DataFrame:
#     """
#     :param pds:
#     :param columns_mapper:
#     :return: Returns joined DataFrame
#     """
#     def columns(dataset):
#         columns = columns_mapper(dataset)
#         return list(columns.keys()), columns
#
#     result = None
#     for k, df in pds.items():
#         cols, mapping = columns(k)
#         if result is None:
#             result = df[cols].rename(mapping)
#         else:
#             result = result.join(df[cols].rename(mapping), how='outer')
#
#     return result
