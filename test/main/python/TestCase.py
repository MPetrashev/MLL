import inspect
import os
import unittest
from typing import Union

import pandas as pd


class TestCase(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.test_class_dir = os.path.dirname(inspect.getfile(self.__class__))

    def absolute(self, file_name, relative_cls=None):
        """
        :param file_name:
        :param relative_cls:
        :return: Absolute file name relatively to relative_cls class or self
        """
        class_dir = os.path.dirname(inspect.getfile(relative_cls)) if relative_cls else self.test_class_dir
        return os.path.join(class_dir, file_name)

    def get_test_data(self, file_name, index_col: str =None):
        file = self.absolute(file_name)
        return pd.read_csv(file, na_filter=False, parse_dates=True, index_col=index_col)

    def assert_frame_equal(self, expected_df: Union[str, pd.DataFrame], actual_df:pd.DataFrame, ):
        """
        Compare input DataFrame with the other DataFrame or data in csv file.
        :param actual_df:
        :param other_df:
        :return:
        """
        if isinstance(expected_df, str):
            expected_df = self.get_test_data(expected_df, actual_df.index.name)

        dtypes = actual_df.dtypes.to_dict()
        for column, expected_type in expected_df.dtypes.to_dict().items():
            if expected_type != dtypes[column]:
                actual_df[column] = actual_df[column].astype(expected_type)
        pd.testing.assert_frame_equal(expected_df, actual_df)