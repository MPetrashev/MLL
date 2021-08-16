from typing import Iterable, Sequence
from sklearn.preprocessing import MinMaxScaler

from utils import lazy_property
import numpy as np


class NormalizedData:
    """
    Normalize data to [0,1] range as described here: https://towardsdatascience.com/pytorch-tabular-regression-428e9c9ac93
    """

    def __init__(self, X: Iterable[Sequence[float]], Y: Iterable[float]) -> None:
        self.original_X = X
        self.original_Y = Y
        self.expanded_Y = np.expand_dims(Y, 1)

    @lazy_property
    def scaler_X(self) -> MinMaxScaler:
        return MinMaxScaler().fit(self.original_X)

    @lazy_property
    def scaler_Y(self) -> MinMaxScaler:
        return MinMaxScaler().fit(self.expanded_Y)

    @lazy_property
    def X(self) -> Iterable[Sequence[float]]:
        return self.scaler_X.transform(self.original_X)

    @lazy_property
    def Y(self) -> Iterable[float]:
        return np.squeeze( self.scaler_Y.transform(self.expanded_Y) )
