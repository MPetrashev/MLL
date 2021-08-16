from typing import Sequence

import numpy as np
import torch


def less_than_Npc_exceeds_Mpc_diff(y: Sequence[float], y_prediction: torch.Tensor, alpha: float = 0.01,
                                   rtol: float = 0.001) -> bool:
    """
    :param y:
    :param y_prediction:
    :param alpha:
    :param rtol:
    :return: True if less than 1% of y_prediction elements deviate from original y values more than on 10bps
    """
    y = y.squeeze()
    y_prediction = y_prediction.detach().cpu().numpy().squeeze()
    arr = np.isclose(y, y_prediction, rtol=rtol) # 10bps remove y.squeeze() and reuse original array
    n = np.count_nonzero(arr)
    return n / y.shape[0] > (1-alpha)