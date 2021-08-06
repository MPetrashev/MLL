import math

from torch import Tensor


def DataLoader(x: Tensor, y: Tensor, batch_size: int = None):
    """
    This is a patch generator to torch.utils.data import DataLoader which is VERY slow on 1st iteration
    """
    if batch_size is None:
        yield x, y
    else:
        n_batches = math.ceil(y.shape[0] / batch_size)
        offset = 0
        for batch_ndx in range(n_batches):
            offset += batch_size
            yield x[offset:offset+batch_size, :], y[offset:offset+batch_size]