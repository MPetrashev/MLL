import gc
from torch import Tensor, cuda
import logging

logger = logging.getLogger(__file__)


def optimizer_to(optimizer, device):
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def gpu_memory_info():
    """
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    :return:
    """
    reserved = cuda.memory_reserved(0)
    allocated = cuda.memory_allocated(0)
    return {
        'Total': cuda.get_device_properties(0).total_memory,
        'Reserved': reserved,
        'Allocated': allocated,
        'Free': reserved - allocated
    }


def wipe_memory():
    gc.collect()
    cuda.empty_cache()
    logger.info(f'GPU cache is cleared')