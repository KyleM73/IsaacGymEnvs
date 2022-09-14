import torch

def apply_tensor_value(tensor,idx,value,num=None):
    """
    applies 'value' to 'tensor' at position 'idx' with step size 'num' 
    """
    if num is None and len(tensor.size()) == 2:
        num = tensor.size()[1]
    if num == 0:
        tensor[idx, ...] = value
    else:
        tensor[idx::num, ...] = value
    return tensor
