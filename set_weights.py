import numpy as np
import torch

def set_weights(parameters, weights, use_cuda):
    """
    Function used to set the value of a set of torch parameters given a
    vector of values.

    Args:
        parameters (list): list of parameters to be considered;
        weights (numpy.ndarray): array of the new values for
            the parameters;
        use_cuda (bool): whether the parameters are cuda tensors or not;

    """
    idx = 0
    for p in parameters:
        shape = p.data.shape

        c = 1
        for s in shape:
            c *= s

        w = np.reshape(weights[idx:idx + c], shape)

        if not use_cuda:
            w_tensor = torch.from_numpy(w).type(p.data.dtype)
        else:
            w_tensor = torch.from_numpy(w).type(p.data.dtype).cuda()

        p.data = w_tensor
        idx += c

    assert idx == weights.size

