import numpy as np
import matplotlib.pyplot as plt
import math
from ctypes import CDLL, c_double, c_int, POINTER

import torch

PersistencePython = CDLL("/home/licaizi/crz/miccai2025/utils/PersistencePython.so")
PersistencePython.cubePers.argtypes = [
    POINTER(c_double), c_int,  # double* data, int data_size
    POINTER(c_int), c_int,     # int* ints, int ints_size
    c_double                   # double arg
]
PersistencePython.cubePers.restype = None


# from PersistencePython import cubePers

def compute_persistence_2DImg_1DHom_lh(f, padwidth=2, homo_dim=1, pers_thd=0.001):
    """
    compute persistence diagram in a 2D function (can be N-dim) and critical pts
    only generate 1D homology dots and critical points
    """
    print(len(f.shape))
    assert len(f.shape) == 2  # f has to be 2D function
    dim = 2

    # pad the function with a few pixels of minimum values
    # this way one can compute the 1D topology as loops
    # remember to transform back to the original coordinates when finished
    # padwidth = 2
    # padvalue = min(f.min(), 0.0)
    padvalue = f.min()
    # print (type(f.cpu().detach().numpy()))
    # print (padvalue)
    if (not isinstance(f, np.ndarray)):
        f_padded = np.pad(f.cpu().detach().numpy(), padwidth, 'constant',
                          constant_values=padvalue.cpu().detach().numpy())
    else:
        f_padded = np.pad(f, padwidth, 'constant', constant_values=padvalue)

    # call persistence code to compute diagrams
    # loads PersistencePython.so (compiled from C++); should be in current dir
    # from PersistencePython import cubePers

    # persistence_result = cubePers(a, list(f_padded.shape), 0.001)
    persistence_result = PersistencePython.cubePers(np.reshape(
        f_padded, f_padded.size).tolist(), list(f_padded.shape), pers_thd)

    # print("persistence_result", type(persistence_result))
    # print(type(persistence_result))
    # print (persistence_result)
    # print(len(persistence_result))

    # only take 1-dim topology, first column of persistence_result is dimension
    persistence_result_filtered = np.array(list(filter(lambda x: x[0] == homo_dim,
                                                       persistence_result)))

    # persistence diagram (second and third columns are coordinates)
    # print (persistence_result_filtered)
    if (persistence_result_filtered.shape[0] == 0):
        return np.array([]), np.array([]), np.array([])
    dgm = persistence_result_filtered[:, 1:3]

    # critical points
    birth_cp_list = persistence_result_filtered[:, 4:4 + dim]
    death_cp_list = persistence_result_filtered[:, 4 + dim:]

    # when mapping back, shift critical points back to the original coordinates
    birth_cp_list = birth_cp_list - padwidth
    death_cp_list = death_cp_list - padwidth

    return dgm, birth_cp_list, death_cp_list


def compute_persistence_2DImg_1DHom_gt(f, padwidth=2, homo_dim=1, pers_thd=0.001):
    """
    compute persistence diagram in a 2D function (can be N-dim) and critical pts
    only generate 1D homology dots and critical points
    """
    # print (len(f.shape))
    assert len(f.shape) == 2  # f has to be 2D function
    dim = 2

    # pad the function with a few pixels of minimum values
    # this way one can compute the 1D topology as loops
    # remember to transform back to the original coordinates when finished
    # padwidth = 2
    # padvalue = min(f.min(), 0.0)
    padvalue = f.min()
    # print(f)
    # print (type(f.cpu().numpy()))
    if (not isinstance(f, np.ndarray)):
        f_padded = np.pad(f.cpu().detach().numpy(), padwidth, 'constant',
                          constant_values=padvalue.cpu().detach().numpy())
    else:
        f_padded = np.pad(f, padwidth, 'constant', constant_values=padvalue)

    # call persistence code to compute diagrams
    # loads PersistencePython.so (compiled from C++); should be in current dir
    # from src.PersistencePython import cubePers
    # from PersistencePython import cubePers

    # persistence_result = cubePers(a, list(f_padded.shape), 0.001)
    persistence_result = PersistencePython.cubePers(np.reshape(
        f_padded, f_padded.size).tolist(), list(f_padded.shape), pers_thd)

    # print("persistence_result", type(persistence_result))
    # print(type(persistence_result))
    # print (persistence_result)
    # print(len(persistence_result))

    # only take 1-dim topology, first column of persistence_result is dimension
    persistence_result_filtered = np.array(list(filter(lambda x: x[0] == homo_dim,
                                                       persistence_result)))

    # persistence diagram (second and third columns are coordinates)
    # print (persistence_result_filtered)
    # print ('shape of persistence_result_filtered')
    # print (persistence_result_filtered.shape)
    if (persistence_result_filtered.shape[0] == 0):
        return np.array([]), np.array([]), np.array([])
    dgm = persistence_result_filtered[:, 1:3]

    # critical points
    birth_cp_list = persistence_result_filtered[:, 4:4 + dim]
    death_cp_list = persistence_result_filtered[:, 4 + dim:]

    # when mapping back, shift critical points back to the original coordinates
    birth_cp_list = birth_cp_list - padwidth
    death_cp_list = death_cp_list - padwidth

    return dgm, birth_cp_list, death_cp_list

