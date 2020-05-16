"""
The code implement the Method for Multimodal Compact Bilinear Pooling (MCB)
"""

import numpy as np
import pickle
import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle

def mcb(features1, features2, d:int=16000, save=False, filename="mcb_feature.pickle"):
    if features1.shape[0] != features2.shape[0]:
        _raise_sample_size_exception()

    h, s = _count_sketch_init([features1.shape[1], features2.shape[1]], d)

    sketch_features1 = []
    sketch_features2 = []

    for v0, v1 in zip(features1, features2):
        sketch_features1.append(_count_sketch(d, h[0], s[0], v0))
        sketch_features2.append(_count_sketch(d, h[1], s[1], v1))

    # fft
    fft_features1 = []
    fft_features2 = []

    for v0, v1 in zip(sketch_features1, sketch_features2):
        fft_features1.append(np.fft.fft(v0))
        fft_features2.append(np.fft.fft(v1))

    ewp_features = np.multiply(fft_features1, fft_features2)

    ifft_features = np.fft.ifft(ewp_features)

    mcb_features = np.real(ifft_features)


    if save:
        try:
            with open(filename, "wb") as fout:
                pickle.dump(mcb_features, fout)
        except Exception as e:
            raise e

    return mcb_features

def _count_sketch(d, h, s, v):

    cs_vector = np.zeros(d).astype("float64")

    for dim_num, _ in enumerate(v):
        cs_vector[h[dim_num]] += s[dim_num] * v[dim_num]
        
    return cs_vector

def _count_sketch_init(feature_dims, d):
    h = [None, None]
    s = [None, None]

    for vec_num in range(2):
        h[vec_num] = np.random.randint(0, d-1, size=(feature_dims[vec_num], ))
        s[vec_num] = (np.floor(np.random.uniform(0, 2, size=(feature_dims[vec_num], ))) * 2 - 1).astype("int64")
    return h, s


class MCBException(Exception):
    """base class for mcb exceptions"""

class SampleSizeException(MCBException):
    """raise when sample size of two features does not match"""

def _raise_sample_size_exception():
    raise SampleSizeException("size of samples does not match")
