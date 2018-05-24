import h5py
import numpy as np

from keras.utils.data_utils import get_file
from keras.applications.inception_v3 import WEIGHTS_PATH_NO_TOP as INCEPTION_WEIGHTS_PATH_NO_TOP
from keras.applications.xception import TF_WEIGHTS_PATH_NO_TOP as XCEPTION_WEIGHTS_PATH_NO_TOP
from keras.applications.densenet import DENSENET121_WEIGHT_PATH_NO_TOP


def _load_weights(filename, url, dir_weights, cache_subdir):
    return get_file(filename, url,
                    cache_dir=dir_weights,
                    cache_subdir=cache_subdir)


"""Loads pretrained weights for a particular model, e.g. InceptionV3, Xception
 or Densenet121. The weights have been pretained on ImageNet and are modified
 to support 6-channel inputs.
    model: String. One of 'inception', 'xception' or 'densenet121'.
    dir_weights: String. Path to directory where model weights are saved.
"""
def get_weights(model, dir_weights):
    if model is None or model.lower() not in ('inception', 'xception', 'densenet121'):
        raise ValueError('Invalid value for `model`: ', model,
                         '; valid values are: inception, xception, densenet121.')
    if model.lower() == 'inception':
        weights_path = _load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                     INCEPTION_WEIGHTS_PATH_NO_TOP,
                                     dir_weights=dir_weights,
                                     cache_subdir='inceptionV3')
        dataset_path = '/conv2d_1/conv2d_1/kernel:0'
    elif model.lower() == 'xception':
        weights_path = _load_weights('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                     XCEPTION_WEIGHTS_PATH_NO_TOP,
                                     dir_weights=dir_weights,
                                     cache_subdir='xception')
        dataset_path = '/convolution2d_1/convolution2d_1_W:0'
    else:
        weights_path = _load_weights('densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                     DENSENET121_WEIGHT_PATH_NO_TOP,
                                     dir_weights=dir_weights,
                                     cache_subdir='denseNet121')
        dataset_path = '/conv1/conv/conv1/conv/kernel:0'
    
    f = h5py.File(weights_path, 'r+')

    if model.lower() == 'inception':
        dataset = f['conv2d_1']['conv2d_1']['kernel:0']
    elif model.lower() == 'xception':
        dataset = f['convolution2d_1']['convolution2d_1_W:0']
    else:
        dataset = f['conv1']['conv']['conv1']['conv']['kernel:0']

    if dataset.shape[2] == 3:
        x1 = dataset.value
        x2 = np.array(x1, copy=True)
        x3 = np.concatenate([x1, x2], axis=2)
        
        if model.lower() == 'inception':
            del f['conv2d_1']['conv2d_1']['kernel:0']
        elif model.lower() == 'xception':
            del f['convolution2d_1']['convolution2d_1_W:0']
        else:
            del f['conv1']['conv']['conv1']['conv']['kernel:0']
        
        _ = f.create_dataset(dataset_path, data=x3)

    return weights_path

