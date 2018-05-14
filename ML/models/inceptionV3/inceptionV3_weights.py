import h5py
import numpy as np

from keras.utils.data_utils import get_file
from keras.applications.inception_v3 import WEIGHTS_PATH_NO_TOP


def inceptionV3_weights(cache_dir):
    return get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    WEIGHTS_PATH_NO_TOP,
                    cache_dir=cache_dir,
                    cache_subdir='.')


def inceptionV3_imagenet(cache_dir='.'):
    weights_path = inceptionV3_weights(cache_dir)
    f = h5py.File(weights_path, 'r+')

    if f['conv2d_1']['conv2d_1']['kernel:0'].shape[2] == 3:
        x1 = f['conv2d_1']['conv2d_1']['kernel:0'].value
        x2 = np.array(x1, copy=True)
        x3 = np.concatenate([x1, x2], axis=2)

        del f['conv2d_1']['conv2d_1']['kernel:0']
        _ = f.create_dataset('/conv2d_1/conv2d_1/kernel:0', data=x3)

    return weights_path
