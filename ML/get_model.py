import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
#from ML.models.xception.xception import Xception
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Input, Dense, Dropout

from ML.imagenet_weights import get_weights


"""Returns a pretrained model of either InceptionV3, Xception or DenseNet121.
 Models are pretrained on ImageNet and capable of operating on 6-channel images.
    model: String. One of 'inception', 'xception' or 'densenet121'.
    dir_weights: String. Path to directory where model weights are saved.
    shape: Tuple of three Integers. Input shape of the model. The channel axis
        is supposed to have dimension of 6.
    dense: List of Integers. The number of units of each fully-connected layer.
    dropout: Float. The Dropout probability between each FC layer.
    data_format: String. One of 'channels_last' or 'channels_first'. If None
    (default), the data format is determined automatically.
"""
def get_model(model, dir_weights, shape=(224, 224, 6), dense=[1024], dropout=0.5, data_format=None):
    if model is None or model.lower() not in ('inception', 'xception', 'densenet121'):
        raise ValueError('Invalid value for `model`: ', model,
                         '; valid values are: inception, xception, densenet121.')
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in ('channels_last', 'channels_first'):
        raise ValueError('`data_format` should be `"channels_last"` '
                         'or `"channels_first"`. Received: %s' % data_format)
    img_row_axis = 0 if data_format == 'channels_last' else 1
    img_col_axis = 1 if data_format == 'channels_last' else 2
    img_channel_axis = 2 if data_format == 'channels_last' else 0
    if len(shape) != 3 or shape[img_channel_axis] != 6:
        raise ValueError('Input shape should be of rank 3 with 6 channels.')

    weights = get_weights(model, dir_weights)
    input_tensor = Input(shape=shape)

    if model.lower() == 'inception':
        base_model = InceptionV3(input_tensor=input_tensor, weights=weights,
                                 include_top=False, pooling='avg')
    elif model.lower() == 'xception':
        base_model = Xception(input_tensor=input_tensor, weights=weights,
                              include_top=False, pooling='avg')
    else:
        base_model = DenseNet121(input_tensor=input_tensor, weights=weights,
                                 include_top=False, pooling='avg')

    x = base_model.output
    for nodes in dense:
        x = Dense(nodes, activation='relu')(x)
        x = Dropout(dropout)(x)
    prediction = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=base_model.input, outputs=prediction)

