import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, Dense, Dropout

from ML.models.inceptionV3.inceptionV3_weights import inceptionV3_weights


def get_model(shape=(299, 299, 6), dense=[1024, 1024], dropout=0.5, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in ('channels_last', 'channels_first'):
        raise ValueError(
            '`data_format` should be `"channels_last"` '
            'or `"channels_first"`. Received: %s' % data_format)
    img_row_axis = 0 if data_format == 'channels_last' else 1
    img_col_axis = 1 if data_format == 'channels_last' else 2
    img_channel_axis = 2 if data_format == 'channels_last' else 0
    if len(shape) != 3 or shape[img_channel_axis] != 6:
        raise ValueError('Input shape should be of rank 3 with 6 channels.')
    if shape[img_row_axis] < 139 or shape[img_col_axis] < 139:
        raise ValueError('Input should be at least of size 139x139.')

    weights = inceptionV3_weights()
    input_tensor = Input(shape=shape)

    base_model = InceptionV3(input_tensor=input_tensor, weights=weights,
                             include_top=False, pooling='avg')

    x = base_model.output
    for nodes in dense:
        x = Dense(nodes, activation='relu')(x)
        x = Dropout(dropout)(x)
    prediction = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=prediction)

    return model
