"""
This file provides the standardized function `train` for training
various models, e.g. InceptionV3, Xception or DenseNet121.
"""
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau

from utils.my_model_checkpoint import MyModelCheckpoint
from utils.preprocessing.image_data_generator import ImageDataGenerator
from ML.get_model import get_model


def get_data(f_train, f_val, f_test,
             preprocess_input=None,
             target_size=(224, 224),
             batch_size=32,
             seed=None):
    generator_train = ImageDataGenerator(rescale=preprocess_input,
                                         rotation_range=360.,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         brightness_range=(0.925, 1.075),
                                         channel_shift_range=20.)
    generator_val = ImageDataGenerator(rescale=preprocess_input)
    generator_test = ImageDataGenerator(rescale=preprocess_input)
    iterator_train = generator_train.flow_from_files(f_train,
                                                     target_size=target_size,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     seed=seed)
    iterator_val = generator_val.flow_from_files(f_val,
                                                 target_size=target_size,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 seed=seed)
    iterator_test = generator_test.flow_from_files(f_test,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   seed=seed)
    return iterator_train, iterator_val, iterator_test


def set_model(model, dir_weights=None,
              target_size=(224, 224),
              dense=[1024],
              freeze=-1,
              lr=0.001):
    if isinstance(model, str):
        model = get_model(model, dir_weights, shape=target_size + (6,), dense=dense)

    for layer in model.layers[:freeze+1]:
        layer.trainable = False
    for layer in model.layers[freeze+1:]:
        layer.trainable = True

    rmsprop = RMSprop(lr=lr, rho=0.9, decay=0.0)
    model.compile(optimizer=rmsprop,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def training(model, iterator_train, iterator_val,
             epochs=10,
             path_checkpoints=None,
             path_logs=None):
    callbacks = [EarlyStopping(patience=2),
                 ReduceLROnPlateau(factor=0.5,
                                   patience=1,
                                   min_delta=0.005,
                                   min_lr=1e-6,
                                   verbose=1)]
    if path_checkpoints:
        model_checkpoint = MyModelCheckpoint(path_checkpoints,
                                             save_best_only=True,
                                             save_weights_only=True)
        callbacks.append(model_checkpoint)
    if path_logs:
        callbacks.append(CSVLogger(path_logs, separator=',', append=False))

    _ = model.fit_generator(iterator_train,
                            steps_per_epoch=len(iterator_train),
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=iterator_val,
                            validation_steps=len(iterator_val),
                            max_queue_size=2*iterator_train.batch_size,
                            workers=3,
                            use_multiprocessing=True,
                            shuffle=True,
                            initial_epoch=0)
    if path_checkpoints:
        return model, model_checkpoint.path_best_weights
    return model


def evaluate(model, iterator_test):
    return model.evaluate_generator(iterator_test,
                                    steps=len(iterator_test),
                                    max_queue_size=2*iterator_test.batch_size,
                                    workers=3,
                                    use_multiprocessing=True,
                                    verbose=1)


def predict(model, iterator_test):
    return model.predict_generator(iterator_test,
                                   steps=len(iterator_test),
                                   max_queue_size=2*iterator_test.batch_size,
                                   workers=3,
                                   use_multiprocessing=True,
                                   verbose=1)


def train(model, files_train, files_val, files_test,
          preprocess_input=None,
          target_size=(224, 224),
          dense=[1024],
          freeze=132,
          batch_size=32,
          seed=None,
          dir_weights=None,
          path_checkpoints=None,
          path_logs=None,
          mode=None):
    if model is None or model.lower() not in ('inception', 'xception', 'densenet121'):
        raise ValueError('Invalid value for `model`: ', model,
                         '; valid values are: `inception`, `xception`, `densenet121`.')
    if mode not in ('evaluate', 'predict', None):
        raise ValueError('Invalid value for `mode`: ', mode,
                         '; valid values are: `evaluate`, `predict` or `None`')
    # 0) Get the iterators for training, validation and testing
    iterator_train, iterator_val, iterator_test = get_data(files_train,
                                                           files_val, files_test,
                                                           preprocess_input=preprocess_input,
                                                           target_size=target_size,
                                                           batch_size=batch_size,
                                                           seed=seed)
    # 1) Train only FC layer for one epoch
    model = set_model(model, dir_weights,
                      target_size=target_size,
                      dense=dense,
                      freeze=freeze,
                      lr=0.001)
    model = training(model, iterator_train, iterator_val, epochs=1)
    # 2) Train complete model
    model = set_model(model=model, freeze=-1, lr=0.0001)
    model, path_best_weights = training(model, iterator_train, iterator_val,
                                        path_checkpoints=path_checkpoints,
                                        path_logs=path_logs,
                                        epochs=7)
    # 3) Evaluate model on test set using best weights
    model.load_weights(path_best_weights)
    if mode == 'evaluate':
        return evaluate(model, iterator_test)
    if mode == 'predict':
        return predict(model, iterator_test)
    return model

