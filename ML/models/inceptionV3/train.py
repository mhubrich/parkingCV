"""
TODO:   * train, val, test split: stratification based on lat/long
        * k-fold cross-validation
"""
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from utils.preprocessing.image_data_generator import ImageDataGenerator
from utils.stratification import train_val_test_split
from utils.misc import list_files
from ML.models.inceptionV3.model import get_model


VERBOSITY = 1


def fit_stats(iterator, row_axis=1, col_axis=2, mean=None, std=None):
    if mean is None:
        mean = np.zeros(6, dtype=np.float32)
        count = 0
        iterator.reset()
        for _ in range(len(iterator)):
            x = iterator.next()
            if iterator.class_mode == 'binary':
                x = x[0]
            mean += np.mean(x, axis=(0, row_axis, col_axis))
            count += 1
        mean /= count
    else:
        if len(mean) != 6:
            raise ValueError('Invalid mean-array: ', mean,
                             '; expected array of length 6.')
    if std is None:
        std = np.zeros(6, dtype=np.float32)
        count = 0
        iterator.reset()
        for _ in range(len(iterator)):
            x = iterator.next()
            if iterator.class_mode == 'binary':
                x = x[0]
            std += np.sum(np.power(x - mean, 2), axis=(0, row_axis, col_axis))
            count += x.shape[0] * x.shape[row_axis] * x.shape[col_axis]
        std = np.sqrt(std / count)
    else:
        if len(std) != 6:
            raise ValueError('Invalid std-array: ', std,
                             '; expected array of length 6.')
    return mean, std


def get_data(f_train, f_val, f_test, target_size=(299, 299), batch_size=32, seed=None):
    generator_stats = ImageDataGenerator(rescale=1/255.)
    iterator_stats = generator_stats.flow_from_files(f_train,
                                                     target_size=target_size,
                                                     batch_size=batch_size,
                                                     class_mode=None,
                                                     shuffle=False,
                                                     seed=seed)
    mean, std = fit_stats(iterator_stats, mean=None, std=None)
    generator_train = ImageDataGenerator(rescale=1/255.,
                                         rotation_range=90.,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         brightness_range=(0.95, 1.05),
                                         channel_shift_range=10.)
    generator_val = ImageDataGenerator(rescale=1/255.)
    generator_test = ImageDataGenerator(rescale=1/255.)
    generator_train.set_stats(mean, std)
    generator_val.set_stats(mean, std)
    generator_test.set_stats(mean, std)
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


def get_model(model=None, target_size=(299, 299), dense=[1024], freeze=312):
    if model is None:
        model = get_model(shape=target_size + (6,), dense=dense)

    for layer in model.layers[:freeze]:
        layer.trainable = False
    for layer in model.layers[freeze:]:
        layer.trainable = True

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train(iterator_train, iterator_val, model, epochs=10,
          path_weights=None, path_output=None):
    if path_weights is not None and path_output is not None:
        callbacks = [ModelCheckpoint(path_weights,
                                     save_best_only=True,
                                     save_weights_only=True),
                     CSVLogger(path_output, separator=',', append=False),
                     EarlyStopping(patience=2)]
    else:
        callbacks = None

    _ = model.fit_generator(iterator_train,
                            steps_per_epoch=len(iterator_train),
                            epochs=epochs,
                            verbose=VERBOSITY,
                            callbacks=callbacks,
                            validation_data=iterator_val,
                            validation_steps=len(iterator_val),
                            max_queue_size=2*iterator_train.batch_size,
                            workers=3,
                            use_multiprocessing=True,
                            shuffle=True,
                            initial_epoch=0)
    return model


def evaluate(iterator_test, model):
    return model.evaluate_generator(iterator_test,
                                    steps=len(iterator_test),
                                    max_queue_size=2*iterator_test.batch_size,
                                    workers=3,
                                    use_multiprocessing=True,
                                    verbose=VERBOSITY)


if __name__ == "__main__":
    target_size = (299, 299)
    batch_size = 32
    seed = 0
    path_weights = 'weights.{epoch:02d}-{val_loss:.3f}.hdf5'
    path_output = 'training.log'
    path_images = None # TODO
    path_coords_pos = None # TODO
    path_coords_neg = None # TODO
    files = list_files(path_images, 'satellite')
    # 0) Get data in form of iterators
    f_train, f_val, f_test, = train_val_test_split(files,
                                                   path_coords_pos,
                                                   path_coords_neg,
                                                   val_size=0.2,
                                                   test_size=0.2,
                                                   seed=seed)
    iterator_train, iterator_val, iterator_test = get_data(f_train, f_val, f_test,
                                                           target_size=target_size,
                                                           batch_size=batch_size,
                                                           seed=seed)
    # 1) Train FC layer for one epoch
    model = get_model(model=None, target_size=target_size, dense=[1024], freeze=312)
    model = train(iterator_train, iterator_val, model, epochs=1)
    # 2) Train some of the last convolutional layers + FC layers
    model = get_model(model=model, target_size=target_size, dense=[1024], freeze=312)
    model = train(iterator_train, iterator_val, model, path_weights=path_weights, path_output=path_output, epochs=10)
    # 3) Evaluate model on test set
    results = evaluate(iterator_test, model)
    print(results)
