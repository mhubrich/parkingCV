import os
import numpy as np

import keras.backend as K
from keras.preprocessing.image import Iterator, load_img, img_to_array, array_to_img



class FileIterator(Iterator):
    """Iterator capable of reading two map type images at once from a list of
    filenames (e.g. satellite and roadmap types). The arrays returned by this
    iterator have six channels (i.e. three RGB channels per map type).
    # Arguments
        files: List of filenames (full paths).
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `None`: no targets get yielded (only input images are yielded).
        tags: A tuple of two tuples, each containing 1) a keyword for
            distingushing filenames of the two different map types, and
            2) file extensions for images of these different map types.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
    """
    def __init__(self, files, image_data_generator,
                 target_size=(256, 256), class_mode='binary',
                 tags=(('satellite', 'jpg'), ('roadmap', 'png')),
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None, save_prefix='',
                 save_format='png', subset=None, interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()
        self.filenames = files
        self.image_data_generator = image_data_generator
        print(target_size)
        self.target_size = tuple(target_size)
        if len(tags) != 2:
            raise ValueError('Invalid tags:', tags,
                             '; expected tuple of two tuples.')
        if len(tags[0]) != 2 or len(tags[1]) != 2:
            raise ValueError('Invalid tags:', tags,
                             '; expected tuples of two strings.')
        self.tags = tags
        self.data_format = data_format
        if self.data_format == 'channels_last':
            self.image_shape = self.target_size + (6,)
        else:
            self.image_shape = (6,) + self.target_size
        if class_mode not in {'binary', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "binary" or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError('Invalid subset name: ', subset,
                                 '; expected "training" or "validation"')
        else:
            split = None
        self.subset = subset
        # First, count the number of samples and classes.
        self.samples = 0
        classes = set()
        for f in self.filenames:
            classes.add(os.path.split(os.path.split(f)[0])[-1])
        classes = list(classes)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        num_files = len(self.filenames)
        if split:
            start, stop = int(split[0] * num_files), int(split[1] * num_files)
        else:
            start, stop = 0, num_files
        self.samples = stop - start
        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))
        # Second, build an index of the images
        # in the different class subfolders.
        self.classes = np.zeros((self.samples,), dtype='int32')
        for i, f in enumerate(self.filenames[start:stop]):
            self.classes[i] = self.class_indices[os.path.split(os.path.split(f)[0])[-1]]

        super(FileIterator, self).__init__(self.samples,
                                           batch_size,
                                           shuffle,
                                           seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=K.floatx())
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            fname2 = fname.replace(self.tags[0][0], self.tags[1][0])
            fname2 = os.path.splitext(fname2)[0] + '.' + self.tags[1][1]
            img1 = load_img(os.path.join(self.directory, fname),
                            grayscale=False,
                            target_size=self.target_size,
                            interpolation=self.interpolation)
            x1 = img_to_array(img1, data_format=self.data_format)
            img2 = load_img(os.path.join(self.directory, fname2),
                            grayscale=False,
                            target_size=self.target_size,
                            interpolation=self.interpolation)
            x2 = img_to_array(img2, data_format=self.data_format)
            concat_axis = -1 if self.data_format == 'channels_last' else 0
            x = np.concatenate([x1, x2], axis=concat_axis)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                x1 = batch_x[i,:,:,0:3] if self.data_format == 'channels_last' else batch_x[i,0:3,:,:]
                img = array_to_img(x1, self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
