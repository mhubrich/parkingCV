import numpy as np
import skimage.transform

from utils.sliding_window import SlidingWindow


def predict_map(map, predict, window_h, window_w, stride, batch_size, **kwargs):
    """Creates a probability map of predictions of the high resolution input
    image `map` using a sliding window approach.
        Arguments:
          map: Array. Input image with dimensions of (height, width, 6).
          predict: Function. It takes a batch of images as first input argument
                   followed by `**kwargs` arguments. It returns exactly one
                   probability per batch sample. Hence, for an input of
                   len(batch) = n, it returns a list of n floats.
                   For example, this function could be choosen to be
                   * a model making predictions of the given batch
                   * the averaged predictions of augmentations of the batch
                   * predictions of an ensemble of classifiers
          window_h: Integer. Height of the sliding window.
          window_w: Integer. Width of the sliding window.
          stride: Integer. Step size of the sliding window.
          kwargs: Arguments passed to function `predict`.
        Returns:
          A probability map of predictions of `map`.
    """
    iter = SlidingWindow(map, window_h, window_w, stride)
    map_proba = np.zeros(iter.img.shape[:2], dtype=np.float32)
    map_counts = np.zeros(iter.img.shape[:2], dtype=np.int32)
    for i in range(int(np.ceil(iter.n / float(batch_size)))):
        num_samples = min((i+1) * batch_size, iter.n) - i * batch_size
        batch_windows = np.zeros((num_samples,) + iter.window_size, dtype=np.uint8)
        batch_pos = np.zeros((num_samples, 2), dtype=np.int32)
        for j in range(num_samples):
            window, pos = iter.next()
            batch_windows[j] = window
            batch_pos[j] = pos
        predicitons = predict(batch_windows, **kwargs)
        for j in range(len(predicitons)):
            y1, y2 = batch_pos[j, 0], batch_pos[j, 0] + window_h
            x1, x2 = batch_pos[j, 1], batch_pos[j, 1] + window_w
            map_proba[y1:y2, x1:x2] = predicitons[j]
            map_counts[y1:y2, x1:x2] += 1
    map_proba /= map_counts
    return map_proba[:map.shape[0], :map.shape[1]]


def predict_mirror(batch, model, target_size, preprocess_input):
    """Returns predictions of each sample in `batch` by taking the average
    prediction produced by `model` of
      * the original image,
      * horizontally mirrored image,
      * vertically mirrored image,
      * horizontally and vertically mirrored image.
        Arguments:
          batch: Array of images. Has dimensions of (n, height, width, channels).
          model: Model. Classifier used to make predictions of images.
          target_size: Tuple of two Integers. The input shape of the model.
          preprocess_input: Function. Preprocessing of input images.
        Returns:
          Array of predictions of `batch`.
    """
    batch_augmented = np.zeros((len(batch)*4,) + target_size + (batch.shape[-1],),
                               dtype=np.float32)
    for i, img in enumerate(batch):
        img_resized = skimage.transform.resize(img,
                                               target_size,
                                               preserve_range=True)
        batch_augmented[i * 4 + 0] = preprocess_input(img_resized)
        batch_augmented[i * 4 + 1] = preprocess_input(img_resized[::-1, :, :])
        batch_augmented[i * 4 + 2] = preprocess_input(img_resized[:, ::-1, :])
        batch_augmented[i * 4 + 3] = preprocess_input(img_resized[::-1, ::-1, :])
    preds = model.predict_on_batch(batch_augmented)
    y_pred = np.zeros(len(batch), dtype=np.float32)
    for i in range(len(batch)):
        y_pred[i] = np.mean(y_pred[i * 4:(i+1) * 4])
    return y_pred
