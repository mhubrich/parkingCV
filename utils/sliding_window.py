import numpy as np


def sliding_window(img, window_h, window_w, stride):
    """Generator which yields chuncks of the input image by sliding a window
    through it. In case the image size does not match the last stride of the
    window, it gets padded with mode `reflect`.
        Arguments:
          img: Array. Has to have dimensions (width, height, channels).
          window_h: Integer. Height of the sliding window.
          window_w: Integer. Width of the sliding window.
          stride: Integer. Step size of the sliding window.
        Yields:
          Chunks of size (window_h, window_w, channels) and
          tuple (pos_y, pos_x), where pos determines the top left position of
          the corresponding chunk in the input image.
    """
    if len(img.shape) != 3:
        raise ValueError('Expected an array of dimensions '
                         '(width, height, channels). Found: %s' % img.shape)
    if window_h > img.shape[0] or window_w > img.shape[1]:
        raise ValueError('Window size cannot be bigger than image size. '
                         'Image size: %s, window size: %s.'
                         % (img.shape, (window_h, window_w)))
    steps_y = int(np.ceil((img.shape[0] - window_h) / float(stride) + 1))
    steps_x = int(np.ceil((img.shape[1] - window_w) / float(stride) + 1))
    pad_y = (steps_y - 1) * stride + window_h - img.shape[0]
    pad_x = (steps_x - 1) * stride + window_w - img.shape[1]
    img = np.pad(img, pad_width=((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')
    for y in range(steps_y):
        pos_y = y * stride
        for x in range(steps_x):
            pos_x = x * stride
            yield img[pos_y:pos_y+window_h, pos_x:pos_x+window_w], (pos_y, pos_x)
