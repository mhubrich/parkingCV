import numpy as np


"""Iterator which yields chunks of the input image by sliding a window
through it. The image gets padded in case its size does not match the last
stride of the window.
    Arguments:
      img: Array. Has to have dimensions (width, height, channels).
      window_h: Integer. Height of the sliding window.
      window_w: Integer. Width of the sliding window.
      stride: Integer. Step size of the sliding window.
      mode: String. Parameter for `numpy.pad()`.
    Yields:
      Chunks of size (window_h, window_w, channels) and
      tuple (pos_y, pos_x), where pos determines the top left position of
      the corresponding chunk in the input image.
"""
class SlidingWindow(object):
    def __init__(self, img, window_h, window_w, stride, mode='reflect'):
        if len(img.shape) != 3:
            raise ValueError('Expected an array of dimensions '
                             '(width, height, channels). Found: %s' % img.shape)
        if window_h > img.shape[0] or window_w > img.shape[1]:
            raise ValueError('Window size cannot be bigger than image size. '
                             'Image size: %s, window size: %s.'
                             % (img.shape, (window_h, window_w)))
        self.window_h = window_h
        self.window_w = window_w
        self.stride = stride
        self.window_size = (window_h, window_w, img.shape[2])
        steps_y = int(np.ceil((img.shape[0] - window_h) / float(stride) + 1))
        steps_x = int(np.ceil((img.shape[1] - window_w) / float(stride) + 1))
        pad_y = (steps_y - 1) * stride + window_h - img.shape[0]
        pad_x = (steps_x - 1) * stride + window_w - img.shape[1]
        self.img = np.pad(img, pad_width=((0, pad_y), (0, pad_x), (0, 0)), mode=mode)
        self.index_array = [(y, x) for y in range(steps_y) for x in range(steps_x)]
        self.n = len(self.index_array)
        self.count = 0

    def __iter__(self):
        return self

    def next(self):
        if self.count >= self.n:
            raise StopIteration
        pos_y = self.index_array[self.count][0] * self.stride
        pos_x = self.index_array[self.count][1] * self.stride
        window = self.img[pos_y:pos_y+self.window_h, pos_x:pos_x+self.window_w]
        self.count += 1
        return window, (pos_y, pos_x)
