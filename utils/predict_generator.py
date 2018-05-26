"""
Custom `predict_generator` which yields a tuple of (y_true, y_pred).
Source: https://github.com/keras-team/keras/blob/master/keras/engine/training_generator.py#L365
"""
import numpy as np
import warnings

from keras.utils.data_utils import Sequence
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.generic_utils import Progbar


def predict_generator(model, generator,
                      steps=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      verbose=0):
    model._make_predict_function()

    steps_done = 0
    wait_time = 0.01
    all_outs = []
    all_ins = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps is None:
        if is_sequence:
            steps = len(generator)
        else:
            raise ValueError('`steps=None` is only valid for a generator'
                             ' based on the `keras.utils.Sequence` class.'
                             ' Please specify `steps` or use the'
                             ' `keras.utils.Sequence` class.')
    enqueuer = None

    try:
        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if is_sequence:
                output_generator = iter(generator)
            else:
                output_generator = generator

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            if isinstance(generator_output, tuple):
                # Compatibility with the generators
                # used for training.
                if len(generator_output) == 2:
                    x, ys = generator_output
                elif len(generator_output) == 3:
                    x, ys, _ = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
            else:
                # Assumes a generator that only
                # yields inputs (not targets and sample weights).
                x = generator_output

            outs = model.predict_on_batch(x)
            if not isinstance(outs, list):
                outs = [outs]
            if not isinstance(ys, list):
                ys = [ys]

            if not all_outs:
                for _ in outs:
                    all_outs.append([])
            if not all_ins:
                for _ in ys:
                    all_ins.append([])

            for i, out in enumerate(outs):
                all_outs[i].append(out)
            for i, y in enumerate(ys):
                all_ins[i].append(y)

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    if len(all_outs) == 1:
        if steps_done == 1:
            y_pred = all_outs[0][0]
        else:
            y_pred = np.concatenate(all_outs[0])
    if steps_done == 1:
        y_pred = [out[0] for out in all_outs]
    else:
        y_pred = [np.concatenate(out) for out in all_outs]

    if len(all_ins) == 1:
        if steps_done == 1:
            y_true = all_ins[0][0]
        else:
            y_true = np.concatenate(all_ins[0])
    if steps_done == 1:
        y_true = [y[0] for y in all_ins]
    else:
        y_true = [np.concatenate(y) for y in all_ins]

    return y_true, y_pred
