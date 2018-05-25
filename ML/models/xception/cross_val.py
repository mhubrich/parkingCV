import numpy as np

from ML.training import train
from utils.stratification import kfold_train_val_test_split
from utils.misc import list_files


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == "__main__":
    model = 'xception'
    target_size = (224, 224)
    batch_size = 32
    seed = 0
    k = 10
    dir_weights = '/home/mhubrich/.parkingCV_weights/'
    path_images = '/home/mhubrich/maps_300x300_resized_224x224/'
    path_checkpoints = 'weights.{epoch:02d}-{val_loss:.3f}.hdf5'
    path_logs = 'training.{fold:02d}.log'
    files = list_files(path_images, 'satellite')
    files_train, files_val, files_test = kfold_train_val_test_split(files,
                                                                    k=k,
                                                                    val_size=0.2,
                                                                    seed=seed)
    losses = np.zeros(k, dtype=np.float32)
    accs = np.zeros(k, dtype=np.float32)
    for i in range(k):
        print('Starting fold %02d/%d.' % (i+1, k))
        loss, acc = train(model, files_train[i], files_val[i], files_test[i],
                          preprocess_input=preprocess_input,
                          target_size=target_size,
                          dense=[1024],
                          freeze=132,
                          batch_size=batch_size,
                          seed=seed,
                          dir_weights=dir_weights,
                          path_checkpoints=path_checkpoints,
                          path_logs=path_logs.format(fold=i+1))
        print('Test set results: Loss: %.3f - Acc: %.3f' % (loss, acc))
        losses[i] = loss
        accs[i] = acc
    
    print('Results of the %d-fold cross-validation:' % k)
    print('Mean: Loss: %.3f - Acc: %.3f' % (np.mean(losses), np.mean(accs)))
    print(' Std: Loss: %.3f - Acc: %.3f' % (np.std(losses), np.std(accs)))

