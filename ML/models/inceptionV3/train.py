from ML.training import train
from utils.stratification import train_val_test_split
from utils.misc import list_files


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == "__main__":
    model = 'inception'
    target_size = (224, 224)
    batch_size = 32
    seed = 0
    dir_weights = '/home/mhubrich/.parkingCV_weights/'
    path_images = '/home/mhubrich/maps_300x300_resized_224x224/'
    path_checkpoints = 'weights.{epoch:02d}-{val_loss:.3f}.hdf5'
    path_logs = 'training.log'
    files = list_files(path_images, 'satellite')
    files_train, files_val, files_test, = train_val_test_split(files,
                                                               None,
                                                               None,
                                                               val_size=0.2,
                                                               test_size=0.2,
                                                               seed=seed)
    loss, acc = train(model, files_train, files_val, files_test,
                      preprocess_input=preprocess_input,
                      target_size=target_size,
                      dense=[1024],
                      freeze=311,
                      batch_size=batch_size,
                      seed=seed,
                      dir_weights=dir_weights,
                      path_checkpoints=path_checkpoints,
                      path_logs=path_logs,
                      mode='evaluate')
    
    print('Test results: Loss: %.3f - Acc: %.3f' % (loss, acc))

