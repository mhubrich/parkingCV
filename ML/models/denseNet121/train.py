from ML.training import train
from utils.stratification import train_val_test_split
from utils.misc import list_files


def preprocess_input(x):
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]
    return x


if __name__ == "__main__":
    model = 'densenet121'
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
                      freeze=426,
                      batch_size=batch_size,
                      seed=seed,
                      dir_weights=dir_weights,
                      path_checkpoints=path_checkpoints,
                      path_logs=path_logs)
    
    print('Test results: Loss: %.3f - Acc: %.3f' % (loss, acc))

