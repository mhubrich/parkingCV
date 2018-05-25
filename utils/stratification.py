import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from utils.misc import list_files


def load_coords(path):
    return json.load(open(path, 'rb'))['parking']


def get_filename(long, lat, width, zoom, maptype):
    return '%f_%f_%dx%d_%d_%s' % (lat, long, width, width, zoom, maptype)


def get_groups(coords_pos, coords_neg, files, width=299, zoom=19, tag='satellite'):
    groups = np.zeros(len(files), dtype=np.int32)
    files_rel = [os.path.splitext(os.path.basename(f))[0] for f in files]
    for i, city in enumerate(coords_neg):
        for coord in city['coords']:
            lat, long = coord['lat'], coord['long']
            if get_filename(long, lat, width, zoom, tag) in files_rel:
                j = files_rel.index(get_filename(long, lat, width, zoom, tag))
                groups[j] = i
    for i, city in enumerate(coords_pos):
        for coord in city['coords']:
            lat, long = coord['lat'], coord['long']
            if get_filename(long, lat, width, zoom, tag) in files_rel:
                j = files_rel.index(get_filename(long, lat, width, zoom, tag))
                groups[j] = i + len(coords_neg)
    return groups


#def train_val_test_split(files, path_coords_pos, path_coords_neg, val_size=0.2, test_size=0.2, seed=None):
#    if val_size >= 1 or test_size >= 1:
#        raise ValueError('Split values should be in range [0,1].')
#    coords_pos = load_coords(path_coords_pos)
#    coords_neg = load_coords(path_coords_neg)
#    zoom = int(files[0].split('_')[-2])
#    width = int(files[0].split('_')[-3].split('x')[0])
#    tag = os.path.splitext(files[0])[0].split('_')[-1]
#    groups = get_groups(coords_pos, coords_neg, files, width=width, zoom=zoom, tag=tag)
#    ind, ind_test = train_test_split(np.arange(len(files)),
#                                     test_size=test_size, stratify=groups,
#                                     shuffle=True, random_state=seed)
#    ind_train, ind_val = train_test_split(ind,
#                                          test_size=val_size, stratify=groups[ind],
#                                          shuffle=True, random_state=seed)
#    file_array = np.array(files)
#    X_train = list(files_array[ind_train])
#    X_val = list(files_array[ind_val])
#    X_test = list(files_array[ind_test])
#    return X_train, X_val, X_test


def train_val_test_split(files, path_coords_pos, path_coords_neg, val_size=0.2, test_size=0.2, seed=None):
    if val_size >= 1 or test_size >= 1:
        raise ValueError('Split values should be in range [0,1].')
    y = np.zeros(len(files), dtype=np.int32)
    for i, f in enumerate(files):
        y[i] = 1 if 'pos' in os.path.split(f)[-2] else 0
    ind, ind_test = train_test_split(np.arange(len(files)),
                                     test_size=test_size, stratify=y,
                                     shuffle=True, random_state=seed)
    ind_train, ind_val = train_test_split(ind,
                                          test_size=val_size, stratify=y[ind],
                                          shuffle=True, random_state=seed)
    files_array = np.array(files)
    X_train = list(files_array[ind_train])
    X_val = list(files_array[ind_val])
    X_test = list(files_array[ind_test])
    return X_train, X_val, X_test


def kfold_train_val_test_split(files, k=10, val_size=0.2, seed=None):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    train, val, test, tmp = [], [], [], []
    X = np.array(files)
    y = np.zeros(len(files), dtype=np.int32)
    for i, f in enumerate(files):
        y[i] = 1 if 'pos' in os.path.split(f)[-2] else 0
    for index, test_index in skf.split(X, y):
        tmp.append(index)
        test.append(X[test_index])
    for index in tmp:
        X_train, X_val, _, _ = train_test_split(X[index], y[index],
                                                stratify=y[index],
                                                test_size=val_size,
                                                shuffle=True,
                                                random_state=seed)
        train.append(X_train)
        val.append(X_val)
    return train, val, test

