import numpy as np

def shuffle_data(X, Y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    return X, Y, indices

def val_split(X, Y, val_split):
    nb_train_samples = int((1-val_split) * X.shape[0])
    xtrain = X[0:nb_train_samples]
    xval   = X[nb_train_samples:]

    ytrain = Y[0:nb_train_samples]
    yval   = Y[nb_train_samples:]

    return xtrain, ytrain, xval, yval

