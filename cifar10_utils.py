import numpy as np

def load_pickle(f):
    from six.moves import cPickle as pickle
    return  pickle.load(f, encoding='latin1')

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10():
    """ load all of cifar """
    import urllib.request, tarfile, os
    xs, ys, root = [], [], os.getcwd()+'/cifar-10-batches-py'
    if not os.path.isdir(root):
        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "CIFAR10.tar.gz"
        urllib.request.urlretrieve(url, filename)
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()
        os.remove(filename)
    for b in range(1, 6):
        X, Y = load_CIFAR_batch(os.path.join(root, 'data_batch_%d' % (b, )))
        xs.append(X)
        ys.append(Y)
    Xtr, Ytr = np.concatenate(xs), np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(root, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
