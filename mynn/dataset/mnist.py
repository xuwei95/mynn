import numpy as np
from mynn.util import np_utils
import struct
from glob import glob


def load_mnist(path, kind='train'):
    """Load MNIST dataset from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
def load_data():
    images, labels = load_mnist('./dataset/mnist')
    test_images, test_labels = load_mnist('./dataset/mnist', 't10k')
    y_train = np_utils.to_categorical(labels, num_classes=10)
    y_test = np_utils.to_categorical(test_labels, num_classes=10)
    X_train=images.reshape(60000,784)/255.0
    X_test=test_images.reshape(10000,784)/255.0
    return (X_train, y_train), (X_test, y_test)