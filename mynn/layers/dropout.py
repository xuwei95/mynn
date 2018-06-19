import numpy as np

class DropOut(object):
    def __init__(self, shape , keep_prob):
        shape = [int(x) for x in shape]
        self.input_shape = shape

    def forward(self, x):
        pass

    def gradient(self, eta):
        pass

    def backward(self, alpha=0.00001):
        pass

