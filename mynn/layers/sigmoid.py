import numpy as np

class Sigmoid(object):
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape
        self.layertype = 'sigmoid'
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def forward(self, x):
        self.x = x
        return self.sigmoid(x)

    def gradient(self, eta):
        self.eta = eta
        self.eta = self.eta * (1 - self.eta)
        return self.eta