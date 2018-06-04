import numpy as np

class Sigmoid(object):
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape
        self.layertype = 'sigmoid'
    def forward(self, x):
        self.x = x
        self.output=1.0/(1.0+np.exp(-self.x))
        return self.output

    def gradient(self, eta):
        self.eta=eta
        self.eta = self.eta*self.output*(1-self.output)
        return self.eta
