import numpy as np
from functools import reduce

class FullyConnect(object):
    def __init__(self, shape, output_num=2):
        shape = [int(x) for x in shape]
        self.input_shape = shape
        self.batchsize = shape[0]
        self.layertype='fc'
        input_len = reduce(lambda x, y: x * y, shape[1:])
        self.weights = np.random.randn(input_len, output_num)/100
        self.bias = np.random.randn(output_num)/100
        self.output_shape = [self.batchsize, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        if len(x.shape)>2:
            self.x = x.reshape([x.shape[0], -1])
        else:
            self.x=x
        output = np.dot(self.x, self.weights)+self.bias
        return output

    def gradient(self, eta):
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.bias.shape)

        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def backward(self, alpha=0.00001):
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
