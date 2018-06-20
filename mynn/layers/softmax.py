import numpy as np
class Softmax(object):
    def __init__(self, shape):
        self.softmax = np.zeros(shape)
        self.eta = np.zeros(shape)
        self.batchsize = shape[0]
        self.layertype = 'softmax'
    def cal_loss(self, prediction, label):
        self.label = label
        self.prediction = prediction
        self.predict(prediction)
        self.loss=0
        for i in range(prediction.shape[0]):
            self.loss+=np.log(np.sum(np.exp(prediction[i]))) - prediction[i, np.argmax(label[i])]
        return self.loss
    def predict(self, prediction):
        self.softmax = np.zeros(prediction.shape)
        for i in range(prediction.shape[0]):
            self.softmax[i] = np.exp(prediction[i]) / np.sum(np.exp(prediction[i]))
        return self.softmax
    def gradient(self):
        self.eta = self.softmax.copy()
        for i in range(self.prediction.shape[0]):
            self.eta[i, np.argmax(self.label[i])] -= 1
        return self.eta