import numpy as np
from mynn.layers.fc import FullyConnect
from mynn.layers.softmax import Softmax
from mynn.layers.relu import Relu
from mynn.dataset import mnist
from mynn.model import model
from mynn.util import np_utils

images, labels = mnist.load_mnist('./data/mnist')
test_images, test_labels = mnist.load_mnist('./data/mnist', 't10k')
labels = np_utils.to_categorical(labels, num_classes=10)
test_labels = np_utils.to_categorical(test_labels, num_classes=10)
batch_size = 64
fc1=FullyConnect([batch_size,784],40)
relu=Relu(fc1.output_shape)
fc = FullyConnect(relu.output_shape, 10)
sf = Softmax(fc.output_shape)
model=model.model()
model.add(fc1)
model.add(relu)
model.add(fc)
model.add(sf)
try:
    model.load('fc.pkl')
except:
    pass
def test():
    n=0
    for i in range(len(test_images)-1):
        x=test_images[i:i+1]
        y=test_labels[i:i+1]
        y=np.argmax(y)
        a=model.predict(x)
        a=np.argmax(a[0])
        if a==y:
            n+=1
    return n/len(test_images)
learning_rate = 1e-5
learning_rate_decay = 0.001
for epoch in range(100):
    learning_rate = learning_rate * 1.0 / (1.0 + learning_rate_decay * epoch)
    for i in range(images.shape[0] // batch_size):
        img = images[i * batch_size:(i + 1) * batch_size]
        label = labels[i * batch_size:(i + 1) * batch_size]
        loss=model.fit(img,label,alpha=learning_rate)
        if i%100==0:
            acc = test()
            print('epoch:%s---step:%s---loss:%.5f----acc:%s---learning_rate:%s'%(epoch,i,loss/batch_size,acc,learning_rate))
            model.save('fc.pkl')
