import numpy as np
from mynn.layers.fc import FullyConnect
from mynn.layers.softmax import Softmax
from mynn.layers.relu import Relu
from mynn.dataset import mnist
from mynn.model import model
images, labels = mnist.load_mnist('./data/mnist')
test_images, test_labels = mnist.load_mnist('./data/mnist', 't10k')
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
    model.load('model.pkl')
except:
    pass
def test():
    n=0
    for i in range(len(test_images)-1):
        x=test_images[i:i+1]
        y=test_labels[i:i+1]
        a=model.predict(x)
        a=np.argmax(a[0])
        if a==y[0]:
            n+=1
    return n/len(test_images)
for epoch in range(100):
    learning_rate = 1e-5
    for i in range(images.shape[0] // batch_size):
        img = images[i * batch_size:(i + 1) * batch_size]
        label = labels[i * batch_size:(i + 1) * batch_size]
        loss=model.fit(img,label)
        if i%100==0:
            acc = test()
            print('epoch:%s---step:%s---loss:%.5f----acc:%s'%(epoch,i,loss/batch_size,acc))
            model.save('model.pkl')
