import numpy as np
from mynn.layers.conv import Conv2D
from mynn.layers.fc import FullyConnect
from mynn.layers.pooling import MaxPooling
from mynn.layers.softmax import Softmax
from mynn.layers.relu import Relu
from mynn.dataset import mnist
from mynn.model import model
import time
images, labels = mnist.load_mnist('./data/mnist')
test_images, test_labels = mnist.load_mnist('./data/mnist', 't10k')
batch_size = 64
conv1 = Conv2D([batch_size, 28, 28, 1], 32, 5, 1)
relu1 = Relu(conv1.output_shape)
pool1 = MaxPooling(relu1.output_shape)
conv2 = Conv2D(pool1.output_shape, 64, 5, 1)
relu2 = Relu(conv2.output_shape)
pool2 = MaxPooling(relu2.output_shape)
fc1 = FullyConnect(pool2.output_shape, 1024)
relu3=Relu(fc1.output_shape)
fc2=FullyConnect(relu3.output_shape,10)
sf = Softmax(fc2.output_shape)
model=model.model()
model.add(conv1)
model.add(relu1)
model.add(pool1)
model.add(conv2)
model.add(relu2)
model.add(pool2)
model.add(fc1)
model.add(relu3)
model.add(fc2)
model.add(sf)
try:
    model.load('cnn.pkl')
except:
    pass
def test(num):
    n=0
    for i in range(num):
        x=test_images[i:i+1].reshape([1, 28, 28, 1])
        y=test_labels[i:i+1]
        a=model.predict(x)
        a=np.argmax(a[0])
        if a==y[0]:
            n+=1
    return n/num
for epoch in range(100):
    learning_rate = 1e-5
    for i in range(images.shape[0] // batch_size):
        img = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = labels[i * batch_size:(i + 1) * batch_size]
        loss=model.fit(img,label,alpha=learning_rate)
        if i%100==0:
            acc = test(100)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'  epoch:%s---step:%s---loss:%.5f----acc:%s'%(epoch,i,loss/batch_size,acc))
            model.save('cnn.pkl')
    acc = test(len(test_images))
    print('test acc：%s'%acc)
