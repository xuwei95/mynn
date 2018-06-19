from mynn.model import model
import cv2
import numpy as np
from mynn.dataset import mnist
model=model.model()
model.load('cnn.pkl')
images, labels = mnist.load_mnist('./data/mnist')
test_images, test_labels = mnist.load_mnist('./data/mnist', 't10k')
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
# import time
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# acc = test(len(test_images))
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print('test acc：%s'%acc)
def testimg(image):
    img=cv2.imread(image)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=img.reshape([1, 28, 28, 1])
    a=model.predict(img)
    a=np.argmax(a[0])
    return a
a=testimg('test.jpg')
print(a)
