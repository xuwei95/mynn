from mynn.model import model
import cv2
import numpy as np
model=model.model()
model.load('cnn.pkl')
def testimg(image):
    img=cv2.imread(image)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=img.reshape([1, 28, 28, 1])
    a=model.predict(img)
    a=np.argmax(a[0])
    return a
a=testimg('test.jpg')
print(a)
