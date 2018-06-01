from mynn.dataset import mnist
import numpy as np
from mynn.layers.fc import FullyConnect
from mynn.layers.softmax import Softmax
from mynn.layers.relu import Relu
import time
images, labels = mnist.load_mnist('./data/mnist')
test_images, test_labels = mnist.load_mnist('./data/mnist', 't10k')
batch_size = 64

fc1=FullyConnect([batch_size,784],40)
relu=Relu(fc1.output_shape)
fc = FullyConnect(relu.output_shape, 10)
sf = Softmax(fc.output_shape)


for epoch in range(1):
    learning_rate = 1e-5

    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0
    for i in range(images.shape[0] // batch_size):
        img = images[i * batch_size:(i + 1) * batch_size]
        label = labels[i * batch_size:(i + 1) * batch_size]
        fc1_out=fc1.forward(img)
        relu_out=relu.forward(fc1_out)
        fc_out = fc.forward(relu_out)
        batch_loss += sf.cal_loss(fc_out, np.array(label))
        train_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == label[j]:
                batch_acc += 1
                train_acc += 1
        fc1.gradient(relu.gradient(fc.gradient(sf.gradient())))

        if i % 1 == 0:
            fc1.backward(alpha=learning_rate)
            fc.backward(alpha=learning_rate)

            if i % 10 == 0:
                print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
                      "  epoch: %d ,  batch: %5d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (epoch,
                                                                                                 i, batch_acc / float(
                          batch_size), batch_loss / batch_size, learning_rate))


            batch_loss = 0
            batch_acc = 0


    print (time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime()) + "  epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (
            epoch, train_acc / float(images.shape[0]), train_loss / images.shape[0]))

    # validation
    for i in range(test_images.shape[0] // batch_size):
        img = test_images[i * batch_size:(i + 1) * batch_size]
        label = test_labels[i * batch_size:(i + 1) * batch_size]
        fc1_out = fc1.forward(img)
        relu_out = relu.forward(fc1_out)
        fc_out = fc.forward(relu_out)
        val_loss += sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == label[j]:
                val_acc += 1

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (
        epoch, val_acc / float(test_images.shape[0]), val_loss / test_images.shape[0]))