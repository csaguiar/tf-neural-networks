import sys
import os
root_path = os.path.abspath(os.path.join(sys.path[0], os.pardir))
sys.path.append(root_path)

from tfneural.models import DeepNeuralNet
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    data_path = os.path.join(root_path, 'data/tmp/')
    print(data_path)
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    dnn = DeepNeuralNet(n_classes=10, n_inputs=784, hidden_layers=[500, 500, 500])
    sess = dnn.train(mnist.train.images, mnist.train.labels, 10, 100)
    acc = dnn.test(sess, mnist.test.images, mnist.test.labels)
    print('Accuracy', acc)
    sess.close()
