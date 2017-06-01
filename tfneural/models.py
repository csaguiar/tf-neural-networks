import tensorflow as tf
import numpy as np


class DeepNeuralNet(object):
    """Define a new class"""

    def __init__(self, n_classes=None, n_inputs=None, hidden_layers=None):
        """Constructor for DeepNeuralNet"""
        self.n_classes = n_classes
        self.hidden_layers = []
        self.graph_layers = []
        self._current_batch_start = None
        self._current_batch_end = None
        n_connections = n_inputs

        self.x = tf.placeholder('float', [None, n_inputs])
        self.y = tf.placeholder('float')

        for n_nodes in hidden_layers:
            self.hidden_layers.append({
                'weights': tf.Variable(tf.random_normal([n_connections, n_nodes])),
                'biases': tf.Variable(tf.random_normal([n_nodes]))
            })
            n_connections = n_nodes

        self.output_layer = {
            'weights': tf.Variable(tf.random_normal([n_connections, n_classes])),
            'biases': tf.Variable(tf.random_normal([n_classes]))
        }

        self.nn_graph = self._generate_graph(self.x)

    def _generate_graph(self, train_data):
        """Generate graph to compute DNN"""

        feed_layer = train_data

        for layer in self.hidden_layers:
            feed_layer = tf.add(tf.matmul(feed_layer, layer['weights']), layer['biases'])
            feed_layer = tf.nn.relu(feed_layer)

        return tf.matmul(feed_layer, self.output_layer['weights']) + self.output_layer['biases']

    def train(self, train_x, train_y, total_epochs, batch_size, save_to=None):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.nn_graph, labels=self.y))

        optimizer = tf.train.AdamOptimizer().minimize(cost)
        if save_to:
            saver = tf.train.Saver()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(total_epochs):
            epoch_loss = 0
            batch = 0
            self._current_batch_start = 0

            while batch < train_x.shape[0]:
                self._current_batch_start = batch
                self._current_batch_end = batch + batch_size
                batch_x = np.array(train_x[self._current_batch_start:self._current_batch_end])
                batch_y = np.array(train_y[self._current_batch_start:self._current_batch_end])
                _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                epoch_loss += c
                batch += batch_size

            print('Epoch', epoch, 'completed out of', total_epochs, 'loss:', epoch_loss)

        if save_to:
            saver.save(sess, save_to)

        return sess

    def test(self, session, test_x, test_y):
        correct = tf.equal(tf.argmax(self.nn_graph, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        return accuracy.eval({self.x: test_x, self.y: test_y}, session=session)






