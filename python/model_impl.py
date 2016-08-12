import cPickle as pkl

import tensorflow as tf

from model import *


def init_var_map(init_actions, init_path=None):
    if init_path is not None:
        var_map = pkl.load(open(init_path, 'rb'))
        print 'init variable from', init_path
    else:
        var_map = {}
    for var_name, var_shape, init_method, dtype in init_actions:
        if var_name in var_map:
            print var_name, 'already exists'
        else:
            if init_method == 'zero':
                var_map[var_name] = tf.Variable(tf.zeros(var_shape, dtype=dtype), dtype=dtype)
            elif init_method == 'one':
                var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype), dtype=dtype)
            elif init_method == 'normal':
                var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=0.01, dtype=dtype),
                                                dtype=dtype)
            elif init_method == 'uniform':
                var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=-0.01, maxval=0.01, dtype=dtype),
                                                dtype=dtype)
            else:
                print 'BadParam: init method', init_method
    return var_map


def get_loss(loss_func, y, y_true):
    if loss_func == 'sigmoid_log_loss':
        y_prob = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_true))
    elif loss_func == 'softmax_log_loss':
        y_prob = tf.nn.softmax(y)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_true))
    else:
        y_prob = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_true))
    return y_prob, loss


def get_optimizer(optimizer, learning_rate, loss):
    if optimizer == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif optimizer == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif optimizer == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif optimizer == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif optimizer == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


class tf_model:
    def __init__(self, output_space, init_actions, init_path=None, **argv):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.index_holder = tf.placeholder(tf.int64, [None, 2])
            self.value_holder = tf.placeholder(tf.float32, [None])
            self.shape_holder = tf.placeholder(tf.int64, [2])
            self.label_holder = tf.placeholder(tf.float32, [None, output_space])
            self.vars = init_var_map(init_actions, init_path)
            self.__sess = tf.Session()
            self.y, self.y_prob, self.loss, self.optimizer = self.build_graph(**argv)
            tf.initialize_all_variables().run(session=self.__sess)

    def __del__(self):
        self.__sess.close()

    def build_graph(self, **argv):
        pass

    def get_graph(self):
        return self.__graph

    def get_sess(self):
        return self.__sess

    def run(self, fetches, feed_dict=None):
        return self.__sess.run(fetches, feed_dict)


class logistic_regression(classifier, tf_model):
    def __init__(self, name, eval_metric, num_class, input_space, l1_alpha, l2_lambda, optimizer, learning_rate):
        classifier.__init__(self, name, eval_metric, num_class)
        tf_model.__init__(self, num_class,
                          [('w', [input_space, num_class], 'normal', tf.float32),
                           ('b', [num_class], 'zero', tf.float32)],
                          l1_alpha=l1_alpha,
                          l2_lambda=l2_lambda,
                          optimizer=optimizer,
                          learning_rate=learning_rate)

    def build_graph(self, l1_alpha=None, l2_lambda=None, optimizer=None, learning_rate=None):
        tf_model.build_graph(self)
        x = tf.SparseTensor(self.index_holder, self.value_holder, self.shape_holder)
        y = tf.sparse_tensor_dense_matmul(x, self.vars['w']) + self.vars['b']
        y_prob, loss = get_loss(self.get_eval_metric(), y, self.label_holder)
        loss += l1_alpha * (tf.reduce_sum(tf.abs(self.vars['w'])) + tf.reduce_sum(tf.abs(self.vars['b'])))
        loss += l2_lambda * (tf.nn.l2_loss(self.vars['w']) + tf.nn.l2_loss(self.vars['b']))
        optimizer = get_optimizer(optimizer, learning_rate, loss)
        return y, y_prob, loss, optimizer

    def train(self, indices=None, values=None, shape=None, labels=None):
        if indices is None:
            _, l, y, y_prob = self.run(fetches=[self.optimizer, self.loss, self.y, self.y_prob])
        else:
            _, l, y, y_prob = self.run(fetches=[self.optimizer, self.loss, self.y, self.y_prob],
                                       feed_dict={self.index_holder: indices, self.value_holder: values,
                                                  self.shape_holder: shape, self.label_holder: labels})
        return l, y, y_prob

    def predict(self, indices, values, shape):
        return self.run([self.y, self.y_prob], feed_dict={self.index_holder: indices, self.value_holder: values,
                                                          self.shape_holder: shape})

    def dump(self):
        var_map = {}
        for name, var in self.vars():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(self.get_bin_path(), 'wb'))
        print 'model dumped at', self.get_bin_path()

# class factorization_machine(classifier, tf_model):
#     def __init__(self, name, eval_metric, num_class, input_space, l2_lambda, optimizer, learning_rate):
#         classifier.__init__(self, name, eval_metric, num_class)
#         output_space = self.get_num_class()
#         with self.get_graph().as_default():
#             print ''
