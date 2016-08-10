import cPickle as pkl

import tensorflow as tf

from model import *


def init_var_map(init_actions, init_path=None):
    if init_path is not None:
        var_map = pkl.load(open(init_path, 'rb'))
        print 'init variable from', init_path
    else:
        var_map = {}
    for var_name, var_shape, init_method in init_actions:
        if var_name in var_map:
            print var_name, 'already exists'
        else:
            if init_method == 'zero':
                var_map[var_name] = tf.zeros(var_shape)
            elif init_method == 'one':
                var_map[var_name] = tf.ones(var_shape)
            elif init_method == 'norm':
                var_map[var_name] = tf.random_normal(var_shape, mean=0.0, stddev=0.01)
            elif init_method == 'unif':
                var_map[var_name] = tf.random_uniform(var_shape, minval=-0.01, maxval=0.01)
            else:
                print 'BadParam: init method', init_method
    return var_map


def get_optimizer(optimizer, learning_rate, loss):
    if optimizer == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif optimizer == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


class logistic_regression(classifier):
    def init(self, input_space, input_rank, l2_lambda, optimizer, learning_rate, init_path=None):
        self.__graph = tf.Graph()
        output_space = self.get_num_class()
        with self.__graph.as_default():
            self.index_holder = tf.placeholder(tf.int32, [None, input_rank])
            self.value_holder = tf.placeholder(tf.float32, [None, input_rank])
            self.label_holder = tf.placeholder(tf.float32, [None, output_space])
            var_map = init_var_map([('w', [input_space, output_space], 'norm'),
                                    ('b', [output_space], 'zero')], init_path)
            self.w = tf.Variable(var_map['w'])
            self.b = tf.Variable(var_map['b'])
            self.x = tf.gather(self.w, self.index_holder)
            self.y = tf.mul(self.x,
                            tf.tile(tf.reshape(self.value_holder, [-1, input_rank, 1]), [1, 1, output_space])) + self.b
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.label_holder))
            # self.loss += l2_lambda * (tf.nn.l2_loss(self.w) + tf.nn.l2_loss(self.b))
            # self.optimizer = get_optimizer(optimizer, learning_rate, self.loss)
            # self.y_prob = tf.nn.softmax(self.y)
        with tf.Session(graph=self.__graph) as self.__sess:
            tf.initialize_all_variables().run(session=self.__sess)

    def train(self, indices, values, labels):
        _, l = self.__sess.run(fetches=[self.optimizer, self.loss],
                               feed_dict={self.index_holder: indices, self.value_holder: values,
                                          self.label_holder: labels})
        return l

    def predict(self, indices, values):
        return self.y_prob.eval(feed_dict={self.index_holder: indices, self.value_holder: values})

    def dump(self):
        var_map = {'w': self.w.eval(), 'b': self.b.eval()}
        pkl.dump(var_map, open(self.get_bin_path(), 'wb'))
        print 'model dumped at', self.get_bin_path()
