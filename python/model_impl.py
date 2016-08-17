import cPickle as pkl

import tensorflow as tf

from model import *


class opt_property:
    def __init__(self, name, learning_rate):
        self.name = name
        self.learning_rate = learning_rate


def init_var_map(init_actions, init_path=None, stddev=0.01, minval=-0.01, maxval=0.01):
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
                var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=stddev, dtype=dtype),
                                                dtype=dtype)
            elif init_method == 'uniform':
                var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=minval, maxval=maxval, dtype=dtype),
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


def get_optimizer(opt_algo, learning_rate, loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def get_l1_loss(weights):
    return tf.reduce_sum(tf.abs(weights))


def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights


class tf_classifier(classifier):
    def __init__(self, name, eval_metric, num_class, init_actions, init_path=None, stddev=0.01, minval=-0.01,
                 maxval=0.01, **argv):
        classifier.__init__(self, name, eval_metric, num_class)
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.x = tf.sparse_placeholder(tf.float32)
            self.y_true = tf.placeholder(tf.float32)
            self.drops = tf.placeholder(tf.float32)
            self.vars = init_var_map(init_actions, init_path, stddev=stddev, minval=minval, maxval=maxval)
            self.__sess = tf.Session()
            self.y, self.y_prob, self.loss, self.optimizer = self.build_graph(**argv)
            tf.initialize_all_variables().run(session=self.__sess)

    def __del__(self):
        self.__sess.close()

    def build_graph(self, **argv):
        pass

    # def get_graph(self):
    #     return self.__graph
    #
    # def get_sess(self):
    #     return self.__sess

    def run(self, fetches, feed_dict=None):
        return self.__sess.run(fetches, feed_dict)

    def train(self, indices, values, shape, labels, drops=0):
        _, l, y, y_prob = self.run(fetches=[self.optimizer, self.loss, self.y, self.y_prob],
                                   feed_dict={self.x: (indices, values, shape), self.y_true: labels,
                                              self.drops: drops})
        return l, y, y_prob

    def predict(self, indices, values, shape, drops=0):
        return self.run([self.y, self.y_prob], feed_dict={self.x: (indices, values, shape), self.drops: drops})

    def dump(self):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(self.get_bin_path(), 'wb'))
        print 'model dumped at', self.get_bin_path()


class tf_classifier_multi(tf_classifier):
    def __init__(self, name, eval_metric, num_class, num_input, init_actions, init_path=None, stddev=0.01, minval=-0.01,
                 maxval=0.01, **argv):
        classifier.__init__(self, name, eval_metric, num_class)
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.x = [tf.sparse_placeholder(tf.float32) for i in range(num_input)]
            self.y_true = tf.placeholder(tf.float32)
            self.drops = tf.placeholder(tf.float32)
            self.vars = init_var_map(init_actions, init_path, stddev=stddev, minval=minval, maxval=maxval)
            self.__sess = tf.Session()
            self.y, self.y_prob, self.loss, self.optimizer = self.build_graph(**argv)
            tf.initialize_all_variables().run(session=self.__sess)


# class logistic_regression(tf_classifier):
#     def __init__(self, name, eval_metric, num_class, input_space, opt_algo, learning_rate, l1_w=0, l2_w=0, l2_b=0):
#         tf_classifier.__init__(self, name, eval_metric, num_class,
#                                [('w', [input_space, num_class], 'normal', tf.float32),
#                                 ('b', [num_class], 'zero', tf.float32)],
#                                opt_algo=opt_algo, learning_rate=learning_rate, l1_w=l1_w, l2_w=l2_w, l2_b=l2_b)
#
#     def build_graph(self, opt_algo, learning_rate, l1_w, l2_w, l2_b):
#         tf_classifier.build_graph(self)
#         w = self.vars['w']
#         b = self.vars['b']
#         y = tf.sparse_tensor_dense_matmul(self.x, w) + b
#         y_prob, loss = get_loss(self.get_eval_metric(), y, self.y_true)
#         loss += l1_w * get_l1_loss(w) + l2_w * tf.nn.l2_loss(w) + l2_b * tf.nn.l2_loss(b)
#         optimizer = get_optimizer(opt_algo, learning_rate, loss)
#         return y, y_prob, loss, optimizer


class factorization_machine(tf_classifier):
    def __init__(self, name, eval_metric, num_class, input_space, factor_order, opt_algo, learning_rate, l1_w=0, l1_v=0,
                 l2_w=0, l2_v=0, l2_b=0, ):
        tf_classifier.__init__(self, name, eval_metric, num_class,
                               init_actions=[('w', [input_space, num_class], 'normal', tf.float32),
                                             ('v', [input_space, factor_order * num_class], 'normal', tf.float32),
                                             ('b', [num_class], 'zero', tf.float32)],
                               factor_order=factor_order, opt_algo=opt_algo, learning_rate=learning_rate,
                               l1_w=l1_w, l1_v=l1_v, l2_w=l2_w, l2_v=l2_v, l2_b=l2_b)

    def build_graph(self, factor_order, opt_algo, learning_rate, l1_w, l1_v, l2_w, l2_v, l2_b):
        x_square = tf.SparseTensor(self.x.indices, tf.square(self.x.values), self.x.shape)
        w = self.vars['w']
        v = self.vars['v']
        b = self.vars['b']
        y = tf.sparse_tensor_dense_matmul(self.x, w) + b
        y += tf.reduce_sum(tf.reshape(
            tf.square(tf.sparse_tensor_dense_matmul(self.x, v)) - tf.sparse_tensor_dense_matmul(x_square, tf.square(v)),
            [-1, factor_order, self.get_num_class()]), reduction_indices=[1])
        y_prob, loss = get_loss(self.get_eval_metric(), y, self.y_true)
        loss += l1_w * get_l1_loss(w) + l1_v * get_l1_loss(v) + l2_w * tf.nn.l2_loss(w) + l2_v * tf.nn.l2_loss(
            v) + l2_b * tf.nn.l2_loss(b)
        optimizer = get_optimizer(opt_algo, learning_rate, loss)
        return y, y_prob, loss, optimizer


class multi_layer_perceptron(tf_classifier):
    def __init__(self, name, eval_metric, layer_sizes, layer_activates, opt_algo, learning_rate):
        init_actions = []
        for i in range(len(layer_sizes) - 1):
            init_actions.append(('w%d' % i, [layer_sizes[i], layer_sizes[i + 1]], 'normal', tf.float32))
            init_actions.append(('b%d' % i, [layer_sizes[i + 1]], 'zero', tf.float32))
        print init_actions
        tf_classifier.__init__(self, name, eval_metric, layer_sizes[-1],
                               init_actions=init_actions,
                               layer_activates=layer_activates,
                               opt_algo=opt_algo,
                               learning_rate=learning_rate)

    def build_graph(self, layer_activates, opt_algo, learning_rate):
        w0 = self.vars['w0']
        b0 = self.vars['b0']
        # self.dropouts = tf.placeholder(dtype=tf.float32, shape=[None])
        l = tf.nn.dropout(activate(tf.sparse_tensor_dense_matmul(self.x, w0) + b0, layer_activates[0]),
                          keep_prob=self.drops[0])
        print 0, layer_activates[0]
        for i in range(1, len(self.vars) / 2):
            print i, layer_activates[i]
            wi = self.vars['w%d' % i]
            bi = self.vars['b%d' % i]
            l = tf.nn.dropout(activate(tf.matmul(l, wi) + bi, layer_activates[i]), keep_prob=self.drops[i])
        y_prob, loss = get_loss(self.get_eval_metric(), l, self.y_true)
        optimizer = get_optimizer(opt_algo, learning_rate, loss)
        return l, y_prob, loss, optimizer


class multiplex_neural_network(tf_classifier_multi):
    def __init__(self, name, eval_metric, layer_sizes, layer_activates, opt_algo, learning_rate):
        init_actions = []
        for i in range(len(layer_sizes[0])):
            init_actions.append(('w0_%d' % i, [layer_sizes[0][i], layer_sizes[1]], 'normal', tf.float32))
            init_actions.append(('b0_%d' % i, [layer_sizes[1]], 'zero', tf.float32))
        init_actions.append(('w1', [layer_sizes[1] * len(layer_sizes[0]), layer_sizes[2]], 'normal', tf.float32))
        init_actions.append(('b1', [layer_sizes[2]], 'zero', tf.float32))
        print init_actions
        for i in range(2, len(layer_sizes) - 1):
            init_actions.append(('w%d' % i, [layer_sizes[i], layer_sizes[i + 1]], 'normal', tf.float32))
            init_actions.append(('b%d' % i, [layer_sizes[i + 1]], 'zero', tf.float32))
        tf_classifier_multi.__init__(self, name, eval_metric, layer_sizes[-1], len(layer_sizes[0]),
                                     init_actions=init_actions,
                                     num_input=len(layer_sizes[0]),
                                     layer_activates=layer_activates,
                                     opt_algo=opt_algo,
                                     learning_rate=learning_rate)

    def build_graph(self, num_input, layer_activates, opt_algo, learning_rate):
        w0 = [self.vars['w0_%d' % i] for i in range(num_input)]
        b0 = [self.vars['b0_%d' % i] for i in range(num_input)]
        l = tf.nn.dropout(
            activate(tf.concat(1, [tf.sparse_tensor_dense_matmul(self.x[i], w0[i]) + b0[i] for i in range(num_input)]),
                     layer_activates[0]), self.drops[0])
        for i in range(1, len(self.vars) / 2 - num_input):
            wi = self.vars['w%d' % i]
            bi = self.vars['b%d' % i]
            l = tf.nn.dropout(activate(tf.matmul(l, wi) + bi, layer_activates[i]), keep_prob=self.drops[i])
        y_prob, loss = get_loss(self.get_eval_metric(), l, self.y_true)
        optimizer = get_optimizer(opt_algo, learning_rate, loss)
        return l, y_prob, loss, optimizer
