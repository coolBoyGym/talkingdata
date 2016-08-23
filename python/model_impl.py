import cPickle as pkl

import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import log_loss

import utils
from model import *


class GBLinear(Classifier):
    def __init__(self, name, eval_metric, input_spaces, num_class,
                 num_round=10, early_stop_round=None, verbose=True,
                 gblinear_alpha=0, gblinear_lambda=0, random_state=0):
        Classifier.__init__(self, name, eval_metric, input_spaces, None, num_class)
        self.params = {
            'booster': 'gblinear',
            'silent': 1,
            'num_class': num_class,
            'lambda': gblinear_lambda,
            'alpha': gblinear_alpha,
            'objective': 'multi:softprob',
            'seed': random_state,
            'eval_metric': 'mlogloss',
        }
        self.bst = None
        self.num_round = num_round
        self.early_stop_round = early_stop_round
        self.verbose = verbose

    def train(self, dtrain, dvalid=None):
        if dvalid is not None:
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.bst = xgb.train(self.params, dtrain,
                                 num_boost_round=self.num_round,
                                 early_stopping_rounds=self.early_stop_round,
                                 evals=watchlist,
                                 verbose_eval=self.verbose)
            train_score = log_loss(dtrain.get_label(), self.predict(dtrain))
            valid_score = log_loss(dvalid.get_label(), self.predict(dvalid))
            print '[-1]\ttrain: %f\tvalid: %f' % (train_score, valid_score)
            return train_score, valid_score
        else:
            watchlist = [(dtrain, 'train')]
            self.bst = xgb.train(self.params, dtrain,
                                 num_boost_round=self.num_round,
                                 evals=watchlist,
                                 verbose_eval=self.verbose)
            train_pred = self.predict(dtrain)
            train_score = log_loss(dtrain.get_label(), train_pred)
            print '[-1]\ttrain: %f' % train_score
            return train_score

    def predict(self, data):
        return self.bst.predict(data)

    def dump(self):
        self.bst.save_model(self.get_bin_path())
        self.bst.dump_model(self.get_file_path())
        print 'model dumped at', self.get_bin_path(), self.get_file_path()


class GBTree(Classifier):
    def __init__(self, name, eval_metric, input_spaces, num_class, num_round=10, early_stop_round=None, verbose=True,
                 eta=0.1, max_depth=3, subsample=0.7, colsample_bytree=0.7, gbtree_alpha=0, gbtree_lambda=0,
                 random_state=0):
        Classifier.__init__(self, name, eval_metric, input_spaces, None, num_class)
        self.params = {
            "booster": 'gbtree',
            "silent": 1,
            "num_class": num_class,
            "eta": eta,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "lambda": gbtree_lambda,
            "alpha": gbtree_alpha,
            "objective": "multi:softprob",
            "seed": random_state,
            "eval_metric": "mlogloss",
        }
        self.bst = None
        self.num_round = num_round
        self.early_stop_round = early_stop_round
        self.verbose = verbose

    def train(self, dtrain, dvalid=None):
        if dvalid is not None:
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            self.bst = xgb.train(self.params, dtrain,
                                 num_boost_round=self.num_round,
                                 early_stopping_rounds=self.early_stop_round,
                                 evals=watchlist,
                                 verbose_eval=self.verbose)
            train_score = log_loss(dtrain.get_label(), self.predict(dtrain))
            valid_score = log_loss(dvalid.get_label(), self.predict(dvalid))
            print '[-1]\ttrain: %f\tvalid: %f' % (train_score, valid_score)
            return train_score, valid_score
        else:
            watchlist = [(dtrain, 'train')]
            self.bst = xgb.train(self.params, dtrain,
                                 num_boost_round=self.num_round,
                                 evals=watchlist,
                                 verbose_eval=self.verbose)
            train_pred = self.predict(dtrain)
            train_score = log_loss(dtrain.get_label(), train_pred)
            print '[-1]\ttrain: %f' % train_score
            return train_score

    def predict(self, data):
        return self.bst.predict(data, ntree_limit=self.bst.best_iteration)

    def dump(self):
        self.bst.save_model(self.get_bin_path())
        self.bst.dump_model(self.get_file_path())
        print 'model dumped at', self.get_bin_path(), self.get_file_path()


class TFClassifier(Classifier):
    def __init__(self, name, eval_metric, input_spaces, input_types, num_class,
                 batch_size=None, num_round=None, early_stop_round=None, verbose=True, save_log=True):
        Classifier.__init__(self, name, eval_metric, input_spaces, input_types, num_class)
        self.__graph = None
        self.__sess = None
        self.x = None
        self.y_true = None
        self.drops = None
        self.vars = None
        self.y = None
        self.y_prob = None
        self.loss = None
        self.optimizer = None
        self.init_path = None
        self.init_actions = None
        self.layer_drops = None
        self.batch_size = batch_size
        self.num_round = num_round
        self.early_stop_round = early_stop_round
        self.verbose = verbose
        self.save_log = save_log

    def __del__(self):
        self.__sess.close()

    def build_graph(self):
        pass

    def run(self, fetches, feed_dict=None):
        return self.__sess.run(fetches, feed_dict)

    def compile(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.x = utils.init_input_units(self.get_input_spaces(), self.get_input_types())
            self.y_true = tf.placeholder(tf.float32)
            self.drops = tf.placeholder(tf.float32)
            self.vars = utils.init_var_map(self.init_actions, self.init_path)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.__sess = tf.Session(config=config)
            self.build_graph()
            tf.initialize_all_variables().run(session=self.__sess)

    def __train_batch(self, batch_data, labels):
        feed_dict = {self.y_true: labels, self.drops: self.layer_drops}
        utils.init_feed_dict(self.get_input_types(), batch_data, self.x, feed_dict)
        _, l, y_prob = self.__sess.run(fetches=[self.optimizer, self.loss, self.y_prob], feed_dict=feed_dict)
        return l, y_prob

    def __train_round_csr(self, dtrain):
        csr_mats, labels = dtrain
        batch_size = self.batch_size
        input_types = self.get_input_types()
        if batch_size == -1:
            input_data = utils.feature_slice_inputs(input_types, csr_mats, 0, batch_size)
            loss, y_prob = self.__train_batch(input_data, labels)
            return loss, y_prob
        loss = []
        y_prob = []
        for i in range(len(labels) / batch_size + 1):
            batch_i = utils.feature_slice_inputs(input_types, csr_mats, i * batch_size, batch_size)
            labels_i = labels[i * batch_size: (i + 1) * batch_size]
            batch_loss, batch_y_prob = self.__train_batch(batch_i, labels_i)
            loss.append(batch_loss)
            y_prob.extend(batch_y_prob)
        return np.array(loss), np.array(y_prob)

    def train(self, dtrain, dvalid=None):
        train_labels = dtrain[-1]
        train_scores = []
        if dvalid is None:
            for i in range(self.num_round):
                train_loss, train_y_prob = self.__train_round_csr(dtrain)
                train_score = log_loss(train_labels, train_y_prob)
                if self.save_log:
                    self.write_log('%d\t%f\t%f\n' % (i, train_loss.mean(), train_score))
                if self.verbose:
                    print '[%d]\tloss: %f\ttrain_score: %f\t' % (i, train_loss.mean(), train_score)
                train_scores.append(train_score)
            print '[-1]\ttrain_score: %f' % train_scores[-1]
            return train_scores[-1]
        valid_labels = dvalid[-1]
        valid_scores = []
        for i in range(self.num_round):
            train_loss, train_y_prob = self.__train_round_csr(dtrain)
            train_score = log_loss(train_labels, train_y_prob)
            valid_y_prob = self.predict(dvalid[0])
            valid_score = log_loss(valid_labels, valid_y_prob)
            train_scores.append(train_score)
            valid_scores.append(valid_score)
            if self.save_log:
                log_str = '%d\t%f\t%f\t%f\n' % (i, train_loss.mean(), train_score, valid_score)
                self.write_log(log_str)
            if self.verbose:
                print '[%d]\tloss: %f \ttrain_score: %f\tvalid_score: %f' % \
                      (i, train_loss.mean(), train_score, valid_score)
            if utils.check_early_stop(valid_scores, self.early_stop_round, 'no_decrease'):
                best_iteration = i + 1 - self.early_stop_round
                print 'best iteration:\n[%d]\ttrain_score: %f\tvalid_score: %f' % (
                    best_iteration, train_scores[best_iteration], valid_scores[best_iteration])
                break
        print '[-1]\ttrain_score: %f\tvalid_score: %f' % (train_scores[-1], valid_scores[-1])
        return train_scores[-1], valid_scores[-1]

    def __predict_batch(self, batch_data):
        feed_dict = {self.drops: [1] * len(self.layer_drops)}
        utils.init_feed_dict(self.get_input_types(), batch_data, self.x, feed_dict)
        y_prob = self.run(self.y_prob, feed_dict=feed_dict)
        return y_prob

    def predict(self, csr_mat):
        input_spaces = self.get_input_spaces()
        input_types = self.get_input_types()
        batch_size = self.batch_size
        if batch_size == -1:
            input_data = utils.feature_slice_inputs(input_types, csr_mat, 0, batch_size)
            y_prob = self.__predict_batch(input_data)
            return y_prob
        y_prob = []
        if utils.check_type(input_spaces, 'int'):
            data_size = csr_mat.shape[0]
        else:
            data_size = csr_mat[0].shape[0]
        for i in range(data_size / batch_size + 1):
            batch_i = utils.feature_slice_inputs(input_types, csr_mat, i * batch_size, batch_size)
            y_prob_i = self.__predict_batch(batch_i)
            y_prob.extend(y_prob_i)
        return np.array(y_prob)

    def dump(self):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(self.get_bin_path(), 'wb'))
        print 'model dumped at', self.get_bin_path()


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


class FactorizationMachine(TFClassifier):
    def __init__(self, name, eval_metric, input_spaces, input_types, num_class, batch_size=None, num_round=None,
                 early_stop_round=None, verbose=True, save_log=True,
                 factor_order=None, opt_algo=None, learning_rate=None,
                 l1_w=0, l1_v=0, l2_w=0, l2_v=0, l2_b=0):
        TFClassifier.__init__(self, name, eval_metric, input_spaces, input_types, num_class,
                              batch_size, num_round, early_stop_round, verbose, save_log)
        self.init_actions = [('w', [input_spaces, num_class], 'normal', tf.float32),
                             ('v', [input_spaces, factor_order * num_class], 'normal', tf.float32),
                             ('b', [num_class], 'zero', tf.float32)],
        self.factor_order = factor_order
        self.opt_algo = opt_algo
        self.learning_rate = learning_rate,
        self.l1_w = l1_w
        self.l1_v = l1_v
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.l2_b = l2_b

    def build_graph(self):
        w = self.vars['w']
        v = self.vars['v']
        b = self.vars['b']
        if self.get_input_types() == 'sparse':
            x_square = tf.SparseTensor(self.x.indices, tf.square(self.x.values), self.x.shape)
            self.y = tf.sparse_tensor_dense_matmul(self.x, w) + b
            self.y += tf.reduce_sum(tf.reshape(
                tf.square(tf.sparse_tensor_dense_matmul(self.x, v)) - \
                tf.sparse_tensor_dense_matmul(x_square, tf.square(v)),
                [-1, self.factor_order, self.get_num_class()]), reduction_indices=[1])
        else:
            self.y = tf.matmul(self.x, w) + b
            self.y += tf.reduce_sum(
                tf.reshape(tf.square(tf.matmul(self.x, v)) - tf.matmul(tf.square(self.x), tf.square(v)),
                           [-1, self.factor_order, self.get_num_class()]), reduction_indices=[1])
        self.y_prob, self.loss = utils.get_loss(self.get_eval_metric(), self.y, self.y_true)
        self.loss += self.l1_w * utils.get_l1_loss(w) + self.l1_v * utils.get_l1_loss(v) + \
                     self.l2_w * tf.nn.l2_loss(w) + self.l2_v * tf.nn.l2_loss(v) + self.l2_b * tf.nn.l2_loss(b)
        self.optimizer = utils.get_optimizer(self.opt_algo, self.learning_rate, self.loss)


class MultiLayerPerceptron(TFClassifier):
    def __init__(self, name, eval_metric, input_spaces, input_types, num_class,
                 batch_size=None, num_round=None, early_stop_round=None, verbose=True, save_log=True,
                 layer_sizes=None, layer_activates=None, layer_drops=None, layer_inits=None, init_path=None,
                 opt_algo=None, learning_rate=None):
        TFClassifier.__init__(self, name, eval_metric, input_spaces, input_types, num_class, batch_size=batch_size,
                              num_round=num_round, early_stop_round=early_stop_round, verbose=verbose,
                              save_log=save_log)
        self.init_actions = []
        for i in range(len(layer_sizes) - 1):
            self.init_actions.append(('w%d' % i, [layer_sizes[i], layer_sizes[i + 1]], layer_inits[i][0], tf.float32))
            self.init_actions.append(('b%d' % i, [layer_sizes[i + 1]], layer_inits[i][1], tf.float32))
        self.layer_inits = layer_inits
        self.init_path = init_path
        self.layer_activates = layer_activates
        self.layer_drops = layer_drops
        self.opt_algo = opt_algo
        self.learning_rate = learning_rate

    def build_graph(self):
        w0 = self.vars['w0']
        b0 = self.vars['b0']
        l = utils.embed_input_units(self.get_input_types(), self.x, w0, b0)
        l = utils.activate(l, self.layer_activates[0])
        l = tf.nn.dropout(l, keep_prob=self.drops[0])
        for i in range(1, len(self.vars) / 2):
            wi = self.vars['w%d' % i]
            bi = self.vars['b%d' % i]
            l = utils.activate(tf.matmul(l, wi) + bi, self.layer_activates[i])
            l = tf.nn.dropout(l, keep_prob=self.drops[i])
        self.y = l
        self.y_prob, self.loss = utils.get_loss(self.get_eval_metric(), l, self.y_true)
        self.optimizer = utils.get_optimizer(self.opt_algo, self.learning_rate, self.loss)


class MultiplexNeuralNetwork(TFClassifier):
    def __init__(self, name, eval_metric, input_spaces, input_types, num_class,
                 batch_size=None, num_round=None, early_stop_round=None, verbose=True, save_log=True,
                 layer_sizes=None, layer_activates=None, layer_drops=None, layer_inits=None, init_path=None,
                 opt_algo=None, learning_rate=None):
        TFClassifier.__init__(self, name, eval_metric, input_spaces, input_types, num_class, batch_size=batch_size,
                              num_round=num_round, early_stop_round=early_stop_round, verbose=verbose,
                              save_log=save_log)
        self.init_actions = []
        for i in range(len(layer_sizes[0])):
            layer_input = layer_sizes[0][i]
            layer_output = layer_sizes[1][i]
            self.init_actions.append(('w0_%d' % i, [layer_input, layer_output], layer_inits[0][0], tf.float32))
            self.init_actions.append(('b0_%d' % i, [layer_output], layer_inits[0][1], tf.float32))
        self.init_actions.append(('w1', [sum(layer_sizes[1]), layer_sizes[2]], layer_inits[1][0], tf.float32))
        self.init_actions.append(('b1', [layer_sizes[2]], layer_inits[1][1], tf.float32))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            self.init_actions.append(('w%d' % i, [layer_input, layer_output], layer_inits[i][0], tf.float32))
            self.init_actions.append(('b%d' % i, [layer_output], layer_inits[i][1], tf.float32))
        self.layer_inits = layer_inits
        self.init_path = init_path
        self.layer_activates = layer_activates
        self.layer_drops = layer_drops
        self.opt_algo = opt_algo
        self.learning_rate = learning_rate

    def build_graph(self):
        num_input = len(self.get_input_spaces())
        w0 = [self.vars['w0_%d' % i] for i in range(num_input)]
        b0 = [self.vars['b0_%d' % i] for i in range(num_input)]
        l = tf.concat(1, utils.embed_input_units(self.get_input_types(), self.x, w0, b0))
        l = tf.nn.dropout(utils.activate(l, self.layer_activates[0]), self.drops[0])
        for i in range(1, len(self.vars) / 2 - num_input + 1):
            wi = self.vars['w%d' % i]
            bi = self.vars['b%d' % i]
            l = tf.nn.dropout(utils.activate(tf.matmul(l, wi) + bi, self.layer_activates[i]), keep_prob=self.drops[i])
        self.y = l
        self.y_prob, self.loss = utils.get_loss(self.get_eval_metric(), l, self.y_true)
        self.optimizer = utils.get_optimizer(self.opt_algo, self.learning_rate, self.loss)

# class convolutional_neural_network(tf_classifier_multi):
#     def __init__(self, name, eval_metric, input_spaces, num_class, layer_sizes, layer_activates, kernel_sizes,
#                  opt_algo,
#                  learning_rate,
#                  layer_inits=None, init_path=None, layer_pools=None, layer_pool_strides=None):
#         if layer_inits is None:
#             layer_inits = [('normal', 'zero')] * (len(layer_sizes) - 1)
#         init_actions = []
#         for i in range(len(layer_sizes[0])):
#             init_actions.append(('w0_%d' % i, [layer_sizes[0][i], layer_sizes[1][i]], layer_inits[0][0], tf.float32))
#             init_actions.append(('b0_%d' % i, [layer_sizes[1][i]], layer_inits[0][1], tf.float32))
#         for i in range(len(kernel_sizes)):
#             init_actions.append(('k%d' % i, kernel_sizes[i], 'normal', tf.float32))
#         l1_input_w = len(layer_sizes[0])
#         l1_input_h = layer_sizes[1][0]
#         l1_input_c = 1
#         for i in range(len(kernel_sizes)):
#             w, h, ci, co = kernel_sizes[i]
#             l1_input_w -= (w - 1)
#             l1_input_h -= (h - 1)
#             l1_input_w /= layer_pool_strides[i][1]
#             l1_input_h /= layer_pool_strides[i][2]
#             l1_input_c = co
#         l1_input_size = l1_input_w * l1_input_h * l1_input_c
#         init_actions.append(('w1', [l1_input_size, layer_sizes[2]], layer_inits[1][0], tf.float32))
#         init_actions.append(('b1', [layer_sizes[2]], layer_inits[1][1], tf.float32))
#         for i in range(2, len(layer_sizes) - 1):
#             init_actions.append(('w%d' % i, [layer_sizes[i], layer_sizes[i + 1]], layer_inits[i][0], tf.float32))
#             init_actions.append(('b%d' % i, [layer_sizes[i + 1]], layer_inits[i][1], tf.float32))
#         print init_actions
#         tf_classifier_multi.__init__(self, name, eval_metric, input_spaces, num_class,
#                                      init_actions=init_actions,
#                                      init_path=init_path,
#                                      layer_activates=layer_activates,
#                                      layer_pools=layer_pools,
#                                      layer_pool_strides=layer_pool_strides,
#                                      opt_algo=opt_algo,
#                                      learning_rate=learning_rate, )
#
#     def build_graph(self, layer_activates, layer_pools, layer_pool_strides, opt_algo, learning_rate):
#         num_input = len(self.get_input_spaces())
#         w0 = [self.vars['w0_%d' % i] for i in range(num_input)]
#         b0 = [self.vars['b0_%d' % i] for i in range(num_input)]
#         l = [tf.reshape(tf.sparse_tensor_dense_matmul(self.x[i], w0[i]) + b0[i],
#                           [tf.shape(self.x[i])[0], 1, -1, 1]) for
#              i in range(num_input)]
#         l = activate(tf.concat(1, l), layer_activates[0])
#         for i in range(len(layer_pools)):
#             l = tf.nn.conv2d(l, self.vars['k%d' % i], strides=[1, 1, 1, 1], padding='VALID')
#             if layer_pools is not None:
#                 l = tf.nn.max_pool(l, layer_pools[i], strides=layer_pool_strides[i], padding='VALID')
#         l = tf.nn.dropout(l, self.drops[0])
#         l = tf.reshape(l, [tf.shape(self.x[0])[0], -1])
#         for i in range(1, (len(self.vars) - len(layer_pools)) / 2 - num_input + 1):
#             wi = self.vars['w%d' % i]
#             bi = self.vars['b%d' % i]
#             l = tf.nn.dropout(activate(tf.matmul(l, wi) + bi, layer_activates[i]), keep_prob=self.drops[i])
#         y_prob, loss = get_loss(self.get_eval_metric(), l, self.y_true)
#         optimizer = get_optimizer(opt_algo, learning_rate, loss)
#         return l, y_prob, loss, optimizer


# class text_convolutional_neural_network(tf_classifier):
#     def __init__(self, name, eval_metric, input_spaces, num_class, layer_sizes, layer_activates, kernel_depth,
#                   opt_algo,
#                  learning_rate,
#                  layer_inits=None, kernel_inits=None, init_path=None):
#         if layer_inits is None:
#             layer_inits = [('normal', 'zero')] * (len(layer_sizes) - 1)
#         init_actions = []
#         for i in range(len(layer_sizes[0])):
#             init_actions.append(('w0_%d' % i, [layer_sizes[0][i], layer_sizes[1][i]], layer_inits[0][0], tf.float32))
#             init_actions.append(('b0_%d' % i, [layer_sizes[1][i]], layer_inits[0][1], tf.float32))
#         for i in range(len(layer_sizes[0])):
#             init_actions.append(
#                 ('k%d' % i, [i + 1, layer_sizes[1][0], 1, kernel_depth], kernel_inits[i][0], tf.float32))
#             init_actions.append(('kb%d' % i, [kernel_depth], kernel_inits[i][1], tf.float32))
#         init_actions.append(('w1', [len(layer_sizes[0]) * kernel_depth, layer_sizes[2]],
#                               layer_inits[1][0], tf.float32))
#         init_actions.append(('b1', [layer_sizes[2]], layer_inits[1][1], tf.float32))
#         for i in range(2, len(layer_sizes) - 1):
#             init_actions.append(('w%d' % i, [layer_sizes[i], layer_sizes[i + 1]], layer_inits[i][0], tf.float32))
#             init_actions.append(('b%d' % i, [layer_sizes[i + 1]], layer_inits[i][1], tf.float32))
#         print init_actions
#         tf_classifier_multi.__init__(self, name, eval_metric, input_spaces, num_class,
#                                      init_actions=init_actions,
#                                      init_path=init_path,
#                                      layer_activates=layer_activates,
#                                      kernel_depth=kernel_depth,
#                                      opt_algo=opt_algo,
#                                      learning_rate=learning_rate, )
#
#     def build_graph(self, layer_activates, kernel_depth, opt_algo, learning_rate):
#         num_input = len(self.get_input_spaces())
#         w0 = [self.vars['w0_%d' % i] for i in range(num_input)]
#         b0 = [self.vars['b0_%d' % i] for i in range(num_input)]
#         l = [tf.reshape(tf.sparse_tensor_dense_matmul(self.x[i], w0[i]) + b0[i],
#                           [tf.shape(self.x[i])[0], 1, -1, 1]) for
#              i in range(num_input)]
#         l = activate(tf.concat(1, l), layer_activates[0])
#         print 0, layer_activates[0]
#         l_arr = []
#         for i in range(len(self.x)):
#             li = tf.nn.conv2d(l, self.vars['k%d' % i], strides=[1, 1, 1, 1], padding='VALID')
#             li = tf.nn.bias_add(li, self.vars['kb%d' % i])
#             li = tf.nn.max_pool(li, [1, len(self.x) - i, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
#             l_arr.append(li)
#         l = tf.concat(1, l_arr)
#         l = tf.reshape(l, [-1, len(self.x) * kernel_depth])
#         l = tf.nn.dropout(l, self.drops[0])
#         for i in range(1, len(self.vars) / 2 - len(self.x) - num_input + 1):
#             wi = self.vars['w%d' % i]
#             bi = self.vars['b%d' % i]
#             print i, layer_activates[i]
#             l = tf.nn.dropout(activate(tf.matmul(l, wi) + bi, layer_activates[i]), keep_prob=self.drops[i])
#         y_prob, loss = get_loss(self.get_eval_metric(), l, self.y_true)
#         optimizer = get_optimizer(opt_algo, learning_rate, loss)
#         return l, y_prob, loss, optimizer
#         # return None, None, None, None
