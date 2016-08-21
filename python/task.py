import time
import xgboost as xgb
import feature
import utils
import numpy as np
from model_impl import GBLinear, GBTree, MultiLayerPerceptron, MultiplexNeuralNetwork


class Task:
    def __init__(self, dataset, booster, version,
                 eval_metric='softmax_log_loss',
                 random_state=0,
                 num_class=12,
                 train_size=59716,
                 valid_size=14929,
                 test_size=112071):
        self.dataset = dataset
        self.booster = booster
        self.version = version
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.path_train = '../input/' + dataset + '.train'
        self.path_test = '../input/' + dataset + '.test'
        self.path_train_train = self.path_train + '.train'
        self.path_train_valid = self.path_train + '.valid'
        self.tag = '%s_%s_%d' % (dataset, booster, version)
        print 'initializing task', self.tag
        self.path_log = '../model/' + self.tag + '.log'
        self.path_bin = '../model/' + self.tag + '.bin'
        self.path_dump = '../model/' + self.tag + '.dump'
        self.path_submission = '../output/' + self.tag + '.submission'
        fea_tmp = feature.multi_feature(name=dataset)
        fea_tmp.load_meta()
        self.space = fea_tmp.get_space()
        self.rank = fea_tmp.get_rank()
        self.size = fea_tmp.get_size()
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.num_class = num_class
        if fea_tmp.load_meta_extra():
            self.sub_features = fea_tmp.get_sub_features()
            self.sub_spaces = fea_tmp.get_sub_spaces()
            self.sub_ranks = fea_tmp.get_sub_ranks()
        print 'feature space: %d, rank: %d, size: %d, num class: %d' % (
            self.space, self.rank, self.size, self.num_class)

    def __write_log(self, log_str):
        with open(self.path_log, 'a') as fout:
            fout.write(log_str)

    def __load_data(self, path, batch_size=-1, num_class=None):
        if self.booster in {'gblinear', 'gbtree'}:
            return xgb.DMatrix(path)
        elif self.booster in {'mlp', 'mnn'}:
            if num_class is None:
                num_class = self.num_class
            return utils.read_feature(open(path), batch_size, num_class)

    def tune(self, dtrain=None, dvalid=None, params=None, batch_size=None, num_round=None, early_stop_round=None,
             verbose=True, save_log=True, save_model=False, dtest=None, save_feature=False):
        if dtrain is None:
            dtrain = self.__load_data(self.path_train_train)
        if dvalid is None:
            dvalid = self.__load_data(self.path_train_valid)
        if save_feature and dtest is None:
            dtest = self.__load_data(self.path_test)
        if self.booster == 'gblinear':
            print 'params [batch_size, save_log] will not be used'
            model = GBLinear(self.tag, self.eval_metric, self.space, self.num_class,
                             num_round=num_round,
                             early_stop_round=early_stop_round,
                             verbose=verbose,
                             **params)
        elif self.booster == 'gbtree':
            print 'params [batch_size, save_log] will not be used'
            model = GBTree(self.tag, self.eval_metric, self.space, self.num_class,
                           num_round=num_round,
                           early_stop_round=early_stop_round,
                           verbose=verbose,
                           **params)
        elif self.booster == 'mlp':
            model = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.num_class,
                                         batch_size=batch_size,
                                         num_round=num_round,
                                         early_stop_round=early_stop_round,
                                         verbose=verbose,
                                         save_log=save_log,
                                         **params)
            model.compile()
        elif self.booster == 'mnn':
            model = MultiplexNeuralNetwork(self.tag, self.eval_metric, self.sub_spaces, self.num_class,
                                           batch_size=batch_size,
                                           num_round=num_round,
                                           early_stop_round=early_stop_round,
                                           verbose=verbose,
                                           save_log=save_log,
                                           **params)
            model.compile()

        start_time = time.time()
        model.train(dtrain, dvalid)
        print 'time elapsed: %f' % (time.time() - start_time)

        if save_model:
            model.dump()
        if save_feature:
            train_pred = model.predict(dtrain)
            valid_pred = model.predict(dtest)
            test_pred = model.predict(dtest)
            utils.make_feature_model_output(self.tag, [train_pred, valid_pred, test_pred], self.num_class)

    def train(self, dtrain=None, dtest=None, params=None, batch_size=None, num_round=None, verbose=True,
              save_model=False, save_submission=True):
        if dtrain is None:
            dtrain = self.__load_data(self.path_train)
        if dtest is None:
            dtest = self.__load_data(self.path_test)
        if self.booster == 'gblinear':
            print 'param [batch_size] will not be used'
            model = GBLinear(self.tag, self.eval_metric, self.space, self.num_class,
                             num_round=num_round,
                             verbose=verbose,
                             **params)
        elif self.booster == 'gbtree':
            print 'param [batch_size] will not be used'
            model = GBTree(self.tag, self.eval_metric, self.space, self.num_class,
                           num_round=num_round,
                           verbose=verbose,
                           **params)
        elif self.booster == 'mlp':
            model = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.num_class,
                                         batch_size=batch_size,
                                         num_round=num_round,
                                         verbose=verbose,
                                         **params)
            model.compile()
        elif self.booster == 'mnn':
            model = MultiplexNeuralNetwork(self.tag, self.eval_metric, self.sub_spaces, self.num_class,
                                           batch_size=batch_size,
                                           num_round=num_round,
                                           verbose=verbose,
                                           **params)
            model.compile()
        start_time = time.time()
        model.train(dtrain)
        print 'time elapsed: %f' % (time.time() - start_time)

        if save_model:
            model.dump()
        if save_submission:
            test_pred = model.predict(dtest)
            utils.make_submission(self.path_submission, test_pred)

    def predict_mlpmodel(self, data=None, params=None, batch_size=None):
        model = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.num_class,
                                     batch_size=batch_size,
                                     **params)
        model.compile()
        data_pred = model.predict(data)
        # values = data_pred
        # indices = np.zeros_like(values, dtype=np.int64) + range(self.num_class)
        # fea_pred = feature.multi_feature(name=self.tag, dtype='f', space=self.num_class, rank=self.num_class,
        #                                  size=len(indices))
        # fea_pred.set_value(indices, values)
        # fea_pred.dump()
        utils.make_feature_model_output(self.tag, [data_pred], self.num_class, dump=True)
