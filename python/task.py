import time

import numpy as np
import xgboost as xgb
from scipy.sparse import vstack
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

import feature
import utils
from model_impl import GBLinear, GBTree, MultiLayerPerceptron, MultiplexNeuralNetwork


class Task:
    def __init__(self, dataset, booster, version,
                 input_type=None,
                 sub_input_types=None,
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
        fea_tmp = feature.MultiFeature(name=dataset)
        fea_tmp.load_meta()
        self.space = fea_tmp.get_space()
        self.rank = fea_tmp.get_rank()
        self.input_type = input_type
        self.sub_input_types = sub_input_types
        if self.input_type is None:
            if self.space == self.rank:
                self.input_type = 'dense'
            else:
                self.input_type = 'sparse'
        self.size = fea_tmp.get_size()
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.num_class = num_class
        if fea_tmp.load_meta_extra():
            self.sub_features = fea_tmp.get_sub_features()
            self.sub_spaces = fea_tmp.get_sub_spaces()
            self.sub_ranks = fea_tmp.get_sub_ranks()
            if self.sub_input_types is None:
                self.sub_input_types = []
                for i in range(len(self.sub_features)):
                    if self.sub_spaces[i] == self.sub_ranks[i]:
                        self.sub_input_types.append('dense')
                    else:
                        self.sub_input_types.append('sparse')
        print 'feature space: %d, rank: %d, size: %d, num class: %d' % (
            self.space, self.rank, self.size, self.num_class)

    def upgrade_version(self, offset=1, version=None):
        if version is None:
            self.version += offset
        else:
            self.version = version
        self.tag = '%s_%s_%d' % (self.dataset, self.booster, self.version)
        print 'upgrading task version', self.tag
        self.path_log = '../model/' + self.tag + '.log'
        self.path_bin = '../model/' + self.tag + '.bin'
        self.path_dump = '../model/' + self.tag + '.dump'
        self.path_submission = '../output/' + self.tag + '.submission'

    def __write_log(self, log_str):
        with open(self.path_log, 'a') as fout:
            fout.write(log_str)

    def load_data(self, path, batch_size=-1, num_class=None):
        start_time = time.time()
        if num_class is None:
            num_class = self.num_class
        if self.booster in {'gblinear', 'gbtree'}:
            data = xgb.DMatrix(path)
        elif self.booster in {'mlp', 'net2net_mlp', 'predict'}:
            indices, values, labels = utils.read_feature(open(path), batch_size, num_class)
            data = [utils.libsvm_2_feature(indices, values, self.space, self.input_type), labels]
        elif self.booster in {'mnn'}:
            indices, values, labels = utils.read_feature(open(path), batch_size, num_class)
            split_indices, split_values = utils.split_feature(indices, values, self.sub_spaces)
            features = utils.libsvm_2_feature(split_indices, split_values, self.sub_spaces, self.sub_input_types)
            data = [features, labels]
        print 'load data', path, time.time() - start_time
        return data

    def tune(self, dtrain=None, dvalid=None, params=None, batch_size=None, num_round=None, early_stop_round=None,
             verbose=True, save_log=True, save_model=False, dtest=None, save_feature=False):
        if dtrain is None:
            dtrain = self.load_data(self.path_train_train)
        if dvalid is None:
            dvalid = self.load_data(self.path_train_valid)
        if save_feature and dtest is None:
            dtest = self.load_data(self.path_test)

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
            model = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.input_type, self.num_class,
                                         batch_size=batch_size,
                                         num_round=num_round,
                                         early_stop_round=early_stop_round,
                                         verbose=verbose,
                                         save_log=save_log,
                                         **params)
        elif self.booster == 'mnn':
            model = MultiplexNeuralNetwork(self.tag, self.eval_metric, self.sub_spaces, self.sub_input_types,
                                           self.num_class,
                                           batch_size=batch_size,
                                           num_round=num_round,
                                           early_stop_round=early_stop_round,
                                           verbose=verbose,
                                           save_log=save_log,
                                           **params)

        start_time = time.time()
        model.train(dtrain, dvalid)
        print 'training time elapsed: %f' % (time.time() - start_time)

        if save_model:
            model.dump()
        if save_feature:
            if self.booster in {'gbtree', 'gblinear'}:
                train_pred = model.predict(dtrain)
                valid_pred = model.predict(dvalid)
                test_pred = model.predict(dtest)
            else:
                train_pred = model.predict(dtrain[0])
                valid_pred = model.predict(dvalid[0])
                test_pred = model.predict(dtest[0])
            preds = np.vstack((valid_pred, train_pred, test_pred))
            utils.make_feature_model_output(self.tag, preds, self.num_class)
        return model

    def train(self, dtrain=None, dtest=None, params=None, batch_size=None, num_round=None, verbose=True,
              save_log=True, save_model=False, save_submission=True):
        if dtrain is None:
            dtrain = self.load_data(self.path_train)
        if dtest is None:
            dtest = self.load_data(self.path_test)
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
            model = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.input_type, self.num_class,
                                         batch_size=batch_size,
                                         num_round=num_round,
                                         verbose=verbose,
                                         save_log=save_log,
                                         **params)
        elif self.booster == 'mnn':
            model = MultiplexNeuralNetwork(self.tag, self.eval_metric, self.sub_spaces, self.sub_input_types,
                                           self.num_class,
                                           batch_size=batch_size,
                                           num_round=num_round,
                                           verbose=verbose,
                                           save_log=save_log,
                                           **params)

        start_time = time.time()
        model.train(dtrain)
        print 'time elapsed: %f' % (time.time() - start_time)

        if save_model:
            model.dump()
        if save_submission:
            test_pred = model.predict(dtest[0])
            utils.make_submission(self.path_submission, test_pred)

    def predict_mlpmodel(self, data=None, params=None, batch_size=None):
        data = utils.libsvm_2_csr(data[0], data[1], self.space)
        model = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.input_type, self.num_class,
                                     batch_size=batch_size,
                                     **params)
        data_pred = model.predict(data)
        utils.make_feature_model_output(self.tag, [data_pred], self.num_class)

    def kfold(self, n_fold=5, n_repeat=10, dtrain=None, dtest=None, params=None, batch_size=None, num_round=None,
              early_stop_round=None, verbose=True):
        if dtrain is None:
            dtrain = self.load_data(self.path_train)
        if dtest is None:
            dtest = self.load_data(self.path_test)
        train_data, train_labels = dtrain
        kf = KFold(train_data.shape[0], n_folds=n_fold, shuffle=True)
        for i in range(n_repeat):
            for train_indices, valid_indices in kf:
                X_train, y_train = train_data[train_indices], train_labels[train_indices]
                X_valid, y_valid = train_data[valid_indices], train_labels[valid_indices]
                model_i = self.tune([X_train, y_train], [X_valid, y_valid],
                                    params=params,
                                    batch_size=batch_size,
                                    num_round=num_round,
                                    early_stop_round=early_stop_round,
                                    verbose=verbose,
                                    save_log=False,
                                    save_model=False,
                                    dtest=None,
                                    save_feature=False)
                train_pred = model_i.predict(dtrain[0])
                test_pred = model_i.predict(dtest[0])
                pred_k = np.vstack((train_pred, test_pred))
                utils.make_feature_model_output(self.tag, [pred_k], self.num_class)
                self.upgrade_version()

    def net2net_mlp(self, dtrain=None, dvalid=None, params_1=None, params_2=None, batch_size=None, num_round=None,
                    early_stop_round=None,
                    verbose=True, save_log=True, save_model=False, split_cols=0):
        if dtrain is None:
            dtrain = self.load_data(self.path_train_train)
        if dvalid is None:
            dvalid = self.load_data(self.path_train_valid)

        model_1 = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.input_type, self.num_class,
                                       batch_size=batch_size,
                                       **params_1)

        model_2 = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.input_type, self.num_class,
                                       batch_size=batch_size,
                                       num_round=num_round,
                                       early_stop_round=early_stop_round,
                                       verbose=verbose,
                                       save_log=save_log,
                                       **params_2)

        train_split_index = utils.split_data_by_col(dtrain[0], self.space, self.sub_spaces, split_cols)
        dtrain_event_data = dtrain[0][train_split_index[0]]
        dtrain_event_label = dtrain[1][train_split_index[0]]
        dtrain_event = [dtrain_event_data, dtrain_event_label]
        dtrain_no_event_data = dtrain[0][train_split_index[1]]

        valid_split_index = utils.split_data_by_col(dvalid[0], self.space, self.sub_spaces, split_cols)
        dvalid_event_data = dvalid[0][valid_split_index[0]]
        dvalid_event_label = dvalid[1][valid_split_index[0]]
        dvalid_event = [dvalid_event_data, dvalid_event_label]
        dvalid_no_event_data = dvalid[0][valid_split_index[1]]

        start_time = time.time()
        model_2.train(dtrain_event, dvalid_event)
        print 'training time elapsed: %f' % (time.time() - start_time)

        train_pred_1 = model_1.predict(dtrain_no_event_data)
        train_pred_2 = model_2.predict(dtrain_event_data)
        train_pred = np.zeros([self.train_size, self.num_class])
        train_pred[train_split_index[0]] = train_pred_2
        train_pred[train_split_index[1]] = train_pred_1
        train_score = log_loss(dtrain[-1], train_pred)

        valid_pred_1 = model_1.predict(dvalid_no_event_data)
        valid_pred_2 = model_2.predict(dvalid_event_data)
        valid_pred = np.zeros([self.valid_size, self.num_class])
        valid_pred[valid_split_index[0]] = valid_pred_2
        valid_pred[valid_split_index[1]] = valid_pred_1
        valid_score = log_loss(dvalid[-1], valid_pred)

        print "final train_score:", train_score, "final valid_score:", valid_score

        if save_model:
            model_2.dump()
        return model_2

    def net2net_mlp_train(self, dtrain=None, dtest=None, params_1=None, params_2=None, batch_size=None, num_round=None,
                          verbose=True, save_log=True, save_model=False, split_cols=0, save_submission=True):
        if dtrain is None:
            dtrain = self.load_data(self.path_train)
        if dtest is None:
            dtest = self.load_data(self.path_test)
        model_1 = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.input_type, self.num_class,
                                       batch_size=batch_size,
                                       **params_1)
        model_2 = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.input_type, self.num_class,
                                       batch_size=batch_size,
                                       num_round=num_round,
                                       verbose=verbose,
                                       save_log=save_log,
                                       **params_2)

        train_split_index = utils.split_data_by_col(dtrain[0], self.space, self.sub_spaces, split_cols)
        dtrain_event_data = dtrain[0][train_split_index[0]]
        dtrain_event_label = dtrain[1][train_split_index[0]]
        dtrain_event = [dtrain_event_data, dtrain_event_label]
        dtrain_no_event_data = dtrain[0][train_split_index[1]]

        test_split_index = utils.split_data_by_col(dtest[0], self.space, self.sub_spaces, split_cols)
        dtest_event_data = dtest[0][test_split_index[0]]
        dtest_event_label = dtest[1][test_split_index[0]]
        dtest_event = [dtest_event_data, dtest_event_label]
        dtest_no_event_data = dtest[0][test_split_index[1]]

        start_time = time.time()
        model_2.train(dtrain_event)
        print 'time elapsed: %f' % (time.time() - start_time)

        if save_model:
            model_2.dump()
        if save_submission:
            test_pred_1 = model_1.predict(dtest_no_event_data)
            test_pred_2 = model_2.predict(dtest_event_data)
            test_pred = np.zeros([self.test_size, self.num_class], dtype=np.float32)
            test_pred[test_split_index[0]] = test_pred_2
            test_pred[test_split_index[1]] = test_pred_1
            print test_pred.shape
            print test_pred_1.shape
            print test_pred_2.shape

            utils.make_submission(self.path_submission, test_pred)

    def fea_net_mlp(self, fea_name, dtrain=None, dvalid=None, params_2=None, batch_size=None, num_round=None,
                    early_stop_round=None, verbose=True, save_log=True, save_model=False, split_cols=0):
        # fea_task = Task(fea_name, 'mlp', 0)
        # fea_dtrain = fea_task.load_data(fea_task.path_train_train)
        # fea_train_data, fea_train_label = fea_dtrain

        fea_path_train = '../input/' + fea_name + '.train'
        fea_path_test = '../input/' + fea_name + '.test'
        fea_path_train_train = fea_path_train + '.train'
        fea_path_train_valid = fea_path_train + '.valid'

        indices, values, labels = utils.read_feature(open(fea_path_train_train), batch_size, self.num_class)
        fea_train = utils.libsvm_2_feature(indices, values, 12, 'dense')
        indices, values, labels = utils.read_feature(open(fea_path_train_valid), batch_size, self.num_class)
        fea_valid = utils.libsvm_2_feature(indices, values, 12, 'dense')

        if dtrain is None:
            dtrain = self.load_data(self.path_train_train)
        if dvalid is None:
            dvalid = self.load_data(self.path_train_valid)

        model_2 = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.input_type, self.num_class,
                                       batch_size=batch_size,
                                       num_round=num_round,
                                       early_stop_round=early_stop_round,
                                       verbose=verbose,
                                       save_log=save_log,
                                       **params_2)

        train_split_index = utils.split_data_by_col(dtrain[0], self.space, self.sub_spaces, split_cols)
        dtrain_event_data = dtrain[0][train_split_index[0]]
        dtrain_event_label = dtrain[1][train_split_index[0]]
        dtrain_event = [dtrain_event_data, dtrain_event_label]
        dtrain_no_event_data = dtrain[0][train_split_index[1]]

        valid_split_index = utils.split_data_by_col(dvalid[0], self.space, self.sub_spaces, split_cols)
        dvalid_event_data = dvalid[0][valid_split_index[0]]
        dvalid_event_label = dvalid[1][valid_split_index[0]]
        dvalid_event = [dvalid_event_data, dvalid_event_label]
        dvalid_no_event_data = dvalid[0][valid_split_index[1]]

        start_time = time.time()
        model_2.train(dtrain_event, dvalid_event)
        print 'training time elapsed: %f' % (time.time() - start_time)

        train_pred_1 = fea_train[train_split_index[1]]
        train_pred_2 = model_2.predict(dtrain_event_data)
        train_pred = np.zeros([self.train_size, self.num_class])
        train_pred[train_split_index[0]] = train_pred_2
        train_pred[train_split_index[1]] = train_pred_1
        train_score = log_loss(dtrain[-1], train_pred)

        valid_pred_1 = fea_valid[valid_split_index[1]]
        valid_pred_2 = model_2.predict(dvalid_event_data)
        valid_pred = np.zeros([self.valid_size, self.num_class])
        valid_pred[valid_split_index[0]] = valid_pred_2
        valid_pred[valid_split_index[1]] = valid_pred_1
        valid_score = log_loss(dvalid[-1], valid_pred)

        print "final train_score:", train_score, "final valid_score:", valid_score

        if save_model:
            model_2.dump()
        return model_2

    def predict(self, data=None, model=None, params=None, batch_size=None, save_feature=False):
        if data is None:
            dtrain = self.load_data(self.path_train_train)
            dvalid = self.load_data(self.path_train_valid)
            dtest = self.load_data(self.path_test)
            data = vstack((dtrain[0], dvalid[0], dtest[0]))
        if model is None:
            model = MultiLayerPerceptron(self.tag, self.eval_metric, self.space, self.input_type, self.num_class,
                                         batch_size=batch_size,
                                         **params)
        preds = model.predict(data)

        if save_feature:
            utils.make_feature_model_output(self.tag, [preds], self.num_class)

        return preds
