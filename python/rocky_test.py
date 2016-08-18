import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split

import train_impl as ti

# def save_sparse_csr(filename, array):
#     np.savez(filename, data=array.data, indices=array.indices,
#              indptr=array.indptr, shape=array.shape)
def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
train_data_csr = load_sparse_csr('../input/bagofapps_train_csr.npz')
test_data_csr = load_sparse_csr('../input/bagofapps_test_csr.npz')
train_label = np.load('../input/bagofapps_train_label.npy')
X_train, X_valid, y_train, y_valid = train_test_split(train_data_csr, train_label, test_size=0.2, random_state=0)

ti.init_constant(dataset='concat_1', booster='multi_layer_perceptron', version=15, random_state=0)

if __name__ == '__main__':
    if ti.BOOSTER == 'gblinear':
        dtrain_train = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dtrain_valid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        # train_score, valid_score = ti.tune_gblinear(dtrain_train, dtrain_valid, 0, 10, verbose_eval=True)
        # train_score, valid_score = ti.tune_gblinear(dtrain_train, dtrain_valid,
        # gblinear_alpha=0, gblinear_lambda=10, verbose_eval=True,early_stopping_rounds=50, dtest=dtest)
        # print train_score, valid_score
        # ti.train_gblinear(dtrain, dtest, 0, 10, 0, 1)

        # for gblinear_alpha in [0,0.005, 0.01,0.015, 0.02, 0.03,0.05]:
        #     for gblinear_lambda in [1,2,5,7,8,9,10,12,15]:
        #         train_score, valid_score = ti.tune_gblinear(dtrain_train, dtrain_valid, gblinear_alpha=gblinear_alpha,
        #                                                     gblinear_lambda=gblinear_lambda, verbose_eval=False)
        #         print 'alpha', gblinear_alpha, 'lambda', gblinear_lambda, train_score, valid_score
    elif ti.BOOSTER == 'gbtree':
        # dtrain_train = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        # dtrain_valid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        # dtrain = xgb.DMatrix(ti.PATH_TRAIN)
        # dtest = xgb.DMatrix(ti.PATH_TEST)
        #
        train_indices, train_values, train_labels = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
        valid_indices, valid_values, valid_labels = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)

        X_train = ti.libsvm_2_csr_matrix(train_indices, train_values)
        X_valid = ti.libsvm_2_csr_matrix(valid_indices, valid_values)
        y_train = ti.label_2_group_id(train_labels)
        y_valid = ti.label_2_group_id(valid_labels)
        dtrain_train = xgb.DMatrix(X_train, label=y_train)
        dtrain_valid = xgb.DMatrix(X_valid, label=y_valid)
        # dtrain = xgb.DMatrix(train_data_csr, label=train_label)

        train_score, valid_score = ti.tune_gbtree(dtrain_train, dtrain_valid, 0.1, 3, 0.7, 0.8, verbose_eval=True)
        # train_score, valid_score = ti.tune_gbtree(dtrain_train, dtrain_valid, 0.05, 4, 0.7, 0.6,verbose_eval= True, dtest=dtest)
        # print train_score, valid_score

        # ti.train_gbtree(dtrain, dtest, 0.1, 3, 0.8, 0.6, 1, 0, 1040)

        # max_depth = 3
        # eta = 0.1
        # subsample = 0.7
        # colsample_bytree = 0.7
        # for max_depth in [3,4,5,6,7]:
        #    for eta in [0.05,0.06,0.07,0.09,0.1]:
        #        for subsample in [0.7,0.8,0.9]:
        #            for colsample_bytree in [0.5,0.6,0.7]:
        #                train_score, valid_score = ti.tune_gbtree(dtrain_train, dtrain_valid, eta, max_depth, subsample,
        #                                   colsample_bytree, False)
        #                print 'max_depth',max_depth,'eta', eta,'subsample', subsample,'colsample_bytree', colsample_bytree, train_score, valid_score
    elif ti.BOOSTER == 'rdforest':
        fin = open(ti.PATH_TRAIN_TRAIN, 'r')
        train_indices, train_values, train_shape, train_labels = ti.read_csr_feature(fin, -1)
        Xtrain = csr_matrix((train_values, (train_indices[:, 0], train_indices[:, 1])), shape=train_shape)
        fin = open(ti.PATH_TRAIN_VALID, 'r')
        valid_indices, valid_values, valid_shape, valid_labels = ti.read_csr_feature(fin, -1)
        Xvalid = csr_matrix((valid_values, (valid_indices[:, 0], valid_indices[:, 1])), shape=valid_shape)
        fin = open(ti.PATH_TRAIN, 'r')
        wtrain_indices, wtrain_values, wtrain_shape, wtrain_labels = ti.read_csr_feature(fin, -1)
        wXtrain = csr_matrix((wtrain_values, (wtrain_indices[:, 0], wtrain_indices[:, 1])), shape=wtrain_shape)
        fin = open(ti.PATH_TEST, 'r')
        test_indices, test_values, test_shape, test_labels = ti.read_csr_feature(fin, -1)
        Xtest = csr_matrix((test_values, (test_indices[:, 0], test_indices[:, 1])), shape=test_shape)

        # for n_estimators in [50,100,200,300,500]:
        #     train_score, valid_score = ti.tune_rdforest(Xtrain, train_labels, Xvalid, valid_labels, n_estimators=n_estimators)
        #     print 'n_estimators', n_estimators, train_score, valid_score
        train_score, valid_score = ti.tune_rdforest(Xtrain, train_labels, Xvalid, valid_labels, n_estimators=100,
                                                    max_features=None)
        print train_score, valid_score
        # ti.train_rdforest(wXtrain, wtrain_labels, Xtest, n_estimators=100)

    elif ti.BOOSTER == 'multi_layer_perceptron':
        # dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
        # dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
        # dtrain = ti.read_feature(open(ti.PATH_TRAIN), -1)
        # dtest = ti.read_feature(open(ti.PATH_TEST), -1)
        y_train = ti.group_id_2_label(y_train)
        # train_indices, train_values = ti.csr_matrix_2_libsvm(X_train)
        dtrain_train = X_train, y_train
        # valid_indices, valid_values = ti.csr_matrix_2_libsvm(X_valid)
        y_valid = ti.group_id_2_label(y_valid)
        dtrain_valid = X_valid, y_valid

        layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
        layer_activates = ['relu', None]
        drops = [0.5, 1]
        learning_rate = 0.2
        num_round = 470
        opt_algo = 'gd'

        # for n in [400, 300, 200, 100]:
        # for learning_rate in [0.5, 0.4, 0.3, 0.2, 0.1]:
        # for second_layer_num in [500, 800, 1500, 2000, 3000]:
        #     layer_sizes = [ti.SPACE, 800, second_layer_num, ti.NUM_CLASS]
        #     layer_sizes = [ti.SPACE, n, ti.NUM_CLASS]

        ti.tune_mlp_csr(dtrain_train, dtrain_valid, layer_sizes, layer_activates, opt_algo,
                                       learning_rate, drops, num_round=num_round, batch_size=10000,
                                       early_stopping_round=10, verbose=True, save_log=True)

        # opt_prop = opt_property('gd', learning_rate)
        # ti.train_multi_layer_perceptron(dtrain, dtest, layer_sizes, layer_activates, opt_algo, learning_rate, drops,
        #                                     num_round=num_round, batch_size=10000)
