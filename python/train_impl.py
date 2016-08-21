import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss

import feature
import utils

VERSION = None
BOOSTER = None
DATASET = None
PATH_TRAIN = None
PATH_TEST = None
PATH_TRAIN_TRAIN = None
PATH_TRAIN_VALID = None
TAG = None
PATH_MODEL_LOG = None
PATH_MODEL_BIN = None
PATH_MODEL_DUMP = None
PATH_SUBMISSION = None
RANDOM_STATE = 0

SPACE = None
RANK = None
SIZE = None
NUM_CLASS = None
SUB_FEATURES = None
SUB_SPACES = None
SUB_RANKS = None


def init_constant(dataset, booster, version, random_state=0):
    global DATASET, BOOSTER, VERSION, PATH_TRAIN, PATH_TEST, TAG, PATH_MODEL_LOG, PATH_MODEL_BIN, PATH_MODEL_DUMP, \
        PATH_SUBMISSION, RANDOM_STATE, SPACE, RANK, SIZE, NUM_CLASS, PATH_TRAIN_TRAIN, PATH_TRAIN_VALID, \
        SUB_FEATURES, SUB_SPACES, SUB_RANKS
    DATASET = dataset
    BOOSTER = booster
    VERSION = version
    PATH_TRAIN = '../input/' + DATASET + '.train'
    PATH_TEST = '../input/' + DATASET + '.test'
    PATH_TRAIN_TRAIN = PATH_TRAIN + '.train'
    PATH_TRAIN_VALID = PATH_TRAIN + '.valid'
    TAG = '%s_%s_%d' % (DATASET, BOOSTER, VERSION)
    PATH_MODEL_LOG = '../model/' + TAG + '.log'
    PATH_MODEL_BIN = '../model/' + TAG + '.model'
    PATH_MODEL_DUMP = '../model/' + TAG + '.dump'
    PATH_SUBMISSION = '../output/' + TAG + '.submission'
    print TAG
    RANDOM_STATE = random_state
    fea_tmp = feature.multi_feature(name=DATASET)
    fea_tmp.load_meta()
    SPACE = fea_tmp.get_space()
    RANK = fea_tmp.get_rank()
    SIZE = fea_tmp.get_size()
    NUM_CLASS = 12
    if fea_tmp.load_meta_extra():
        SUB_FEATURES = fea_tmp.get_sub_features()
        SUB_SPACES = fea_tmp.get_sub_spaces()
        SUB_RANKS = fea_tmp.get_sub_ranks()
    print 'feature space: %d, rank: %d, size: %d, num class: %d' % (SPACE, RANK, SIZE, NUM_CLASS)


def write_log(log_str):
    global PATH_MODEL_LOG
    with open(PATH_MODEL_LOG, 'a') as fout:
        fout.write(log_str)


def tune_rdforest(Xtrain, train_labels, Xvalid, valid_labels, n_estimators=10, max_depth=None, max_features='auto'):
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=8, max_depth=max_depth, max_features=max_features)
    clf.fit(Xtrain, train_labels)
    train_pred = clf.predict(Xtrain)
    train_score = log_loss(train_labels, train_pred)
    valid_pred = clf.predict(Xvalid)
    valid_score = log_loss(valid_labels, valid_pred)
    return train_score, valid_score


def train_rdforest(Xtrain, train_labels, Xtest, n_estimators=10, max_depth=None, max_features='auto'):
    # TODO
    # global NUM_CLASS
    # clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=8, max_depth=max_depth, max_features=max_features)
    # clf.fit(Xtrain, train_labels)
    # test_pred = clf.predict_proba(Xtest)
    # make_submission(test_pred)
    pass


def ensemble_rdforest(Xtrain, train_labels, Xvalid, valid_labels, n_estimators, max_depth, max_features,
                      dtest=None):
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=4, max_depth=max_depth,
                                 max_features=max_features)
    clf.fit(Xtrain, train_labels)
    train_pred = clf.predict_proba(Xtrain)
    train_score = log_loss(train_labels, train_pred)
    valid_pred = clf.predict_proba(Xvalid)
    valid_score = log_loss(valid_labels, valid_pred)
    print 'n_estimators', n_estimators, 'max_depth', max_depth, 'max_features', max_features, train_score, valid_score
    test_pred = clf.predict_proba(dtest)
    fea_out = utils.make_feature_model_output(TAG, [train_pred, valid_pred, test_pred], NUM_CLASS, dump=False)
    return fea_out


def tune_extratree(Xtrain, train_labels, Xvalid, valid_labels, n_estimators=10, max_depth=None, max_features='auto'):
    clf = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=4, criterion='gini', max_depth=max_depth,
                               max_features=max_features)
    clf.fit(Xtrain, train_labels)
    train_pred = clf.predict(Xtrain)
    train_score = log_loss(train_labels, train_pred)
    valid_pred = clf.predict(Xvalid)
    valid_score = log_loss(valid_labels, valid_pred)
    return train_score, valid_score


def ensemble_gblinear(dtrain, dvalid, gblinear_alpha=0, gblinear_lambda=0, gblinear_lambda_bias=0, verbose_eval=True,
                      early_stopping_rounds=50, dtest=None):
    global BOOSTER, RANDOM_STATE
    num_boost_round = 1000

    params = {
        'booster': BOOSTER,
        'silent': 1,
        'num_class': 12,
        'lambda': gblinear_lambda,
        'lambda_bias': gblinear_lambda_bias,
        'alpha': gblinear_alpha,
        'objective': 'multi:softprob',
        'seed': RANDOM_STATE,
        'eval_metric': 'mlogloss',
    }

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval)
    train_pred = bst.predict(dtrain)
    train_score = log_loss(dtrain.get_label(), train_pred)
    valid_pred = bst.predict(dvalid)
    valid_score = log_loss(dvalid.get_label(), valid_pred)
    test_pred = bst.predict(dtest)
    fea_out = utils.make_feature_model_output(TAG, [train_pred, valid_pred, test_pred], NUM_CLASS, dump=False)
    return fea_out


def ensemble_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree, gbtree_lambda=1, gbtree_alpha=0,
                    gamma=0, min_child_weight=1, max_delta_step=0, verbose_eval=False, early_stopping_rounds=50,
                    dtest=None):
    global BOOSTER, RANDOM_STATE
    num_boost_round = 2000

    params = {
        "booster": BOOSTER,
        "silent": 1,
        "num_class": 12,
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lambda": gbtree_lambda,
        "alpha": gbtree_alpha,
        "gamma": gamma,
        "min_child_weight": min_child_weight,
        "max_delta_step": max_delta_step,
        "objective": "multi:softprob",
        "seed": RANDOM_STATE,
        "eval_metric": "mlogloss",
    }

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval)

    train_pred = bst.predict(dtrain, ntree_limit=bst.best_iteration)
    valid_pred = bst.predict(dvalid, ntree_limit=bst.best_iteration)
    test_pred = bst.predict(dtest, ntree_limit=bst.best_iteration)
    fea_out = utils.make_feature_model_output(TAG, [train_pred, valid_pred, test_pred], NUM_CLASS, dump=False)
    return fea_out


def random_sample(train_data, number):
    train_indices, train_values, train_labels = train_data
    random_indices = np.random.randint(0, len(train_indices) - 1, number)
    sample_indices = train_indices[random_indices]
    sample_values = train_values[random_indices]
    sample_labels = train_labels[random_indices]
    return sample_indices, sample_values, sample_labels


def ensemble_model(train_data, valid_data, test_data, model_list):
    global BOOSTER, NUM_CLASS
    train_indices, train_values, train_labels = train_data
    valid_indices, valid_values, valid_labels = valid_data
    test_indices, test_values, test_labels = test_data

    train_csr = utils.libsvm_2_csr_matrix(train_indices, train_values)
    train_labels_groupid = utils.label_2_group_id(train_labels, NUM_CLASS)
    d_train = xgb.DMatrix(train_csr, label=train_labels_groupid)

    valid_csr = utils.libsvm_2_csr_matrix(valid_indices, valid_values)
    valid_labels_groupid = utils.label_2_group_id(valid_labels, NUM_CLASS)
    d_valid = xgb.DMatrix(valid_csr, label=valid_labels_groupid)

    test_csr = utils.libsvm_2_csr_matrix(test_indices, test_values)
    d_test = xgb.DMatrix(test_csr, label=utils.label_2_group_id(test_labels, NUM_CLASS))

    features_for_ensemble = []
    for model in model_list:
        for i in range(model_list[model]):
            print model
            if model == 'gblinear':
                BOOSTER = 'gblinear'
                # sample_data = random_sample(train_data, len(train_indices))
                # sample_indices, sample_values, sample_labels = sample_data
                feature_model = ensemble_gblinear(d_train, d_valid, gblinear_alpha=0, gblinear_lambda=10,
                                                  verbose_eval=True, early_stopping_rounds=8, dtest=d_test)
                features_for_ensemble.append(feature_model)
            elif model == 'gbtree':
                BOOSTER = 'gbtree'
                # sample_data = random_sample(train_data, len(train_indices))
                # sample_indices, sample_values, sample_labels = sample_data
                # train_csr_indices, train_csr_values, train_csr_shape = ti.libsvm_2_csr(sample_indices, sample_values)
                # train_csr = csr_matrix((train_csr_values, (train_csr_indices[:, 0], train_csr_indices[:, 1])),
                #                        shape=train_csr_shape)
                # sample_labels = ti.label_2_group_id(sample_labels, num_class=12)
                # d_train = xgb.DMatrix(train_csr, label=sample_labels)
                feature_model = ensemble_gbtree(d_train, d_valid, 0.1, 3, 0.8, 0.6, verbose_eval=True,
                                                early_stopping_rounds=20, dtest=d_test)
                features_for_ensemble.append(feature_model)
            elif model == 'mlp':
                layer_sizes = [SPACE, 100, NUM_CLASS]
                layer_activates = ['relu', None]
                drops = [0.5, 1]
                num_round = 1000
                opt_algo = 'gd'
                learning_rate = 0.2
                # sample_data = random_sample(train_data, len(train_indices))

                feature_model = ensemble_multi_layer_perceptron(train_data, valid_data, layer_sizes, layer_activates,
                                                                opt_algo=opt_algo, learning_rate=learning_rate,
                                                                drops=drops, num_round=num_round, batch_size=10000,
                                                                early_stopping_round=10, verbose=True, save_log=False,
                                                                test_data=test_data)
                features_for_ensemble.append(feature_model)

            elif model == 'randomforest':

                n_estimators = 1000
                max_depth = 40
                max_features = 0.1
                feature_model = ensemble_rdforest(train_csr, train_labels_groupid, valid_csr, valid_labels_groupid,
                                                  n_estimators, max_depth, max_features, test_csr)
                features_for_ensemble.append(feature_model)
    return features_for_ensemble


if __name__ == '__main__':
    init_constant(dataset='concat_6', booster=None, version=0)
    train_data = utils.read_feature(open(PATH_TRAIN_TRAIN), -1, NUM_CLASS)
    valid_data = utils.read_feature(open(PATH_TRAIN_VALID), -1, NUM_CLASS)
    train_indices, train_values, train_labels = train_data
    valid_indices, valid_values, valid_labels = valid_data
    train_csr = utils.libsvm_2_csr_matrix(train_indices, train_values)
    train_labels_groupid = utils.label_2_group_id(train_labels, NUM_CLASS)
    valid_csr = utils.libsvm_2_csr_matrix(valid_indices, valid_values)
    valid_labels_groupid = utils.label_2_group_id(valid_labels, NUM_CLASS)

    n_estimators = 200
    # max_depth = 40
    # max_features = 0.1
    # for n_estimators in [1000, 2000, 3000]:
    #     for max_depth in [40, 50 ,60 ,70]:
    #         for max_features in [0.05, 0.1, 0.2]:
    clf = ExtraTreesClassifier(n_estimators=n_estimators, criterion='gini', n_jobs=4)
    # max_depth=max_depth, max_features=max_features)

    clf.fit(train_csr, train_labels_groupid)
    train_pred = clf.predict_proba(train_csr)
    train_score = log_loss(train_labels, train_pred)
    valid_pred = clf.predict_proba(valid_csr)
    valid_score = log_loss(valid_labels, valid_pred)
    print 'n_estimators', n_estimators, train_score, valid_score
