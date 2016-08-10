import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

import feature

version = 1
booster = 'gbtree'

dataset = 'concat_6'
nfold = 5

path_train = '../input/' + dataset + '.train'
path_test = '../input/' + dataset + '.test'

tag = '%s_%s_%d' % (dataset, booster, version)
path_model_log = '../model/' + tag + '.log'
path_model_bin = '../model/' + tag + '.model'
path_model_dump = '../model/' + tag + '.dump'

path_submission = '../output/' + tag + '.submission'

random_state = 0


def tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda, verbose_eval):
    num_boost_round = 1000
    early_stopping_rounds = 50

    params = {
        'booster': booster,
        'silent': 1,
        'num_class': 12,
        'lambda': gblinear_lambda,
        'alpha': gblinear_alpha,
        'objective': 'multi:softprob',
        'seed': random_state,
        'eval_metric': 'mlogloss',
    }

    train_score = []
    valid_score = []
    for i in range(nfold):
        watchlist = [(dtrain[i], 'train'), (dvalid[i], 'eval')]
        bst = xgb.train(params, dtrain[i], num_boost_round, evals=watchlist,
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)

        train_pred = bst.predict(dtrain[i])
        train_score.append(log_loss(dtrain[i].get_label(), train_pred))

        valid_pred = bst.predict(dvalid[i])
        valid_score.append(log_loss(dvalid[i].get_label(), valid_pred))

    return train_score, valid_score


def tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree, verbose_eval):
    num_boost_round = 1000
    early_stopping_rounds = 30
    # print 'num_boost_round=', num_boost_round
    # print 'early_stopping_rounds', early_stopping_rounds

    params = {
        "booster": booster,
        "silent": 1,
        "num_class": 12,
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "objective": "multi:softprob",
        "seed": random_state,
        "eval_metric": "mlogloss",
    }

    train_score = []
    valid_score = []
    for i in range(nfold):
        watchlist = [(dtrain[i], 'train'), (dvalid[i], 'eval')]
        bst = xgb.train(params, dtrain[i], num_boost_round, evals=watchlist,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=verbose_eval)

        train_pred = bst.predict(dtrain[i], ntree_limit=bst.best_iteration)
        train_score.append(log_loss(dtrain[i].get_label(), train_pred))

        valid_pred = bst.predict(dvalid[i], ntree_limit=bst.best_iteration)
        valid_score.append(log_loss(dvalid[i].get_label(), valid_pred))

    return train_score, valid_score


def make_submission(test_pred):
    test_device_id = np.loadtxt('../data/raw/gender_age_test.csv', skiprows=1, dtype=np.int64)

    with open(path_submission, 'w') as fout:
        fout.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
        for i in range(len(test_device_id)):
            fout.write('%s,%s\n' % (test_device_id[i], ','.join(map(lambda d: str(d), test_pred[i]))))

    print path_submission


def train_gblinear(dtrain, dtest, num_boost_round, gblinear_alpha, gblinear_lambda):
    params = {
        'booster': booster,
        'silent': 1,
        'num_class': 12,
        'lambda': gblinear_lambda,
        'alpha': gblinear_alpha,
        'objective': 'multi:softprob',
        'seed': random_state,
        'eval_metric': 'mlogloss',
    }

    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, verbose_eval=True)

    bst.save_model(path_model_bin)
    bst.dump_model(path_model_dump)

    train_pred = bst.predict(dtrain)
    test_pred = bst.predict(dtest)

    make_submission(test_pred)
    fea_pred = feature.multi_feature(name=tag, dtype='f', space=12, rank=12, size=len(train_pred) + len(test_pred))
    indices = np.array([range(12)] * (len(train_pred) + len(test_pred)))
    values = np.vstack((train_pred, test_pred))

    fea_pred.set_value(indices, values)
    fea_pred.dump()


def train_gblinear_use_nfold(xgb_alpha=0.1, xgb_lambda=0.1, verbose_eval=False):
    dtrain = [xgb.DMatrix(path_train + '.%d.train' % i) for i in range(nfold)]
    dvalid = [xgb.DMatrix(path_train + '.%d.valid' % i) for i in range(nfold)]

    train_score, valid_score = tune_gblinear(dtrain, dvalid, xgb_alpha, xgb_lambda, verbose_eval)
    print np.mean(train_score), np.mean(valid_score)
    exit(0)


def train_gblinear_find_argument(argument_file_name):
    # argument_file_name = '../output/argument.concat_3.gblinear'
    dtrain = [xgb.DMatrix(path_train + '.%d.train' % i) for i in range(nfold)]
    dvalid = [xgb.DMatrix(path_train + '.%d.valid' % i) for i in range(nfold)]

    fout = open(argument_file_name, 'a')
    for gblinear_alpha in [0.01, 0.02]:
        print 'alpha', gblinear_alpha
        fout.write('alpha ' + str(gblinear_alpha) + '\n')
        for gblinear_lambda in [10, 20, 30, 40]:
            train_score, valid_score = tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda, False)
            print 'lambda', gblinear_lambda, np.mean(train_score), np.mean(valid_score)
            fout.write('lambda ' + str(gblinear_lambda) + ' ' + str(np.mean(train_score)) + ' ' +
                       str(np.mean(valid_score)) + '\n')


def train_gbtree_find_argument(argument_file_name):
    # argument_file_name = '../output/argument.concat_3.gblinear'

    dtrain = [xgb.DMatrix(path_train + '.%d.train' % i) for i in range(nfold)]
    dvalid = [xgb.DMatrix(path_train + '.%d.valid' % i) for i in range(nfold)]

    # max_depth = 3
    eta = 0.1
    # subsample = 0.7
    colsample_bytree = 0.7

    fout = open(argument_file_name, 'a')
    for max_depth in [2, 3]:
        print 'max_depth', max_depth
        fout.write('max_depth ' + str(max_depth) + '\n')
        for subsample in [0.6, 0.7, 0.8]:
            train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree, True)
            print 'subsample', subsample, np.mean(train_score), np.mean(valid_score)
            fout.write('subsample ' + str(subsample) + ' ' + str(np.mean(train_score)) + ' ' +
                       str(np.mean(valid_score)) + '\n')


def train_gblinear_get_result():
    dtrain = xgb.DMatrix(path_train)
    dtest = xgb.DMatrix(path_test)
    train_gblinear(dtrain, dtest, 5, 0.21, 0.8)


# max_depth = 3
# eta = 0.1
# subsample = 0.7
# colsample_bytree = 0.7
#
# for max_depth in [2, 3]:
#     print 'max_depth', max_depth
#     for subsample in [0.6, 0.7, 0.8, 0.9, 1]:
#         train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree)
#         print 'subsample', subsample, train_score, valid_score


# train_gblinear_find_argument('../output/argument.concat_6.gblinear')

if __name__ == '__main__':
    train_gbtree_find_argument('../output/argument.concat_6.gblinear')
    print('main function finish!')
