import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

import feature
from model_impl import logistic_regression

version = 1
booster = 'logistic_regression'
dataset = 'concat_1'
nfold = 5
path_train = '../input/' + dataset + '.train'
path_test = '../input/' + dataset + '.test'
tag = '%s_%s_%d' % (dataset, booster, version)
path_model_log = '../model/' + tag + '.log'
path_model_bin = '../model/' + tag + '.model'
path_model_dump = '../model/' + tag + '.dump'
path_submission = '../output/' + tag + '.submission'
random_state = 0

print tag
fea_tmp = feature.multi_feature(name=dataset)
fea_tmp.load_meta()
space = fea_tmp.get_space()
rank = fea_tmp.get_rank()
size = fea_tmp.get_size()
num_class = 12
print 'feature space: %d, rank: %d, size: %d, num class: %d' % (space, rank, size, num_class)


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


def tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree, verbose_eval):
    num_boost_round = 1000
    early_stopping_rounds = 50

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

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval)

    train_pred = bst.predict(dtrain, ntree_limit=bst.best_iteration)
    train_score = log_loss(dtrain.get_label(), train_pred)

    valid_pred = bst.predict(dvalid, ntree_limit=bst.best_iteration)
    valid_score = log_loss(dvalid.get_label(), valid_pred)

    return train_score, valid_score


def read_feature(file_path, zero_pad):
    indices = []
    values = []
    labels = []
    with open(file_path, 'r') as fin:
        for line in fin:
            fields = line.strip().split()
            tmp_y = [0] * num_class
            tmp_y[int(fields[0])] = 1
            labels.append(tmp_y)
            tmp_i = map(lambda x: int(x.split(':')[0]), fields[1:])
            tmp_v = map(lambda x: float(x.split(':')[1]), fields[1:])
            if zero_pad and len(tmp_i) < space:
                tmp_i.extend([space] * (space - len(tmp_i)))
                tmp_v.extend([0] * (space - len(tmp_v)))
            indices.append(tmp_i)
            values.append(tmp_v)
    indices = np.array(indices)
    values = np.array(values)
    labels = np.array(labels)
    return indices, values, labels


if __name__ == '__main__':

    # dtrain = [xgb.DMatrix(path_train + '.%d.train' % i) for i in range(nfold)]
    # dvalid = [xgb.DMatrix(path_train + '.%d.valid' % i) for i in range(nfold)]

    # dtrain = xgb.DMatrix(path_train)
    # dtest = xgb.DMatrix(path_test)

    if booster == 'gblinear':
        # train_score, valid_score = tune_gblinear(dtrain, dvalid, 0.009, 14, True)
        # print np.mean(train_score), np.mean(valid_score)

        train_gblinear(dtrain, dtest, 100, 0.009, 14)

        # for gblinear_alpha in [0.007, 0.008, 0.009, 0.01, 0.02]:
        #     print 'alpha', gblinear_alpha
        #     for gblinear_lambda in [13, 14, 15]:
        #         train_score, valid_score = tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda, False)
        #         print 'lambda', gblinear_lambda, np.mean(train_score), np.mean(valid_score)
    elif booster == 'gbtree':
        max_depth = 3
        eta = 0.1
        subsample = 0.7
        colsample_bytree = 0.7

        for max_depth in [2, 3]:
            print 'max_depth', max_depth
            for subsample in [0.6, 0.7, 0.8, 0.9, 1]:
                train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree,
                                                       False)
                print 'subsample', subsample, train_score, valid_score
    elif booster == 'logistic_regression':
        lr_model = logistic_regression(tag, 'log_loss', 12)
        lr_model.init(space + 1, rank, 0, 'adam', 0.01)
        # lr_model.write_log_header()
        # lr_model.write_log('loss\ttrain-score\tvalid_score')

        num_round = 100
        for i in range(nfold):
            train_indices, train_values, train_labels = read_feature(path_train + '.%d.train' % i, True)
            print 'read finish'
            # valid_indices, valid_values, valid_labels = read_feature(path_train + '.%d.valid' % i, True)
            for j in range(num_round):
                y = lr_model.y.eval()
                print y
                print y.shape
                exit(0)
                # train_loss = lr_model.train(train_indices, train_values, train_labels)
                # train_pred = lr_model.predict(train_indices, train_values)
                # valid_pred = lr_model.predict(valid_indices, valid_values)
                # train_score = log_loss(train_labels, train_pred)
                # valid_score = log_loss(valid_labels, valid_pred)
                # print 'loss: %f \ttrain_score: %f\tvalid_score: %f' % (train_loss, train_score, valid_score)
                # lr_model.write_log('%f\t%f\t%f\n' % (train_loss, train_score, valid_score))
