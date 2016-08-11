import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

import feature
from model_impl import logistic_regression

version = 1
booster = 'gbtree'
dataset = 'concat_5'
# nfold = 5
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


def make_submission(test_pred):
    test_device_id = np.loadtxt('../data/raw/gender_age_test.csv', skiprows=1, dtype=np.int64)

    with open(path_submission, 'w') as fout:
        fout.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
        for i in range(len(test_device_id)):
            fout.write('%s,%s\n' % (test_device_id[i], ','.join(map(lambda d: str(d), test_pred[i]))))

    print path_submission


def make_feature_model_output(train_pred, valid_pred, test_pred):
    fea_pred = feature.multi_feature(name=tag, dtype='f', space=12, rank=12, size=len(train_pred) + len(test_pred))
    indices = np.array([range(12)] * (len(train_pred) + len(valid_pred) + len(test_pred)))
    values = np.vstack((valid_pred, train_pred, test_pred))
    fea_pred.set_value(indices, values)
    fea_pred.dump()


def tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda, verbose_eval, dtest=None):
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

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval)

    train_pred = bst.predict(dtrain)
    train_score = log_loss(dtrain.get_label(), train_pred)

    valid_pred = bst.predict(dvalid)
    valid_score = log_loss(dvalid.get_label(), valid_pred)

    if dtest is not None:
        train_pred = bst.predict(dtrain)
        valid_pred = bst.predict(dvalid)
        test_pred = bst.predict(dtest)
        make_feature_model_output(train_pred, valid_pred, test_pred)

    return train_score, valid_score


def train_gblinear(dtrain_complete, dtest, gblinear_alpha, gblinear_lambda, num_boost_round):
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

    watchlist = [(dtrain_complete, 'train')]
    bst = xgb.train(params, dtrain_complete, num_boost_round, evals=watchlist, verbose_eval=True)

    bst.save_model(path_model_bin)
    bst.dump_model(path_model_dump)

    test_pred = bst.predict(dtest)
    make_submission(test_pred)


def tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree, verbose_eval, dtest=None):
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

    if dtest is not None:
        train_pred = bst.predict(dtrain, ntree_limit=bst.best_iteration)
        valid_pred = bst.predict(dvalid, ntree_limit=bst.best_iteration)
        test_pred = bst.predict(dtest, ntree_limit=bst.best_iteration)
        make_feature_model_output(train_pred, valid_pred, test_pred)

    return train_score, valid_score


def train_gbtree(dtrain_complete, dtest, eta, max_depth, subsample, colsample_bytree, num_boost_round):
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

    watchlist = [(dtrain_complete, 'train')]
    bst = xgb.train(params, dtrain_complete, num_boost_round, evals=watchlist, verbose_eval=True)

    bst.save_model(path_model_bin)
    bst.dump_model(path_model_dump)

    test_pred = bst.predict(dtest)
    make_submission(test_pred)


def read_buffer(fin, buf_size):
    line_buffer = []
    while True:
        try:
            line_buffer.append(next(fin))
        except StopIteration as e:
            print e
            break
        if len(line_buffer) == buf_size:
            break
    return line_buffer


def read_feature(fin, buf_size, zero_pad=True):
    line_buffer = read_buffer(fin, buf_size)
    indices = []
    values = []
    labels = []
    for line in line_buffer:
        fields = line.strip().split()
        tmp_y = [0] * num_class
        tmp_y[int(fields[0])] = 1
        labels.append(tmp_y)
        tmp_i = map(lambda x: int(x.split(':')[0]), fields[1:])
        tmp_v = map(lambda x: float(x.split(':')[1]), fields[1:])
        if zero_pad and len(tmp_i) < rank:
            tmp_i.extend([rank] * (rank - len(tmp_i)))
            tmp_v.extend([0] * (rank - len(tmp_v)))
        indices.append(tmp_i)
        values.append(tmp_v)
    indices = np.array(indices)
    values = np.array(values)
    labels = np.array(labels)
    return indices, values, labels


def train_with_batch(file_name, batch_size):
    data = open(file_name, 'r')
    loss = []
    preds = []
    labels = []
    while True:
        batch_indices, batch_values, batch_labels = read_feature(data, batch_size, True)
        batch_loss, batch_preds = lr_model.train(batch_indices, batch_values, batch_labels)
        loss.append(batch_loss)
        preds.extend(batch_preds)
        labels.extend(batch_labels)
        if len(batch_indices) < batch_size:
            break
    return np.array(loss), np.array(preds), np.array(labels)


def predict_with_batch(file_name, batch_size):
    data = open(file_name, 'r')
    preds = []
    labels = []
    while True:
        batch_indices, batch_values, batch_labels = read_feature(data, batch_size, True)
        batch_preds = lr_model.predict(batch_indices, batch_values)
        preds.extend(batch_preds)
        labels.extend(batch_labels)
        if len(batch_indices) < batch_size:
            break
    return np.array(preds), np.array(labels)


if __name__ == '__main__':
    if booster == 'gblinear':
        dtrain = xgb.DMatrix(path_train + '.train')
        dvalid = xgb.DMatrix(path_train + '.valid')
        dtrain_complete = xgb.DMatrix(path_train)
        dtest = xgb.DMatrix(path_test)

        train_score, valid_score = tune_gblinear(dtrain, dvalid, 1, 10, True)
        # train_score, valid_score = tune_gblinear(dtrain, dvalid, 1, 10, True, dtest)
        print train_score, valid_score
        # train_gblinear(dtrain_complete, dtest, 100, 1, 10)

        # print train_score, valid_score
        # for gblinear_alpha in [0.007, 0.008, 0.009, 0.01, 0.02]:
        #     print 'alpha', gblinear_alpha
        #     for gblinear_lambda in [13, 14, 15]:
        #         train_score, valid_score = tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda, False)
        #         print 'lambda', gblinear_lambda, np.mean(train_score), np.mean(valid_score)
    elif booster == 'gbtree':
        dtrain = xgb.DMatrix(path_train + '.train')
        dvalid = xgb.DMatrix(path_train + '.valid')
        dtrain_complete = xgb.DMatrix(path_train)
        dtest = xgb.DMatrix(path_test)

        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 3, 0.7, 0.7, True)
        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 3, 0.7, 0.7, True, dtest)
        # print train_score, valid_score
        # train_gbtree(dtrain_complete, dtest, 0.1, 3, 0.7, 0.7, 300)

        max_depth = 3
        eta = 0.1
        subsample = 0.7
        colsample_bytree = 0.7

        for max_depth in [2, 3, 4, 5, 6]:
            for eta in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                for subsample in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    for colsample_bytree in[0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                        train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample,
                                                               colsample_bytree, False)
                        print 'max_depth', max_depth, 'eta', eta, 'subsample', subsample, 'colsample', \
                            colsample_bytree, train_score, valid_score

    elif booster == 'logistic_regression':
        lr_model = logistic_regression(tag, 'log_loss', 12, space + 1, rank, 0.1, 'adam', 0.1, None)
        # lr_model.write_log_header()
        # lr_model.write_log('loss\ttrain-score\tvalid_score')
        num_round = 100
        batch_size = 100
        for j in range(num_round):
            start_time = time.time()
            train_loss, train_preds, train_labels = train_with_batch(path_train + '.train', batch_size)
            valid_preds, valid_labels = predict_with_batch(path_train + '.valid', batch_size)
            train_score = log_loss(train_labels, train_preds)
            valid_score = log_loss(valid_labels, valid_preds)
            print 'loss: %f \ttrain_score: %f\tvalid_score: %f' % (np.mean(train_loss), train_score, valid_score)
            # lr_model.write_log('%f\t%f\t%f\n' % (train_loss, train_score, valid_score))
            print 'one round training', time.time() - start_time
