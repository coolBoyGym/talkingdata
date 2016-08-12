import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

import feature
from model_impl import logistic_regression

version = 1
booster = 'logistic_regression'
dataset = 'concat_5_norm'
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


def write_log(log_str):
    with open(path_model_log, 'a') as fout:
        fout.write(log_str)


def tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda, verbose_eval, early_stopping_rounds=50, dtest=None):
    num_boost_round = 1000

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
    num_boost_round = 2000
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


def read_feature(fin, batch_size, zero_pad=True):
    line_buffer = read_buffer(fin, batch_size)
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


def libsvm_2_csr(indices, values):
    csr_indices = []
    csr_values = []
    for i in range(len(indices)):
        csr_indices.extend(map(lambda x: [i, x], indices[i]))
        csr_values.extend(values[i])
    return csr_indices, csr_values, [len(indices), space]


def read_csr_feature(fin, batch_size):
    indices, values, labels = read_feature(fin, batch_size, False)
    csr_indices, csr_values, csr_shape = libsvm_2_csr(indices, values)
    return csr_indices, csr_values, csr_shape, labels


def train_with_batch_csr(model, indices, values, labels, batch_size):
    loss = []
    y = []
    y_prob = []
    if batch_size == -1:
        indices, values, shape = libsvm_2_csr(indices, values)
        loss, y, y_prob = model.train(indices, values, shape, labels)
    else:
        for i in range(len(indices) / batch_size + 1):
            batch_indices = indices[i * batch_size: (i + 1) * batch_size]
            batch_values = values[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]
            batch_indices, batch_values, batch_shape = libsvm_2_csr(batch_indices, batch_values)
            batch_loss, batch_y, batch_y_prob = model.train(batch_indices, batch_values, batch_shape, batch_labels)
            loss.append(batch_loss)
            y.extend(batch_y)
            y_prob.extend(batch_y_prob)
    return np.array(loss), np.array(y), np.array(y_prob)


def predict_with_batch_csr(model, indices, values, batch_size):
    y = []
    y_prob = []
    if batch_size == -1:
        indices, values, shape = libsvm_2_csr(indices, values)
        y, y_prob = model.predict(indices, values, shape)
    else:
        for i in range(len(indices) / batch_size + 1):
            batch_indices = indices[i * batch_size: (i + 1) * batch_size]
            batch_values = values[i * batch_size: (i + 1) * batch_size]
            batch_indices, batch_values, batch_shape = libsvm_2_csr(batch_indices, batch_values)
            batch_y, batch_y_prob = model.predict(batch_indices, batch_values, batch_shape)
            y.extend(batch_y)
            y_prob.extend(batch_y_prob)
    return np.array(y), np.array(y_prob)


if __name__ == '__main__':
    if booster == 'gblinear':
        dtrain = xgb.DMatrix(path_train + '.train')
        dvalid = xgb.DMatrix(path_train + '.valid')
        dtrain_complete = xgb.DMatrix(path_train)
        dtest = xgb.DMatrix(path_test)

        early_stopping_round = 1
        # train_score, valid_score = tune_gblinear(dtrain, dvalid, 0.1, 10, True, early_stopping_round)
        train_score, valid_score = tune_gblinear(dtrain, dvalid, 0.001, 10, True, early_stopping_round, dtest)
        # print train_score, valid_score
        train_gblinear(dtrain_complete, dtest, 0.001, 10, 2)

        # print train_score, valid_score
        # for gblinear_alpha in [0.001]:
        #     for gblinear_lambda in [10]:
        #         train_score, valid_score = tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda, True,
        #                                                  early_stopping_round)
        #         print gblinear_alpha, gblinear_lambda, train_score, valid_score
        # write_log('%f\t%f\t%f\t%f\n' % (gblinear_alpha, gblinear_lambda, train_score, valid_score))
    elif booster == 'gbtree':
        dtrain = xgb.DMatrix(path_train + '.train')
        dvalid = xgb.DMatrix(path_train + '.valid')
        dtrain_complete = xgb.DMatrix(path_train)
        dtest = xgb.DMatrix(path_test)

        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 3, 0.7, 0.7, True)
        train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 4, 0.7, 0.7, True, dtest)
        print train_score, valid_score
        # train_gbtree(dtrain_complete, dtest, 0.1, 3, 0.7, 0.7, 300)

        # max_depth = 3
        # eta = 0.1
        # subsample = 0.7
        # colsample_bytree = 0.7

        # start_time = time.time()
        # colsample_bytree = 0.7
        # for max_depth in [3, 4, 5, 6, 7]:
        #     for eta in [0.1, 0.2, 0.3]:
        #         for subsample in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        #             train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree,
        #                                                    False)
        #             print max_depth, eta, subsample, train_score, valid_score, time.time() - start_time
    elif booster == 'logistic_regression':
        for l1_alpha in [0]:
            for l2_lambda in [10]:
                print '##########################################################################'
                print l1_alpha, l2_lambda
                lr_model = logistic_regression(name=tag, eval_metric='softmax_log_loss', num_class=12,
                                               input_space=space, l1_alpha=l1_alpha, l2_lambda=l2_lambda,
                                               optimizer='adadelta', learning_rate=0.1, )

                # lr_model.write_log_header()
                # lr_model.write_log('loss\ttrain-score\tvalid_score')
                num_round = 500
                batch_size = -1
                train_indices, train_values, train_labels = read_feature(open(path_train + '.train'), -1, False)
                valid_indices, valid_values, valid_labels = read_feature(open(path_train + '.valid'), -1, False)
                for j in range(num_round):
                    start_time = time.time()
                    # train_loss, train_preds, train_labels = train_with_batch(path_train + '.train', batch_size)
                    # valid_preds, valid_labels = predict_with_batch(path_train + '.valid', batch_size)
                    train_loss, train_y, train_y_prob = train_with_batch_csr(lr_model, train_indices, train_values,
                                                                             train_labels, batch_size)
                    valid_y, valid_y_prob = predict_with_batch_csr(lr_model, valid_indices, valid_values, batch_size)
                    train_score = log_loss(train_labels, train_y_prob)
                    valid_score = log_loss(valid_labels, valid_y_prob)
                    print 'loss: %f \ttrain_score: %f\tvalid_score: %f\ttime: %d' % (
                        train_loss.mean(), train_score, valid_score, time.time() - start_time)
                    lr_model.write_log('%d\t%f\t%f\t%f\n' % (j, train_loss.mean(), train_score, valid_score))
