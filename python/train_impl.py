import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

import feature

VERSION = None
BOOSTER = None
DATASET = None
PATH_TRAIN = None
PATH_TEST = None
TAG = None
PATH_MODEL_LOG = None
PATH_MODEL_BIN = None
PATH_MODEL_DUMP = None
PATH_SUBMISSION = None
RANDOME_STATE = 0

SPACE = None
RANK = None
SIZE = None
NUM_CLASS = None


def init_constant(dataset, booster, version, random_state=0):
    global DATASET, BOOSTER, VERSION, PATH_TRAIN, PATH_TEST, TAG, PATH_MODEL_LOG, PATH_MODEL_BIN, PATH_MODEL_DUMP, \
        PATH_SUBMISSION, RANDOME_STATE, SPACE, RANK, SIZE, NUM_CLASS
    DATASET = dataset
    BOOSTER = booster
    VERSION = version
    PATH_TRAIN = '../input/' + DATASET + '.train'
    PATH_TEST = '../input/' + DATASET + '.test'
    TAG = '%s_%s_%d' % (DATASET, BOOSTER, VERSION)
    PATH_MODEL_LOG = '../model/' + TAG + '.log'
    PATH_MODEL_BIN = '../model/' + TAG + '.model'
    PATH_MODEL_DUMP = '../model/' + TAG + '.dump'
    PATH_SUBMISSION = '../output/' + TAG + '.submission'
    print TAG
    RANDOME_STATE = random_state
    fea_tmp = feature.multi_feature(name=DATASET)
    fea_tmp.load_meta()
    SPACE = fea_tmp.get_space()
    RANK = fea_tmp.get_rank()
    SIZE = fea_tmp.get_size()
    NUM_CLASS = 12
    print 'feature space: %d, rank: %d, size: %d, num class: %d' % (SPACE, RANK, SIZE, NUM_CLASS)


def make_submission(test_pred):
    global PATH_SUBMISSION
    test_device_id = np.loadtxt('../data/raw/gender_age_test.csv', skiprows=1, dtype=np.int64)

    with open(PATH_SUBMISSION, 'w') as fout:
        fout.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
        for i in range(len(test_device_id)):
            fout.write('%s,%s\n' % (test_device_id[i], ','.join(map(lambda d: str(d), test_pred[i]))))

    print PATH_SUBMISSION


def make_feature_model_output(train_pred, valid_pred, test_pred):
    fea_pred = feature.multi_feature(name=TAG, dtype='f', space=12, rank=12,
                                     size=len(train_pred) + len(valid_pred) + len(test_pred))
    indices = np.array([range(12)] * (len(train_pred) + len(valid_pred) + len(test_pred)))
    values = np.vstack((valid_pred, train_pred, test_pred))
    fea_pred.set_value(indices, values)
    fea_pred.dump()


def write_log(log_str):
    global PATH_MODEL_LOG
    with open(PATH_MODEL_LOG, 'a') as fout:
        fout.write(log_str)


def tune_gblinear(dtrain, dvalid, gblinear_alpha=0, gblinear_lambda=0, gblinear_lambda_bias=0, verbose_eval=True,
                  early_stopping_rounds=50, dtest=None):
    global BOOSTER, RANDOME_STATE
    num_boost_round = 1000

    params = {
        'booster': BOOSTER,
        'silent': 1,
        'num_class': 12,
        'lambda': gblinear_lambda,
        'lambda_bias': gblinear_lambda_bias,
        'alpha': gblinear_alpha,
        'objective': 'multi:softprob',
        'seed': RANDOME_STATE,
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


def train_gblinear(dtrain_complete, dtest, gblinear_alpha, gblinear_lambda, gblinear_lambda_bias, num_boost_round):
    global BOOSTER, RANDOME_STATE
    params = {
        'booster': BOOSTER,
        'silent': 1,
        'num_class': 12,
        'lambda': gblinear_lambda,
        'lambda_bias': gblinear_lambda_bias,
        'alpha': gblinear_alpha,
        'objective': 'multi:softprob',
        'seed': RANDOME_STATE,
        'eval_metric': 'mlogloss',
    }

    watchlist = [(dtrain_complete, 'train')]
    bst = xgb.train(params, dtrain_complete, num_boost_round, evals=watchlist, verbose_eval=True)

    bst.save_model(PATH_MODEL_BIN)
    bst.dump_model(PATH_MODEL_DUMP)

    test_pred = bst.predict(dtest)
    make_submission(test_pred)


def tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree, gamma=0, min_child_weight=1,
                max_delta_step=0, verbose_eval=False,
                early_stopping_rounds=50, dtest=None):
    global BOOSTER, RANDOME_STATE
    num_boost_round = 2000

    params = {
        "booster": BOOSTER,
        "silent": 1,
        "num_class": 12,
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "gamma":gamma,
        "min_child_weight":min_child_weight,
        "max_delta_step":max_delta_step,
        "objective": "multi:softprob",
        "seed": RANDOME_STATE,
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
    global BOOSTER, RANDOME_STATE
    params = {
        "booster": BOOSTER,
        "silent": 1,
        "num_class": 12,
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "objective": "multi:softprob",
        "seed": RANDOME_STATE,
        "eval_metric": "mlogloss",
    }

    watchlist = [(dtrain_complete, 'train')]
    bst = xgb.train(params, dtrain_complete, num_boost_round, evals=watchlist, verbose_eval=True)

    bst.save_model(PATH_MODEL_BIN)
    bst.dump_model(PATH_MODEL_DUMP)

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
    global NUM_CLASS, RANK
    line_buffer = read_buffer(fin, batch_size)
    indices = []
    values = []
    labels = []
    for line in line_buffer:
        fields = line.strip().split()
        tmp_y = [0] * NUM_CLASS
        tmp_y[int(fields[0])] = 1
        labels.append(tmp_y)
        tmp_i = map(lambda x: int(x.split(':')[0]), fields[1:])
        tmp_v = map(lambda x: float(x.split(':')[1]), fields[1:])
        if zero_pad and len(tmp_i) < RANK:
            tmp_i.extend([RANK] * (RANK - len(tmp_i)))
            tmp_v.extend([0] * (RANK - len(tmp_v)))
        indices.append(tmp_i)
        values.append(tmp_v)
    indices = np.array(indices)
    values = np.array(values)
    labels = np.array(labels)
    return indices, values, labels


def libsvm_2_csr(indices, values):
    global SPACE
    csr_indices = []
    csr_values = []
    for i in range(len(indices)):
        csr_indices.extend(map(lambda x: [i, x], indices[i]))
        csr_values.extend(values[i])
    return csr_indices, csr_values, [len(indices), SPACE]


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


def get_labels(train_size, valid_size):
    global NUM_CLASS
    data = np.loadtxt('../feature/device_id', delimiter=',', usecols=[1], skiprows=1, dtype=np.int64)
    valid_labels = []
    train_labels = []
    for i in range(valid_size):
        tmp = [0] * NUM_CLASS
        tmp[data[i]] = 1
        valid_labels.append(tmp)
    for i in range(valid_size, valid_size + train_size):
        tmp = [0] * NUM_CLASS
        tmp[data[i]] = 1
        train_labels.append(tmp)
    return np.array(train_labels), np.array(valid_labels)


def get_model_preds(model_name_list):
    model_preds = []

    for mn in model_name_list:
        data = np.loadtxt('../feature/' + mn, delimiter=' ', skiprows=1, dtype=str)
        pred = []
        for i in range(12):
            pred.append(map(lambda x: [float(x.split(':')[1])], data[:, i]))
        pred = np.hstack(pred)
        model_preds.append(pred)
    model_preds = np.array(model_preds)
    return model_preds


def average_predict(model_preds, train_size, valid_size, train_labels, valid_labels, model_weights=None):
    if model_weights is None:
        average_preds = np.mean(model_preds, axis=0)
    else:
        weighted_model_preds = []
        for i in range(len(model_preds)):
            weighted_model_preds.append(model_preds[i] * model_weights[i])
        average_preds = np.mean(weighted_model_preds, axis=0)
    valid_pred = average_preds[:valid_size]
    train_pred = average_preds[valid_size:(valid_size + train_size)]
    train_score = log_loss(train_labels, train_pred)
    valid_score = log_loss(valid_labels, valid_pred)
    return train_score, valid_score


def train_gblinear_get_result(train_round, train_alpha, train_lambda):
    global PATH_TRAIN, PATH_TEST
    dtrain = xgb.DMatrix(PATH_TRAIN)
    dtest = xgb.DMatrix(PATH_TEST)
    train_gblinear(dtrain, dtest, train_round, train_alpha, train_lambda)


def train_gbtree_find_argument(argument_file_name):
    global PATH_TRAIN, PATH_TEST
    dtrain = xgb.DMatrix(PATH_TRAIN + '.train')
    dvalid = xgb.DMatrix(PATH_TRAIN + '.valid')

    max_depth = 4
    subsample = 0.8

    fout = open(argument_file_name, 'a')
    for eta in [0.1, 0.15, 0.2, 0.3]:
        print 'eta', eta
        fout.write('eta ' + str(eta) + '\n')
        for colsample_bytree in [0.6, 0.7, 0.8, 0.9]:
            train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree, True)
            print 'colsample_bytree', colsample_bytree, train_score, valid_score
            fout.write('colsample_bytree ' + str(colsample_bytree) + ' ' + str(train_score) + ' ' +
                       str(valid_score) + '\n')


def train_gbtree_confirm_argument(max_depth=4, eta=0.3, subsample=0.7, colsample_bytree=0.7, verbose_eval=False):
    global PATH_TRAIN, PATH_TEST
    dtrain = xgb.DMatrix(PATH_TRAIN + '.train')
    dvalid = xgb.DMatrix(PATH_TRAIN + '.valid')

    train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample,
                                           colsample_bytree, verbose_eval)
    print train_score, valid_score


def train_gbtree_get_result():
    global PATH_TRAIN, PATH_TEST
    dtrain = xgb.DMatrix(PATH_TRAIN)
    dtest = xgb.DMatrix(PATH_TEST)

    num_boost_round = 100
    eta = 0.1
    max_depth = 4
    subsample = 0.8
    colsample_bytree = 0.8

    train_gbtree(dtrain, dtest, num_boost_round, eta, max_depth, subsample, colsample_bytree)
