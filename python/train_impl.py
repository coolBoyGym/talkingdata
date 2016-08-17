import time

import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import feature
from model_impl import factorization_machine, multi_layer_perceptron

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


def init_constant(dataset, booster, version, random_state=0):
    global DATASET, BOOSTER, VERSION, PATH_TRAIN, PATH_TEST, TAG, PATH_MODEL_LOG, PATH_MODEL_BIN, PATH_MODEL_DUMP, \
        PATH_SUBMISSION, RANDOM_STATE, SPACE, RANK, SIZE, NUM_CLASS, PATH_TRAIN_TRAIN, PATH_TRAIN_VALID
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


def get_feature_model_output(train_pred, valid_pred, test_pred):
    fea_pred = feature.multi_feature(name=TAG, dtype='f', space=12, rank=12,
                                     size=len(train_pred) + len(valid_pred) + len(test_pred))
    indices = np.array([range(12)] * (len(train_pred) + len(valid_pred) + len(test_pred)))
    values = np.vstack((valid_pred, train_pred, test_pred))
    fea_pred.set_value(indices, values)
    return fea_pred


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
    global NUM_CLASS
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=8, max_depth=max_depth, max_features=max_features)
    clf.fit(Xtrain, train_labels)
    test_pred = clf.predict_proba(Xtest)
    test_pred = np.transpose(np.array(map(lambda x: x[:, 0], test_pred))) / (NUM_CLASS - 1)
    make_submission(test_pred)


def tune_gblinear(dtrain, dvalid, gblinear_alpha=0, gblinear_lambda=0, verbose_eval=True,
                  early_stopping_rounds=50, dtest=None):
    global BOOSTER, RANDOM_STATE
    num_boost_round = 1000

    params = {
        'booster': BOOSTER,
        'silent': 1,
        'num_class': 12,
        'lambda': gblinear_lambda,
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

    if dtest is not None:
        train_pred = bst.predict(dtrain)
        valid_pred = bst.predict(dvalid)
        test_pred = bst.predict(dtest)
        make_feature_model_output(train_pred, valid_pred, test_pred)

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

    if dtest is not None:
        train_pred = bst.predict(dtrain)
        valid_pred = bst.predict(dvalid)
        test_pred = bst.predict(dtest)
        fea_out = get_feature_model_output(train_pred, valid_pred, test_pred)

    return fea_out


def train_gblinear(dtrain, dtest, gblinear_alpha, gblinear_lambda, gblinear_lambda_bias, num_boost_round):
    global BOOSTER, RANDOM_STATE
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

    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, verbose_eval=True)

    bst.save_model(PATH_MODEL_BIN)
    bst.dump_model(PATH_MODEL_DUMP)

    test_pred = bst.predict(dtest)
    make_submission(test_pred)


def tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree, gbtree_lambda=1, gbtree_alpha=0, gamma=0,
                min_child_weight=1, max_delta_step=0, verbose_eval=False, early_stopping_rounds=50, dtest=None):
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
    train_score = log_loss(dtrain.get_label(), train_pred)

    valid_pred = bst.predict(dvalid, ntree_limit=bst.best_iteration)
    valid_score = log_loss(dvalid.get_label(), valid_pred)

    if dtest is not None:
        train_pred = bst.predict(dtrain, ntree_limit=bst.best_iteration)
        valid_pred = bst.predict(dvalid, ntree_limit=bst.best_iteration)
        test_pred = bst.predict(dtest, ntree_limit=bst.best_iteration)
        make_feature_model_output(train_pred, valid_pred, test_pred)

    return train_score, valid_score


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

    if dtest is not None:
        train_pred = bst.predict(dtrain, ntree_limit=bst.best_iteration)
        valid_pred = bst.predict(dvalid, ntree_limit=bst.best_iteration)
        test_pred = bst.predict(dtest, ntree_limit=bst.best_iteration)
        fea_out = get_feature_model_output(train_pred, valid_pred, test_pred)

    return fea_out


def train_gbtree(dtrain, dtest, eta, max_depth, subsample, colsample_bytree, gbtree_lambda, gbtree_alpha,
                 num_boost_round, gamma=0, min_child_weight=1, max_delta_step=0):
    global BOOSTER, RANDOM_STATE
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

    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, verbose_eval=True)

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


def read_feature(fin, batch_size):
    global NUM_CLASS
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
        # if zero_pad and len(tmp_i) < RANK:
        #     tmp_i.extend([RANK] * (RANK - len(tmp_i)))
        #     tmp_v.extend([0] * (RANK - len(tmp_v)))
        indices.append(tmp_i)
        values.append(tmp_v)
    indices = np.array(indices)
    values = np.array(values)
    labels = np.array(labels)
    return indices, values, labels


def split_feature(fin, batch_size, zero_pad=True):
    global NUM_CLASS


def label_2_group_id(labels, num_class=None):
    global NUM_CLASS
    if num_class is None:
        num_class = NUM_CLASS
    tmp = np.arange(num_class)
    group_ids = labels.dot(tmp)
    return group_ids


def libsvm_2_csr(indices, values, space=None):
    global SPACE
    csr_indices = []
    csr_values = []
    for i in range(len(indices)):
        csr_indices.extend(map(lambda x: [i, x], indices[i]))
        csr_values.extend(values[i])
    if space is None:
        csr_shape = [len(indices), SPACE]
    else:
        csr_shape = [len(indices), space]
    return np.array(csr_indices), np.array(csr_values), csr_shape


def csr_2_libsvm(csr_indices, csr_values, csr_shape, reorder=False):
    data = np.hstack((csr_indices, np.reshape(csr_values, [-1, 1])))
    if reorder:
        data = sorted(data, key=lambda x: (x[0], x[1]))
    print data
    indices = []
    values = []
    for i in range(len(data)):
        r, c, v = data[i]
        if len(indices) <= r:
            while len(indices) <= r:
                indices.append([])
                values.append([])
            indices[r].append(c)
            values[r].append(v)
        elif len(indices) == r + 1:
            indices[r].append(c)
            values[r].append(v)
    while len(indices) < csr_shape[0]:
        indices.append([])
        values.append([])
    return indices, values


def read_csr_feature(fin, batch_size):
    indices, values, labels = read_feature(fin, batch_size)
    csr_indices, csr_values, csr_shape = libsvm_2_csr(indices, values, SPACE)
    return csr_indices, csr_values, csr_shape, labels


def train_with_batch_csr(model, indices, values, labels, drops=0, batch_size=None, verbose=False):
    loss = []
    y = []
    y_prob = []
    if batch_size == -1:
        indices, values, shape = libsvm_2_csr(indices, values, SPACE)
        loss, y, y_prob = model.train(indices, values, shape, labels)
    else:
        for i in range(len(indices) / batch_size + 1):
            batch_indices = indices[i * batch_size: (i + 1) * batch_size]
            batch_values = values[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]
            batch_indices, batch_values, batch_shape = libsvm_2_csr(batch_indices, batch_values, SPACE)
            batch_loss, batch_y, batch_y_prob = model.train(batch_indices, batch_values, batch_shape, batch_labels,
                                                            drops=drops)
            loss.append(batch_loss)
            y.extend(batch_y)
            y_prob.extend(batch_y_prob)
            if verbose:
                print batch_loss
                print batch_y
                print batch_y_prob
    return np.array(loss), np.array(y), np.array(y_prob)


def predict_with_batch_csr(model, indices, values, drops=0, batch_size=None):
    y = []
    y_prob = []
    if batch_size == -1:
        indices, values, shape = libsvm_2_csr(indices, values, SPACE)
        y, y_prob = model.predict(indices, values, shape)
    else:
        for i in range(len(indices) / batch_size + 1):
            batch_indices = indices[i * batch_size: (i + 1) * batch_size]
            batch_values = values[i * batch_size: (i + 1) * batch_size]
            batch_indices, batch_values, batch_shape = libsvm_2_csr(batch_indices, batch_values, SPACE)
            batch_y, batch_y_prob = model.predict(batch_indices, batch_values, batch_shape, drops=drops)
            y.extend(batch_y)
            y_prob.extend(batch_y_prob)
    return np.array(y), np.array(y_prob)


def check_early_stop(valid_scores, early_stopping_round=50, early_stop_precision=0.0001, mode='no_decrease'):
    if np.argmin(valid_scores) + early_stopping_round > len(valid_scores):
        return False
    minimum = np.min(valid_scores)
    if mode == 'increase' and valid_scores[-1] - minimum > early_stop_precision:
        return True
    elif mode == 'no_decrease' and minimum - valid_scores[-1] < early_stop_precision:
        return True
    return False


def tune_factorization_machine(train_data, valid_data, factor_order, opt_algo, learning_rate, l1_w=0, l1_v=0, l2_w=0,
                               l2_v=0, l2_b=0, num_round=200, batch_size=100, early_stopping_round=10, verbose=True,
                               save_log=True):
    train_indices, train_values, train_labels = train_data
    valid_indices, valid_values, valid_labels = valid_data
    fm_model = factorization_machine(name=TAG,
                                     eval_metric='softmax_log_loss',
                                     num_class=12,
                                     input_space=SPACE,
                                     factor_order=factor_order,
                                     opt_algo=opt_algo,
                                     learning_rate=learning_rate,
                                     l1_w=l1_w,
                                     l1_v=l1_v,
                                     l2_w=l2_w,
                                     l2_v=l2_v,
                                     l2_b=l2_b)
    train_scores = []
    valid_scores = []
    for j in range(num_round):
        start_time = time.time()
        train_loss, train_y, train_y_prob = train_with_batch_csr(fm_model, train_indices, train_values,
                                                                 train_labels, batch_size=batch_size, verbose=False)
        valid_y, valid_y_prob = predict_with_batch_csr(fm_model, valid_indices, valid_values, batch_size=batch_size)
        train_score = log_loss(train_labels, train_y_prob)
        valid_score = log_loss(valid_labels, valid_y_prob)
        if verbose:
            print '[%d]\tloss: %f \ttrain_score: %f\tvalid_score: %f\ttime: %d' % \
                  (j, train_loss.mean(), train_score, valid_score, time.time() - start_time)
        if save_log:
            write_log('%d\t%f\t%f\t%f\n' % (j, train_loss.mean(), train_score, valid_score))
        train_scores.append(train_score)
        valid_scores.append(valid_score)
        if check_early_stop(valid_scores, early_stopping_round=early_stopping_round, mode='no_decrease'):
            if verbose:
                best_iteration = j + 1 - early_stopping_round
                print 'best iteration:\n[%d]\ttrain_score: %f\tvalid_score: %f' % (
                    best_iteration, train_scores[best_iteration], valid_scores[best_iteration])
            break
    return train_scores[-1], valid_scores[-1]


def tune_multi_layer_perceptron(train_data, valid_data, layer_sizes, layer_activates, opt_algo, learning_rate, drops,
                                num_round=200, batch_size=100, early_stopping_round=10, verbose=True, save_log=True,
                                save_model=False):
    train_indices, train_values, train_labels = train_data
    valid_indices, valid_values, valid_labels = valid_data
    mlp_model = multi_layer_perceptron(name=TAG, eval_metric='softmax_log_loss',
                                       layer_sizes=layer_sizes,
                                       layer_activates=layer_activates,
                                       opt_algo=opt_algo,
                                       learning_rate=learning_rate)
    exit(0)
    train_scores = []
    valid_scores = []
    for j in range(num_round):
        start_time = time.time()
        train_loss, train_y, train_y_prob = train_with_batch_csr(mlp_model, train_indices, train_values,
                                                                 train_labels, drops, batch_size, verbose=False)
        valid_y, valid_y_prob = predict_with_batch_csr(mlp_model, valid_indices, valid_values, drops=[1] * len(drops),
                                                       batch_size=batch_size)
        train_score = log_loss(train_labels, train_y_prob)
        valid_score = log_loss(valid_labels, valid_y_prob)
        if verbose:
            print '[%d]\tloss: %f \ttrain_score: %f\tvalid_score: %f\ttime: %d' % \
                  (j, train_loss.mean(), train_score, valid_score, time.time() - start_time)
        if save_log:
            write_log('%d\t%f\t%f\t%f\n' % (j, train_loss.mean(), train_score, valid_score))
        train_scores.append(train_score)
        valid_scores.append(valid_score)
        if check_early_stop(valid_scores, early_stopping_round=early_stopping_round, mode='no_decrease'):
            if verbose:
                best_iteration = j + 1 - early_stopping_round
                print 'best iteration:\n[%d]\ttrain_score: %f\tvalid_score: %f' % (
                    best_iteration, train_scores[best_iteration], valid_scores[best_iteration])
            break
    if save_model:
        mlp_model.dump()
    return train_scores[-1], valid_scores[-1]


def ensemble_multi_layer_perceptron(train_data, valid_data, layer_sizes, layer_activates, opt_algo, learning_rate,
                                    drops, num_round=200, batch_size=100, early_stopping_round=10, verbose=True,
                                    save_log=True,
                                    dtest=None):
    train_indices, train_values, train_labels = train_data
    valid_indices, valid_values, valid_labels = valid_data
    mlp_model = multi_layer_perceptron(name=TAG, eval_metric='softmax_log_loss',
                                       layer_sizes=layer_sizes,
                                       layer_activates=layer_activates,
                                       opt_algo=opt_algo,
                                       learning_rate=learning_rate)
    train_scores = []
    valid_scores = []
    for j in range(num_round):
        start_time = time.time()
        train_loss, train_y, train_y_prob = train_with_batch_csr(mlp_model, train_indices, train_values,
                                                                 train_labels, drops, batch_size, verbose=False)
        valid_y, valid_y_prob = predict_with_batch_csr(mlp_model, valid_indices, valid_values, drops=[1] * len(drops),
                                                       batch_size=batch_size)
        train_score = log_loss(train_labels, train_y_prob)
        valid_score = log_loss(valid_labels, valid_y_prob)
        if verbose:
            print '[%d]\tloss: %f \ttrain_score: %f\tvalid_score: %f\ttime: %d' % \
                  (j, train_loss.mean(), train_score, valid_score, time.time() - start_time)
        if save_log:
            write_log('%d\t%f\t%f\t%f\n' % (j, train_loss.mean(), train_score, valid_score))
        train_scores.append(train_score)
        valid_scores.append(valid_score)
        if check_early_stop(valid_scores, early_stopping_round=early_stopping_round, mode='no_decrease'):
            if verbose:
                best_iteration = j + 1 - early_stopping_round
                print 'best iteration:\n[%d]\ttrain_score: %f\tvalid_score: %f' % (
                    best_iteration, train_scores[best_iteration], valid_scores[best_iteration])
            break
    return train_scores[-1], valid_scores[-1]

    if dtest is not None:
        train_pred = bst.predict(dtrain, ntree_limit=bst.best_iteration)
        valid_pred = bst.predict(dvalid, ntree_limit=bst.best_iteration)
        test_pred = bst.predict(dtest, ntree_limit=bst.best_iteration)
        fea_out = get_feature_model_output(train_pred, valid_pred, test_pred)

    return fea_out


def train_multi_layer_perceptron(train_data, test_data, layer_sizes, layer_activates, opt_algo, learning_rate, drops,
                                 num_round, batch_size):
    train_indices, train_values, train_labels = train_data
    test_indices, test_values, test_labels = test_data
    mlp_model = multi_layer_perceptron(name=TAG, eval_metric='softmax_log_loss',
                                       layer_sizes=layer_sizes,
                                       layer_activates=layer_activates,
                                       opt_algo=opt_algo,
                                       learning_rate=learning_rate)
    for j in range(num_round):
        start_time = time.time()
        train_loss, train_y, train_y_prob = train_with_batch_csr(mlp_model, train_indices, train_values,
                                                                 train_labels, drops, batch_size, verbose=False)
        train_score = log_loss(train_labels, train_y_prob)
        print '[%d]\tloss: %f \ttrain_score: %f\ttime: %d' % \
              (j, train_loss.mean(), train_score, time.time() - start_time)

    test_y, test_y_prob = predict_with_batch_csr(mlp_model, test_indices, test_values, [1] * len(drops),
                                                 batch_size=batch_size)
    mlp_model.dump()
    make_submission(test_y_prob)


# def get_labels(train_size, valid_size):
#     global NUM_CLASS
#     data = np.loadtxt('../feature/device_id', delimiter=',', usecols=[1], skiprows=1, dtype=np.int64)
#     valid_labels = []
#     train_labels = []
#     for i in range(valid_size):
#         tmp = [0] * NUM_CLASS
#         tmp[data[i]] = 1
#         valid_labels.append(tmp)
#     for i in range(valid_size, valid_size + train_size):
#         tmp = [0] * NUM_CLASS
#         tmp[data[i]] = 1
#         train_labels.append(tmp)
#     return np.array(train_labels), np.array(valid_labels)


# def get_model_preds(model_name_list):
#     model_preds = []
#
#     for mn in model_name_list:
#         data = np.loadtxt('../feature/' + mn, delimiter=' ', skiprows=1, dtype=str)
#         pred = []
#         for i in range(12):
#             pred.append(map(lambda x: [float(x.split(':')[1])], data[:, i]))
#         pred = np.hstack(pred)
#         model_preds.append(pred)
#     model_preds = np.array(model_preds)
#     return model_preds


# def average_predict(model_preds, train_size, valid_size, train_labels, valid_labels, model_weights=None):
#     if model_weights is None:
#         average_preds = np.mean(model_preds, axis=0)
#     else:
#         weighted_model_preds = []
#         for i in range(len(model_preds)):
#             weighted_model_preds.append(model_preds[i] * model_weights[i])
#         average_preds = np.mean(weighted_model_preds, axis=0)
#     valid_pred = average_preds[:valid_size]
#     train_pred = average_preds[valid_size:(valid_size + train_size)]
#     train_score = log_loss(train_labels, train_pred)
#     valid_score = log_loss(valid_labels, valid_pred)
#     return train_score, valid_score


# def train_gblinear_get_result(train_round, train_alpha, train_lambda):
#     global PATH_TRAIN, PATH_TEST
#     dtrain = xgb.DMatrix(PATH_TRAIN)
#     dtest = xgb.DMatrix(PATH_TEST)
#     train_gblinear(dtrain, dtest, train_round, train_alpha, train_lambda)


# def train_gbtree_find_argument(argument_file_name):
#     global PATH_TRAIN, PATH_TEST
#     dtrain_train = xgb.DMatrix(PATH_TRAIN_TRAIN)
#     dtrain_valid = xgb.DMatrix(PATH_TRAIN_VALID)
#
#     max_depth = 4
#     subsample = 0.8
#
#     fout = open(argument_file_name, 'a')
#     for eta in [0.1, 0.15, 0.2, 0.3]:
#         print 'eta', eta
#         fout.write('eta ' + str(eta) + '\n')
#         for colsample_bytree in [0.6, 0.7, 0.8, 0.9]:
#             train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree, True)
#             print 'colsample_bytree', colsample_bytree, train_score, valid_score
#             fout.write('colsample_bytree ' + str(colsample_bytree) + ' ' + str(train_score) + ' ' +
#                        str(valid_score) + '\n')


# def train_gbtree_confirm_argument(max_depth=4, eta=0.3, subsample=0.7, colsample_bytree=0.7, verbose_eval=False):
#     global PATH_TRAIN, PATH_TEST
#     dtrain_train = xgb.DMatrix(PATH_TRAIN_TRAIN)
#     dtrain_valid = xgb.DMatrix(PATH_TRAIN_VALID)
#
#     train_score, valid_score = tune_gbtree(dtrain_train, dtrain_valid, eta, max_depth, subsample,
#                                            colsample_bytree, verbose_eval)
#     print train_score, valid_score


# def train_gbtree_get_result():
#     global PATH_TRAIN, PATH_TEST
#     dtrain = xgb.DMatrix(PATH_TRAIN)
#     dtest = xgb.DMatrix(PATH_TEST)
#
#     num_boost_round = 100
#     eta = 0.1
#     max_depth = 4
#     subsample = 0.8
#     colsample_bytree = 0.8
#
#     train_gbtree(dtrain, dtest, num_boost_round, eta, max_depth, subsample, colsample_bytree)


if __name__ == '__main__':
    init_constant(dataset='concat_6', booster=None, version=0)
    fin = open(PATH_TRAIN_TRAIN, 'r')
    train_indices, train_values, train_shape, train_labels = read_csr_feature(fin, -1)
    Xtrain = csr_matrix((train_values, (train_indices[:, 0], train_indices[:, 1])), shape=train_shape)
    fin = open(PATH_TRAIN_VALID, 'r')
    valid_indices, valid_values, valid_shape, valid_labels = read_csr_feature(fin, -1)
    Xvalid = csr_matrix((valid_values, (valid_indices[:, 0], valid_indices[:, 1])), shape=valid_shape)

    clf = RandomForestClassifier(n_estimators=200, n_jobs=8, max_depth=None)
    clf.fit(Xtrain, train_labels)
    train_pred = clf.predict_proba(Xtrain)
    train_pred = np.transpose(np.array(map(lambda x: x[:, 0], train_pred)))

    train_pred = np.array(map(lambda x: np.exp(x) / np.sum(np.exp(x)), train_pred))

    # train_pred = np.transpose(np.array(map(lambda x: x[:, 0], train_pred))) / (NUM_CLASS-1)
    train_score = log_loss(train_labels, train_pred)
    valid_pred = clf.predict_proba(Xvalid)
    valid_pred = np.transpose(np.array(map(lambda x: x[:, 0], valid_pred)))

    valid_pred = np.array(map(lambda x: np.exp(x) / np.sum(np.exp(x)), valid_pred))

    # valid_pred = np.transpose(np.array(map(lambda x: x[:, 0], valid_pred))) / (NUM_CLASS-1)
    valid_score = log_loss(valid_labels, valid_pred)
    print train_score, valid_score
