from random import random
from time import *

import xgboost as xgb

cv_rate = 0.75
version = 2


path_train = '../input/train.csv.%.2f.%d' % (cv_rate, version)
path_valid = '../input/valid.csv.%.2f.%d' % (cv_rate, version)
path_test = '../input/test.csv'

path_model_bin = '../model/xgb.model.%.2f.%d' % (cv_rate, version)
path_model_dump = '../model/dunp.raw.txt.%.2f.%d' % (cv_rate, version)

path_submission = '../output/submission.csv.%.2f.%d' % (cv_rate, version)


def split_file():
    print 'cv rate', cv_rate, 'train/valid path', path_train, path_valid

    with open('../data/train_brand_model_installed_active.csv') as fin:
        with open(path_train, 'w') as fout_train:
            with open(path_valid, 'w') as fout_valid:
                for line in fin:
                    if random() < cv_rate:
                        fout_train.write(line)
                    else:
                        fout_valid.write(line)


def wrap_test():
    with open('../data/test_brand_model_installed_active.csv', 'r') as fin:
        with open(path_test, 'w') as fout:
            for line in fin:
                fout.write('0 ' + line)


def xgb_train():
    # X_train, X_valid = train_test_split()
    dtrain = xgb.DMatrix(path_train)
    dvalid = xgb.DMatrix(path_valid)

    eta = 0.1
    random_state = 0
    num_boost_round = 1000
    early_stopping_rounds = 50
    max_depth = 3
    subsample = 0.7
    colsample_bytree = 0.7

    params = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster": "gbtree",
        "eval_metric": "mlogloss",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)

    gbm.save_model(path_model_bin)
    gbm.dump_model(path_model_dump)


def xgb_test():
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model(path_model_bin)

    pred = bst.predict(xgb.DMatrix(path_test))

    with open(path_submission, 'w') as fout:
        fout.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
        with open('../data/gender_age_test.csv') as fin:
            next(fin)
            cnt = 0
            for line in fin:
                did = line.strip().split(',')[0]
                fout.write('%s,%s\n' % (did, ','.join(map(lambda d: str(d), pred[cnt]))))
                cnt += 1


split_file()
xgb_train()
xgb_test()
