from random import random

import xgboost as xgb
from sklearn.metrics import log_loss

cv_rate = 0.75
version = 3
booster = 'gblinear'

path_train = '../input/train.csv.%.2f' % (cv_rate)
path_valid = '../input/valid.csv.%.2f' % (cv_rate)
path_test = '../input/test.csv'

path_model_bin = '../model/xgb.model.%.2f.%d.%s' % (cv_rate, version, booster)
path_model_dump = '../model/dunp.raw.txt.%.2f.%d.%s' % (cv_rate, version, booster)

path_submission = '../output/submission.csv.%.2f.%d.%s' % (cv_rate, version, booster)


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

    random_state = 0
    num_boost_round = 1000
    early_stopping_rounds = 50
    if booster == 'gbtree':
        eta = 0.1
        max_depth = 3
        subsample = 0.7
        colsample_bytree = 0.7

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
    elif booster == 'gblinear':
        xgb_alpha = 0.1
        xgb_lambda = 40

        params = {
            'booster': booster,
            'silent': 1,
            'num_class': 12,
            'lambda': xgb_lambda,
            'alpha': xgb_alpha,
            'objective': 'multi:softprob',
            'seed': random_state,
            'eval_metric': 'mlogloss',
        }

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)

    bst.save_model(path_model_bin)
    bst.dump_model(path_model_dump)
    print path_model_bin, path_model_dump

    if booster == 'gbtree':
        valid_pred = bst.predict(dvalid, ntree_limit=bst.best_iteration)
        valid_score = log_loss(dvalid.get_label(), valid_pred)
        print 'validation on best iteration', valid_score

        test_pred = bst.predict(xgb.DMatrix(path_test), ntree_limit=bst.best_iteration)
    elif booster == 'gblinear':
        valid_pred = bst.predict(dvalid)
        valid_score = log_loss(dvalid.get_label(), valid_pred)
        print 'validation', valid_score

        test_pred = bst.predict(xgb.DMatrix(path_test))

    with open(path_submission, 'w') as fout:
        fout.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
        with open('../data/gender_age_test.csv') as fin:
            next(fin)
            cnt = 0
            for line in fin:
                did = line.strip().split(',')[0]
                fout.write('%s,%s\n' % (did, ','.join(map(lambda d: str(d), test_pred[cnt]))))
                cnt += 1

    print path_submission


def xgb_test():
    bst = xgb.Booster()  # init model
    bst.load_model(path_model_bin)

    pred = bst.predict(xgb.DMatrix(path_test), ntree_limit=bst.best_iteration)

    with open(path_submission, 'w') as fout:
        fout.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
        with open('../data/gender_age_test.csv') as fin:
            next(fin)
            cnt = 0
            for line in fin:
                did = line.strip().split(',')[0]
                fout.write('%s,%s\n' % (did, ','.join(map(lambda d: str(d), pred[cnt]))))
                cnt += 1

    print path_submission


# split_file()
xgb_train()
# xgb_test()
