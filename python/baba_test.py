import xgboost as xgb

from sklearn.metrics import log_loss

version = 1
booster = 'gblinear'

dataset = 'concat_1'

path_train = '../input/' + dataset + '.train'
path_test = '../input/' + dataset + '.train'

tag = '%s.%s.%d' % (dataset, booster, version)
path_model_log = '../model/' + tag + '.log'
path_model_bin = '../model/' + tag + '.model'
path_model_dump = '../model/' + tag + '.dump'

path_submission = '../output/' + tag + '.submission'

# fea_model = feature.multi_feature(name=tag, dtype='f', space=12, rank=12)

random_state = 0


def tune_gbtree(dtrain, dvalid):
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

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)

    valid_pred = bst.predict(dvalid, ntree_limit=bst.best_iteration)
    valid_score = log_loss(dvalid.get_label(), valid_pred)
    print 'validation on best iteration', valid_score


def tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda):
    num_boost_round = 1000
    early_stopping_rounds = 10

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
                    verbose_eval=False)

    train_pred = bst.predict(dtrain)
    train_score = log_loss(dtrain.get_label(), train_pred)

    valid_pred = bst.predict(dvalid)
    valid_score = log_loss(dvalid.get_label(), valid_pred)

    return train_score, valid_score

    # bst.save_model(path_model_bin)
    # bst.dump_model(path_model_dump)
    # print 'save model at', path_model_bin, path_model_dump

    # test_pred = bst.predict(xgb.DMatrix(path_test), ntree_limit=bst.best_iteration)
    # test_pred = bst.predict(xgb.DMatrix(path_test))

    # with open(path_submission, 'w') as fout:
    #     fout.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    #     with open('../data/gender_age_test.csv') as fin:
    #         next(fin)
    #         cnt = 0
    #         for line in fin:
    #             did = line.strip().split(',')[0]
    #             fout.write('%s,%s\n' % (did, ','.join(map(lambda d: str(d), test_pred[cnt]))))
    #             cnt += 1
    #
    # print path_submission

dtrain = xgb.DMatrix(path_train + '.train')
dvalid = xgb.DMatrix(path_train + '.valid')

for gblinear_alpha in [0, 0.01, 0.005, 0.1]:
    print 'alpha', gblinear_alpha
    for gblinear_lambda in range(30, 40):
        train_score, valid_score = tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda)
        print gblinear_lambda, train_score, valid_score


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
