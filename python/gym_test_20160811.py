from train_impl import *

init_constant(dataset='concat_5', booster='gbtree', version=1, random_state=0)

if __name__ == '__main__':
    if BOOSTER == 'gblinear':
        dtrain = xgb.DMatrix(PATH_TRAIN + '.train')
        dvalid = xgb.DMatrix(PATH_TRAIN + '.valid')
        dtrain_complete = xgb.DMatrix(PATH_TRAIN)
        dtest = xgb.DMatrix(PATH_TEST)

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
    elif BOOSTER == 'gbtree':
        dtrain = xgb.DMatrix(PATH_TRAIN + '.train')
        dvalid = xgb.DMatrix(PATH_TRAIN + '.valid')
        dtrain_complete = xgb.DMatrix(PATH_TRAIN)
        dtest = xgb.DMatrix(PATH_TEST)

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
                    for colsample_bytree in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                        train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample,
                                                               colsample_bytree, False)
                        print 'max_depth', max_depth, 'eta', eta, 'subsample', subsample, 'colsample', \
                            colsample_bytree, train_score, valid_score

    elif BOOSTER == 'logistic_regression':
        pass
