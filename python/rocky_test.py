import xgboost as xgb

import train_impl as ti

ti.init_constant(dataset='concat_2_norm', booster='gblinear', version=1, random_state=0)

if __name__ == '__main__':
    if ti.BOOSTER == 'gblinear':
        dtrain_train = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dtrain_valid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        # train_score, valid_score = tune_gblinear(dtrain, dvalid, 0, 7.5, True)
        # train_score, valid_score = tune_gblinear(dtrain, dvalid, 0.001, 8, True, dtest)
        # print train_score, valid_score
        # train_gblinear(dtrain_complete, dtest, 0.001, 8, 140)

        for gblinear_alpha in [0, 0.001, 0.005, 0.01, 0.02]:
            for gblinear_lambda in [7, 7.5, 8, 8.5, 9, 9.5, 10]:
                train_score, valid_score = ti.tune_gblinear(dtrain_train, dtrain_valid, gblinear_alpha,
                                                            gblinear_lambda, False)
                print 'alpha', gblinear_alpha, 'lambda', gblinear_lambda, train_score, valid_score
    elif ti.BOOSTER == 'gbtree':
        dtrain_train = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dtrain_valid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.07, 7, 0.8, 0.5, True)
        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.07, 3, 0.8, 0.6, True, dtest)
        # print train_score, valid_score
        ti.train_gbtree(dtrain, dtest, 0.07, 3, 0.8, 0.6, 500)

        # max_depth = 3
        # eta = 0.1
        # subsample = 0.7
        # colsample_bytree = 0.7
        # for max_depth in [3,4,5,6,7]:
        #    for eta in [0.05,0.06,0.07,0.09,0.1]:
        #        for subsample in [0.7,0.8,0.9]:
        #            for colsample_bytree in [0.5,0.6,0.7]:
        #                train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample,
        #                                   colsample_bytree, False)
        #                print 'max_depth',max_depth,'eta', eta,'subsample', subsample,'colsample_bytree',
        #                                   colsample_bytree, train_score, valid_score
    elif ti.BOOSTER == 'logistic_regression':
        pass
