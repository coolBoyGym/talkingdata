import xgboost as xgb

import train_impl as ti

ti.init_constant(dataset='ensemble_2', booster='gbtree', version=1, random_state=0)

if __name__ == '__main__':
    if ti.BOOSTER == 'gblinear':
        dtrain_train = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dtrain_valid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        # train_score, valid_score = tune_gblinear(dtrain, dvalid, 0.1, 66, True)
        train_score, valid_score = ti.tune_gblinear(dtrain_train, dtrain_valid, 0.1, 13, True, dtest)
        # print train_score, valid_score
        # train_gblinear(dtrain_complete, dtest, 100, 1, 10)

        # print train_score, valid_score
        # fout = open(path_argument_file, 'a')
        #
        # for gblinear_alpha in [0.1]:
        #     for gblinear_lambda in [13]:
        #         train_score, valid_score = tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda, True)
        #         print 'alpha', gblinear_alpha, 'lambda', gblinear_lambda, train_score, valid_score
        #         fout.write('alpha ' + str(gblinear_alpha) + ' lambda ' + str(gblinear_lambda) + ' '
        #                    + str(train_score) + ' ' + str(valid_score) + '\n')

    # elif ti.BOOSTER == 'gbtree':
    #     dtrain_train = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
    #     dtrain_valid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
    #     dtrain = xgb.DMatrix(ti.PATH_TRAIN)
    #     dtest = xgb.DMatrix(ti.PATH_TEST)
        # for gblinear_alpha in [0.007, 0.008, 0.009, 0.01, 0.02]:
        #     print 'alpha', gblinear_alpha
        #     for gblinear_lambda in [13, 14, 15]:
        #         train_score, valid_score = tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda, False)
        #         print 'lambda', gblinear_lambda, np.mean(train_score), np.mean(valid_score)
    elif ti.BOOSTER == 'gbtree':
        dtrain_train = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dtrain_valid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        # train_score, valid_score = ti.tune_gbtree(dtrain_train, dtrain_valid, 0.01, 2, 0.01, 0.01, 19, 0.1,
        #                                           early_stopping_rounds=50, verbose_eval=True)
        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 3, 0.8, 0.7, True, dtest)
        # print train_score, valid_score
        ti.train_gbtree(dtrain, dtest, 0.01, 2, 0.01, 0.01, 19, 0.1, 3980)

        # max_depth = 3
        # eta = 0.1
        # subsample = 0.7
        # colsample_bytree = 0.7

        # fout = open(ti.PATH_MODEL_LOG, 'a')
        # print 'Training begin!'
        #
        # for max_depth in [2]:
        #     for eta in [0.01]:
        #         for subsample in [0.01]:
        #             for colsample_bytree in [0.01]:
        #                 for gbtree_lambda in [19]:
        #                     for gbtree_alpha in [0.1]:
        #                         train_score, valid_score = ti.tune_gbtree(dtrain_train, dtrain_valid, eta,
        #                                                                   max_depth, subsample,
        #                                                                   colsample_bytree,
        #                                                                   gbtree_lambda, gbtree_alpha,
        #                                                                   early_stopping_rounds=50,
        #                                                                   verbose_eval=True)
        #                         fout.write('max_depth ' + str(max_depth) + ' eta ' + str(eta) + ' subsample ' +
        #                                    str(subsample) + ' colsample ' + str(colsample_bytree) + ' lambda ' +
        #                                    str(gbtree_lambda) + ' alpha ' + str(gbtree_alpha) + ' ' +
        #                                    str(train_score) + ' ' + str(valid_score) + '\n')

    elif ti.BOOSTER == 'logistic_regression':
        pass
