from train_impl import *

init_constant(dataset='concat_5', booster='gbtree', version=1, random_state=0)

if __name__ == '__main__':
    if BOOSTER == 'gblinear':
        dtrain = xgb.DMatrix(PATH_TRAIN + '.train')
        dvalid = xgb.DMatrix(PATH_TRAIN + '.valid')
        dtrain_complete = xgb.DMatrix(PATH_TRAIN)
        dtest = xgb.DMatrix(PATH_TEST)

        # train_score, valid_score = tune_gblinear(dtrain, dvalid, 0.1, 66, True)
        train_score, valid_score = tune_gblinear(dtrain, dvalid, 0.1, 13, True, dtest)
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

    elif booster == 'gbtree':
        dtrain = xgb.DMatrix(path_train + '.train')
        dvalid = xgb.DMatrix(path_train + '.valid')
        dtrain_complete = xgb.DMatrix(path_train)
        dtest = xgb.DMatrix(path_test)
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

        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 3, 0.8, 0.9, True)
        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 3, 0.8, 0.9, True, dtest)
        # print train_score, valid_score
        # train_gbtree(dtrain_complete, dtest, 0.1, 3, 0.8, 0.9, 950)

        # max_depth = 3
        # eta = 0.1
        # subsample = 0.7
        # colsample_bytree = 0.7
        #
        fout = open(path_argument_file, 'a')
        print 'Training begin!'

        for max_depth in [3]:
            for eta in [0.1]:
                for subsample in [0.8]:
                    for colsample_bytree in[0.5+p*0.05 for p in range(11)]:
                        train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample,
                                                               colsample_bytree, True)
                        fout.write('max_depth ' + str(max_depth) + ' eta ' + str(eta) + ' subsample ' + str(subsample)
                                   + ' colsample ' + str(colsample_bytree) + ' ' + str(train_score) + ' '
                                   + str(valid_score) + '\n')
                        print 'max_depth', max_depth, 'eta', eta, 'subsample', subsample, 'colsample', \
                            colsample_bytree, train_score, valid_score

    elif BOOSTER == 'logistic_regression':
        pass
