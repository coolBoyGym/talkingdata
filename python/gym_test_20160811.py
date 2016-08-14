import xgboost as xgb

import train_impl as ti
from model_impl import opt_property

ti.init_constant(dataset='concat_10', booster='gbtree', version=1, random_state=0)

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

        # train_score, valid_score = ti.tune_gbtree(dtrain_train, dtrain_valid, 0.1, 3, 0.7, 0.8, 1, 0.1,
        #                                           early_stopping_rounds=50, verbose_eval=True, dtest=dtest)
        # train_score, valid_score = ti.tune_gbtree(dtrain, dvalid, 0.1, 3, 0.8, 0.7, True, dtest)
        # print train_score, valid_score
        # ti.train_gbtree(dtrain, dtest, 0.1, 3, 0.7, 0.8, 1, 0.1, 720)

        # max_depth = 3
        # eta = 0.1
        # subsample = 0.7
        # colsample_bytree = 0.7

        fout = open(ti.PATH_MODEL_LOG, 'a')
        print 'Training begin!'

        for max_depth in [3]:
            for eta in [0.1]:
                for subsample in [0.5, 0.6, 0.9, 1]:
                    for colsample_bytree in [0.8]:
                        for gbtree_lambda in [1]:
                            for gbtree_alpha in [0.1]:
                                train_score, valid_score = ti.tune_gbtree(dtrain_train, dtrain_valid, eta,
                                                                          max_depth, subsample,
                                                                          colsample_bytree,
                                                                          gbtree_lambda, gbtree_alpha,
                                                                          early_stopping_rounds=30,
                                                                          verbose_eval=True)
                                fout.write('max_depth ' + str(max_depth) + ' eta ' + str(eta) + ' subsample ' +
                                           str(subsample) + ' colsample ' + str(colsample_bytree) + ' lambda ' +
                                           str(gbtree_lambda) + ' alpha ' + str(gbtree_alpha) + ' ' +
                                           str(train_score) + ' ' + str(valid_score) + '\n')

    elif ti.BOOSTER == 'logistic_regression':
        pass

    elif ti.BOOSTER == 'factorization_machine':
        train_data = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1, False)
        valid_data = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1, False)
        learning_rate = 0.3
        # gd, ftrl, adagrad, adadelta
        opt_prop = opt_property('adagrad', learning_rate)
        # argument numbers
        factor_order = 10
        l1_w = 0
        l1_v = 0
        l2_w = 0
        l2_v = 0
        l2_b = 0
        num_round = 200
        batch_size = 100
        early_stopping_round = 10
        ti.tune_factorization_machine(train_data, valid_data, factor_order, opt_prop, l1_w=l1_w, l1_v=l1_v,
                                      l2_w=l2_w, l2_v=l2_v, l2_b=l2_b, num_round=num_round, batch_size=batch_size,
                                      early_stopping_round=early_stopping_round, verbose=True, save_log=False)
