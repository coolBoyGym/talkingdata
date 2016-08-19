import xgboost as xgb

import train_impl as ti

import model_impl

ti.init_constant(dataset='concat_13', booster='gbtree', version=1, random_state=0)

if __name__ == '__main__':
    if ti.BOOSTER == 'gblinear':
        dtrain_train = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dtrain_valid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        # train_score, valid_score = tune_gblinear(dtrain, dvalid, 0.1, 66, True)
        # train_score, valid_score = ti.tune_gblinear(dtrain_train, dtrain_valid, 0.1, 13, True, dtest)
        # print train_score, valid_score
        # train_gblinear(dtrain_complete, dtest, 100, 1, 10)

        # print train_score, valid_score
        fout = open('../model/concat_10_gblinear.log', 'a')
        early_stopping_round = 10

        for gblinear_alpha in [0, 0.01, 0.05, 0.2]:
            for gblinear_lambda in [45]:
                train_score, valid_score = ti.tune_gblinear(dtrain_train, dtrain_valid, gblinear_alpha,
                                                            gblinear_lambda, True,
                                                            early_stopping_rounds=early_stopping_round)
                print 'alpha', gblinear_alpha, 'lambda', gblinear_lambda, train_score, valid_score
                fout.write('alpha ' + str(gblinear_alpha) + ' lambda ' + str(gblinear_lambda) + ' '
                           + str(train_score) + ' ' + str(valid_score) + '\n')

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
                for subsample in [0.6, 0.7]:
                    for colsample_bytree in [0.5, 0.6, 0.7, 0.9]:
                        for gbtree_lambda in [1]:
                            for gbtree_alpha in [0.1]:
                                train_score, valid_score = ti.tune_gbtree(dtrain_train, dtrain_valid, eta,
                                                                          max_depth, subsample,
                                                                          colsample_bytree,
                                                                          gbtree_lambda, gbtree_alpha,
                                                                          early_stopping_rounds=20,
                                                                          verbose_eval=True)
                                fout.write('max_depth ' + str(max_depth) + ' eta ' + str(eta) + ' subsample ' +
                                           str(subsample) + ' colsample ' + str(colsample_bytree) + ' lambda ' +
                                           str(gbtree_lambda) + ' alpha ' + str(gbtree_alpha) + ' ' +
                                           str(train_score) + ' ' + str(valid_score) + '\n')

    elif ti.BOOSTER == 'logistic_regression':
        pass

    elif ti.BOOSTER == 'multi_layer_perceptron':
        train_data = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
        valid_data = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
        # train_data = ti.read_feature(open(ti.PATH_TRAIN), -1)
        # test_data = ti.read_feature(open(ti.PATH_TEST), -1)
        opt_algo = 'gd'
        learning_rate = 0.2
        layer_activates = ['relu', None]
        layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
        drops = [0.5, 1]

        # ti.train_multi_layer_perceptron(train_data, test_data, layer_sizes, layer_activates, opt_algo, learning_rate,
        #                                 drops, num_round=469, batch_size=10000)
        #
        for learning_rate in [0.2]:
            ti.tune_multi_layer_perceptron(train_data, valid_data, layer_sizes, layer_activates,
                                           opt_algo=opt_algo, learning_rate=learning_rate,
                                           drops=drops, num_round=600, batch_size=10000, early_stopping_round=20,
                                           verbose=True, save_log=True, save_model=True)

    elif ti.BOOSTER == 'factorization_machine':
        train_data = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
        valid_data = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
        learning_rate = 0.3
        # gd, ftrl, adagrad, adadelta
        opt_algo = 'gd'
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
        ti.tune_factorization_machine(train_data, valid_data, factor_order, opt_algo, learning_rate, l1_w=l1_w,
                                      l1_v=l1_v, l2_w=l2_w, l2_v=l2_v, l2_b=l2_b, num_round=num_round,
                                      batch_size=batch_size, early_stopping_round=early_stopping_round, verbose=True,
                                      save_log=False)
