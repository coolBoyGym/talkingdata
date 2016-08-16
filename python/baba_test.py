import numpy as np
import xgboost as xgb

import train_impl as ti

ti.init_constant(dataset='ensemble_4', booster='multi_layer_perceptron', version=1, random_state=0)

if __name__ == '__main__':
    if ti.BOOSTER == 'gblinear':
        dtrain = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dvalid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain_complete = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        early_stopping_round = 1
        train_score, valid_score = ti.tune_gblinear(dtrain, dvalid, gblinear_alpha=0, gblinear_lambda=1000,
                                                    verbose_eval=True, early_stopping_rounds=early_stopping_round)
        # train_score, valid_score = ti.tune_gblinear(dtrain, dvalid, 0.001, 10, True,
        #                                             early_stopping_rounds=early_stopping_round, dtest=dtest)
        print train_score, valid_score
        # ti.train_gblinear(dtrain_complete, dtest, 0.001, 10, 2)

        # for gblinear_alpha in [0]:
        #     for gblinear_lambda in [100]:
        #         for gblinear_lambda_bias in [0]:
        #             train_score, valid_score = ti.tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda,
        #                                                         gblinear_lambda_bias=gblinear_lambda_bias,
        #                                                         verbose_eval=True,
        #                                                         early_stopping_rounds=early_stopping_round)
        #             # print gblinear_alpha, gblinear_lambda, train_score, valid_score
        #             # write_log('%f\t%f\t%f\t%f\n' % (gblinear_alpha, gblinear_lambda, train_score, valid_score))
    elif ti.BOOSTER == 'gbtree':
        dtrain = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dvalid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        # dtrain_complete = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        max_depth = 3
        eta = 0.1
        subsample = 0.7
        colsample_bytree = 0.7
        early_stopping_round = 50

        train_score, valid_score = ti.tune_gbtree(dtrain, dvalid, eta=0.1, max_depth=3, subsample=0.7,
                                                  colsample_bytree=0.7, verbose_eval=True,
                                                  early_stopping_rounds=50, dtest=dtest)
        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 4, 0.7, 0.7, True, dtest)
        # print train_score, valid_score
        # train_gbtree(dtrain_complete, dtest, 0.1, 3, 0.7, 0.7, 300)

        # start_time = time.time()
        # colsample_bytree = 0.7
        # for max_depth in [3]:
        #     for eta in [0.1]:
        #         for subsample in [0.7]:
        #             for colsample_bytree in [0.7]:
        #                 train_score, valid_score = ti.tune_gbtree(dtrain, dvalid, eta, max_depth, subsample,
        #                                                           colsample_bytree, verbose_eval=True,
        #                                                           early_stopping_rounds=early_stopping_round)
        #                 # print max_depth, eta, subsample, train_score, valid_score, time.time() - start_time
    elif ti.BOOSTER == 'logistic_regression':
        pass
    elif ti.BOOSTER == 'factorization_machine':
        dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1, False)
        dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1, False)
        learning_rate = 0.1
        # gd, ftrl, adagrad, adadelta
        opt_algo = 'gd'
        factor_order = 10
        l1_w = 0
        l1_v = 0
        l2_w = 1
        l2_v = 1
        l2_b = 0
        num_round = 200
        batch_size = 10000
        early_stopping_round = 10
        ti.tune_factorization_machine(dtrain_train, dtrain_valid, factor_order, opt_algo, learning_rate, l1_w=l1_w,
                                      l1_v=l1_v, l2_w=l2_w, l2_v=l2_v, l2_b=l2_b, num_round=num_round,
                                      batch_size=batch_size, early_stopping_round=early_stopping_round, verbose=True,
                                      save_log=True)
    elif ti.BOOSTER == 'multi_layer_perceptron':
        dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1, False)
        dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1, False)
        # dtrain = ti.read_feature(open(ti.PATH_TRAIN), -1, False)
        # dtest = ti.read_feature(open(ti.PATH_TEST), -1, False)
        layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
        layer_activates = ['relu', None]
        drops = [0.5, 0.5]
        opt_algo = 'gd'
        learning_rate = 0.2
        num_round = 500
        batch_size = 10000

        # for n in [1000, 800, 600, 400]:
        # for learning_rate in [0.4, 0.3, 0.2, 0.1]:
        # for opt_algo in ['gd']:
        ti.tune_multi_layer_perceptron(dtrain_train, dtrain_valid, layer_sizes, layer_activates, opt_algo,
                                       learning_rate, drops, num_round=num_round, batch_size=batch_size,
                                       early_stopping_round=10, verbose=True, save_log=True, save_model=True)

        # ti.train_multi_layer_perceptron(dtrain, dtest, layer_sizes, layer_activates, opt_algo, learning_rate, drops,
        #                                 num_round=num_round, batch_size=10000)
    elif ti.BOOSTER == 'average':
        model_name_list = ['concat_1_gblinear_1', 'concat_1_gbtree_1', 'concat_2_gblinear_1', 'concat_2_gbtree_1',
                           'concat_2_norm_gblinear_1', 'concat_2_norm_gbtree_1', 'concat_4_gbtree_1',
                           'concat_5_gbtree_1', 'concat_5_norm_gblinear_1', 'concat_5_norm_gbtree_1']

        train_size = 59716
        valid_size = 14929
        train_labels, valid_labels = ti.get_labels(train_size, valid_size)
        model_preds = ti.get_model_preds(model_name_list)

        model_weights = np.array([0.01968983, 0.0391137, 0.01906632, 0.00886792, 0.0204471,
                                  0.02016268, 0.25404595, 0.2369304, 0.22133675, 0.16033936, ])
        model_weights /= model_weights.sum()
        train_score, valid_score = ti.average_predict(model_preds, train_size=train_size, valid_size=valid_size,
                                                      train_labels=train_labels, valid_labels=valid_labels,
                                                      model_weights=model_weights)
        print train_score, valid_score
        # with open('../model/average_1.log', 'a') as fout:
        #     while True:
        #         model_weights = np.random.random([len(model_name_list)])
        #         model_weights /= model_weights.sum()
        #         train_score, valid_score = average_predict(model_preds, model_weights=model_weights)
        #         print model_weights, train_score, valid_score
        #         fout.write('\t'.join(map(lambda x: str(x), model_weights)) + '\t' + str(train_score) + '\t' + str(
        #             valid_score) + '\n')
