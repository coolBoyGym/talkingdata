from math import log

import xgboost as xgb

import train_impl as ti
from model_impl import convolutional_neural_network

ti.init_constant(dataset='concat_6', booster='multi_layer_perceptron', version=200, random_state=0)

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

        train_score, valid_score = ti.tune_gbtree(dtrain, dvalid, eta=eta, max_depth=max_depth, subsample=subsample,
                                                  colsample_bytree=colsample_bytree, verbose_eval=True,
                                                  early_stopping_rounds=early_stopping_round, dtest=dtest)
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
        dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
        dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
        learning_rate = 0.01
        # gd, ftrl, adagrad, adadelta
        opt_algo = 'adagrad'
        factor_order = 10
        l1_w = 0
        l1_v = 0
        l2_w = 0
        l2_v = 0
        l2_b = 0
        num_round = 200
        batch_size = 10000
        early_stopping_round = 10
        for factor_order in [128, 256]:
            ti.tune_factorization_machine(dtrain_train, dtrain_valid, factor_order, opt_algo, learning_rate, l1_w=l1_w,
                                          l1_v=l1_v, l2_w=l2_w, l2_v=l2_v, l2_b=l2_b, num_round=num_round,
                                          batch_size=batch_size, early_stopping_round=early_stopping_round,
                                          verbose=True,
                                          save_log=True)
    elif ti.BOOSTER == 'multi_layer_perceptron':
        dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
        dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
        # dtrain = ti.read_feature(open(ti.PATH_TRAIN), -1, False)
        # dtest = ti.read_feature(open(ti.PATH_TEST), -1, False)
        layer_sizes = [ti.SPACE, 1024, 128, ti.NUM_CLASS]
        layer_activates = ['relu', 'relu', None]
        layer_inits = [('normal', 'zero'), ('normal', 'zero'), ('normal', 'zero')]
        layer_pool = [256, 64, None]
        drops = [0.5, 0.7, 1]
        opt_algo = 'gd'
        learning_rate = 0.5
        num_round = 500
        early_stopping_round = 10
        batch_size = 10000
        init_path = '../model/concat_6_tfidf_multi_layer_perceptron_1.bin'

        # for n in [1000, 800, 600, 400]:
        # for learning_rate in [0.4, 0.3, 0.2, 0.1]:
        # for opt_algo in ['gd']:
        ti.tune_multi_layer_perceptron(train_data=dtrain_train,
                                       valid_data=dtrain_valid,
                                       layer_sizes=layer_sizes,
                                       layer_activates=layer_activates,
                                       layer_inits=layer_inits,
                                       opt_algo=opt_algo,
                                       learning_rate=learning_rate,
                                       drops=drops,
                                       num_round=num_round,
                                       batch_size=batch_size,
                                       early_stopping_round=early_stopping_round,
                                       verbose=True,
                                       save_log=True,
                                       save_model=True,
                                       init_path=None,
                                       layer_pool=layer_pool)

        # ti.train_multi_layer_perceptron(dtrain, dtest, layer_sizes=layer_sizes, layer_activates=layer_activates,
        #                                 layer_inits=layer_inits, opt_algo=opt_algo, learning_rate=learning_rate,
        #                                 drops=drops,
        #                                 num_round=num_round, batch_size=10000)
    elif ti.BOOSTER == 'multiplex_neural_network':
        dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
        dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
        layer_sizes = [ti.SUB_SPACES, None, ti.NUM_CLASS]
        layer_activates = ['relu', None]
        drops = [0.5, 1]
        opt_algo = 'gd'
        learning_rate = 0.5
        num_round = 500
        batch_size = 10000
        for alpha in [1, 2, 4, 8, 16, 32, 64]:
            layer_sizes = [ti.SUB_SPACES, None, ti.NUM_CLASS]
            layer_sizes[1] = map(lambda x: int(alpha * log(x)), layer_sizes[0])
            print layer_sizes
            ti.tune_multiplex_neural_network(dtrain_train, dtrain_valid, layer_sizes, layer_activates, opt_algo,
                                             learning_rate, drops, num_round=num_round, batch_size=batch_size,
                                             early_stopping_round=10, verbose=True, save_log=True, save_model=True)
    elif ti.BOOSTER == 'convolutional_neural_network':
        train_indices, train_values, train_labels = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
        layer_sizes = [ti.SUB_SPACES, 100, ti.NUM_CLASS]
        layer_activates = ['relu', None]
        cnn_model = convolutional_neural_network(ti.TAG, 'softmax_log_loss', layer_sizes, layer_activates, 'gd', 0.1)
    elif ti.BOOSTER == 'average':
        # model_name_list = ['concat_1_gblinear_1', 'concat_1_gbtree_1', 'concat_2_gblinear_1', 'concat_2_gbtree_1',
        #                    'concat_2_norm_gblinear_1', 'concat_2_norm_gbtree_1', 'concat_4_gbtree_1',
        #                    'concat_5_gbtree_1', 'concat_5_norm_gblinear_1', 'concat_5_norm_gbtree_1']
        #
        # train_size = 59716
        # valid_size = 14929
        # train_labels, valid_labels = ti.get_labels(train_size, valid_size)
        # model_preds = ti.get_model_preds(model_name_list)
        #
        # model_weights = np.array([0.01968983, 0.0391137, 0.01906632, 0.00886792, 0.0204471,
        #                           0.02016268, 0.25404595, 0.2369304, 0.22133675, 0.16033936, ])
        # model_weights /= model_weights.sum()
        # train_score, valid_score = ti.average_predict(model_preds, train_size=train_size, valid_size=valid_size,
        #                                               train_labels=train_labels, valid_labels=valid_labels,
        #                                               model_weights=model_weights)
        # print train_score, valid_score
        # # with open('../model/average_1.log', 'a') as fout:
        # #     while True:
        # #         model_weights = np.random.random([len(model_name_list)])
        # #         model_weights /= model_weights.sum()
        # #         train_score, valid_score = average_predict(model_preds, model_weights=model_weights)
        # #         print model_weights, train_score, valid_score
        # #         fout.write('\t'.join(map(lambda x: str(x), model_weights)) + '\t' + str(train_score) + '\t' + str(
        # #             valid_score) + '\n')
        pass
