import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

import train_impl as ti
from model_impl import logistic_regression

ti.init_constant(dataset='ensemble_1', booster='gblinear', version=1, random_state=0)

if __name__ == '__main__':
    if ti.BOOSTER == 'gblinear':
        dtrain = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dvalid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain_complete = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        early_stopping_round = 1
        # train_score, valid_score = ti.tune_gblinear(dtrain, dvalid, 1, 1000, True, early_stopping_round)
        # train_score, valid_score = ti.tune_gblinear(dtrain, dvalid, 0.001, 10, True,
        #                                             early_stopping_rounds=early_stopping_round, dtest=dtest)
        # print train_score, valid_score
        # ti.train_gblinear(dtrain_complete, dtest, 0.001, 10, 2)

        for gblinear_alpha in [0]:
            for gblinear_lambda in [100]:
                for gblinear_lambda_bias in [0]:
                    train_score, valid_score = ti.tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda,
                                                                gblinear_lambda_bias=gblinear_lambda_bias,
                                                                verbose_eval=True,
                                                                early_stopping_rounds=early_stopping_round)
                    # print gblinear_alpha, gblinear_lambda, train_score, valid_score
                    # write_log('%f\t%f\t%f\t%f\n' % (gblinear_alpha, gblinear_lambda, train_score, valid_score))
    elif ti.BOOSTER == 'gbtree':
        dtrain = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dvalid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain_complete = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        early_stopping_round = 50

        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 10, 0.0001, 0.0001, True, early_stopping_rounds=50)
        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 4, 0.7, 0.7, True, dtest)
        # print train_score, valid_score
        # train_gbtree(dtrain_complete, dtest, 0.1, 3, 0.7, 0.7, 300)

        # max_depth = 3
        # eta = 0.1
        # subsample = 0.7
        # colsample_bytree = 0.7

        # start_time = time.time()
        # colsample_bytree = 0.7
        for max_depth in [1, 2]:
            for eta in [0.01]:
                for subsample in [0.01]:
                    for colsample_bytree in [0.2]:
                        train_score, valid_score = ti.tune_gbtree(dtrain, dvalid, eta, max_depth, subsample,
                                                                  colsample_bytree,
                                                                  True, early_stopping_rounds=early_stopping_round)
                        # print max_depth, eta, subsample, train_score, valid_score, time.time() - start_time
    elif ti.BOOSTER == 'logistic_regression':
        for l1_alpha in [0]:
            for l2_lambda in [10]:
                print '##########################################################################'
                print l1_alpha, l2_lambda
                lr_model = logistic_regression(name=ti.TAG, eval_metric='softmax_log_loss', num_class=12,
                                               input_space=ti.SPACE, l1_alpha=l1_alpha, l2_lambda=l2_lambda,
                                               optimizer='adadelta', learning_rate=0.1, )

                # lr_model.write_log_header()
                # lr_model.write_log('loss\ttrain-score\tvalid_score')
                num_round = 500
                batch_size = -1
                train_indices, train_values, train_labels = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1, False)
                valid_indices, valid_values, valid_labels = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1, False)
                for j in range(num_round):
                    start_time = time.time()
                    # train_loss, train_preds, train_labels = train_with_batch(path_train + '.train', batch_size)
                    # valid_preds, valid_labels = predict_with_batch(path_train + '.valid', batch_size)
                    train_loss, train_y, train_y_prob = ti.train_with_batch_csr(lr_model, train_indices, train_values,
                                                                                train_labels, batch_size)
                    valid_y, valid_y_prob = ti.predict_with_batch_csr(lr_model, valid_indices, valid_values, batch_size)
                    train_score = log_loss(train_labels, train_y_prob)
                    valid_score = log_loss(valid_labels, valid_y_prob)
                    print 'loss: %f \ttrain_score: %f\tvalid_score: %f\ttime: %d' % (
                        train_loss.mean(), train_score, valid_score, time.time() - start_time)
                    lr_model.write_log('%d\t%f\t%f\t%f\n' % (j, train_loss.mean(), train_score, valid_score))
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
