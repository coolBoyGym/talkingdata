from train_impl import *

init_constant(dataset='concat_5', booster='gbtree', version=1, random_state=0)


def train_gblinear_confirm_argument(xgb_alpha=0.1, xgb_lambda=0.1, verbose_eval=False):
    dtrain = [xgb.DMatrix(PATH_TRAIN + '.%d.train' % i) for i in range(nfold)]
    dvalid = [xgb.DMatrix(PATH_TRAIN + '.%d.valid' % i) for i in range(nfold)]

    train_score, valid_score = tune_gblinear(dtrain, dvalid, xgb_alpha, xgb_lambda, verbose_eval)
    print np.mean(train_score), np.mean(valid_score)


def train_gblinear_find_argument(argument_file_name, show_message=False):
    # argument_file_name = '../output/argument.concat_3.gblinear'
    dtrain = [xgb.DMatrix(PATH_TRAIN + '.%d.train' % i) for i in range(nfold)]
    dvalid = [xgb.DMatrix(PATH_TRAIN + '.%d.valid' % i) for i in range(nfold)]

    fout = open(argument_file_name, 'a')
    for gblinear_alpha in [0.1, 0.2, 0.3]:
        print 'alpha', gblinear_alpha
        fout.write('alpha ' + str(gblinear_alpha) + '\n')
        for gblinear_lambda in [0.2, 0.5, 2]:
            train_score, valid_score = tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda, show_message)
            print 'lambda', gblinear_lambda, np.mean(train_score), np.mean(valid_score)
            fout.write('lambda ' + str(gblinear_lambda) + ' ' + str(np.mean(train_score)) + ' ' +
                       str(np.mean(valid_score)) + '\n')


def train_gblinear_get_result(train_round, train_alpha, train_lambda):
    dtrain = xgb.DMatrix(PATH_TRAIN)
    dtest = xgb.DMatrix(PATH_TEST)
    train_gblinear(dtrain, dtest, train_round, train_alpha, train_lambda)


def train_gbtree_find_argument(argument_file_name):
    # argument_file_name = '../output/argument.concat_3.gblinear'

    dtrain = xgb.DMatrix(PATH_TRAIN + '.train')
    dvalid = xgb.DMatrix(PATH_TRAIN + '.valid')

    max_depth = 4
    subsample = 0.8

    fout = open(argument_file_name, 'a')
    for eta in [0.1, 0.15, 0.2, 0.3]:
        print 'eta', eta
        fout.write('eta ' + str(eta) + '\n')
        for colsample_bytree in [0.6, 0.7, 0.8, 0.9]:
            train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample, colsample_bytree, True)
            print 'colsample_bytree', colsample_bytree, train_score, valid_score
            fout.write('colsample_bytree ' + str(colsample_bytree) + ' ' + str(train_score) + ' ' +
                       str(valid_score) + '\n')


def train_gbtree_confirm_argument(max_depth=4, eta=0.3, subsample=0.7, colsample_bytree=0.7, verbose_eval=False):
    dtrain = xgb.DMatrix(PATH_TRAIN + '.train')
    dvalid = xgb.DMatrix(PATH_TRAIN + '.valid')

    train_score, valid_score = tune_gbtree(dtrain, dvalid, eta, max_depth, subsample,
                                           colsample_bytree, verbose_eval)
    print train_score, valid_score


def train_gbtree_get_result():
    dtrain = xgb.DMatrix(PATH_TRAIN)
    dtest = xgb.DMatrix(PATH_TEST)

    num_boost_round = 100
    eta = 0.1
    max_depth = 4
    subsample = 0.8
    colsample_bytree = 0.8

    train_gbtree(dtrain, dtest, num_boost_round, eta, max_depth, subsample, colsample_bytree)


if __name__ == '__main__':
    train_gbtree_find_argument('../output/argument.concat_5.gbtree')

    # train_gblinear_find_argument('../output/argument.concat_3.gblinear')
    # train_gblinear_confirm_argument(0.2, 0.86, True)
    # train_gblinear_get_result(3, 0.2, 0.86)
    # print('main function finish!')
