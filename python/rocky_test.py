import xgboost as xgb
from scipy.sparse import csr_matrix
import train_impl as ti

ti.init_constant(dataset='concat_6', booster='rdforest', version=1, random_state=0)

if __name__ == '__main__':
    if ti.BOOSTER == 'gblinear':
        dtrain_train = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dtrain_valid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        # train_score, valid_score = tune_gblinear(dtrain, dvalid, 0, 7.5, True)
        # train_score, valid_score = ti.tune_gblinear(dtrain_train, dtrain_valid,
                # gblinear_alpha=0, gblinear_lambda=10, verbose_eval=True,early_stopping_rounds=50, dtest=dtest)
        # print train_score, valid_score
        ti.train_gblinear(dtrain, dtest, 0, 10,0, 1)

        # for gblinear_alpha in [0,0.005, 0.01,0.015, 0.02, 0.03,0.05]:
        #     for gblinear_lambda in [1,2,5,7,8,9,10,12,15]:
        #         train_score, valid_score = ti.tune_gblinear(dtrain_train, dtrain_valid, gblinear_alpha=gblinear_alpha,
        #                                                     gblinear_lambda=gblinear_lambda, verbose_eval=False)
        #         print 'alpha', gblinear_alpha, 'lambda', gblinear_lambda, train_score, valid_score
    elif ti.BOOSTER == 'gbtree':
        dtrain_train = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
        dtrain_valid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
        dtrain = xgb.DMatrix(ti.PATH_TRAIN)
        dtest = xgb.DMatrix(ti.PATH_TEST)

        # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.07, 7, 0.8, 0.5, True)
        # train_score, valid_score = ti.tune_gbtree(dtrain_train, dtrain_valid, 0.05, 4, 0.7, 0.6,verbose_eval= True, dtest=dtest)
        # print train_score, valid_score
        # ti.train_gbtree(dtrain, dtest, 0.05, 4, 0.7, 0.6, 500)

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
    elif ti.BOOSTER == 'rdforest':
        fin = open(ti.PATH_TRAIN_TRAIN, 'r')
        train_indices, train_values, train_shape, train_labels = ti.read_csr_feature(fin, -1)
        Xtrain = csr_matrix((train_values, (train_indices[:, 0], train_indices[:, 1])), shape=train_shape)
        fin = open(ti.PATH_TRAIN_VALID, 'r')
        valid_indices, valid_values, valid_shape, valid_labels = ti.read_csr_feature(fin, -1)
        Xvalid = csr_matrix((valid_values, (valid_indices[:, 0], valid_indices[:, 1])), shape=valid_shape)
        fin = open(ti.PATH_TRAIN, 'r')
        wtrain_indices, wtrain_values, wtrain_shape, wtrain_labels = ti.read_csr_feature(fin, -1)
        wXtrain = csr_matrix((wtrain_values, (wtrain_indices[:, 0], wtrain_indices[:, 1])), shape=wtrain_shape)
        fin = open(ti.PATH_TEST, 'r')
        test_indices, test_values, test_shape, test_labels = ti.read_csr_feature(fin, -1)
        Xtest = csr_matrix((test_values, (test_indices[:, 0], test_indices[:, 1])), shape=test_shape)

        # for n_estimators in [50,100,200,300,500]:
        #     train_score, valid_score = ti.tune_rdforest(Xtrain, train_labels, Xvalid, valid_labels, n_estimators=n_estimators)
        #     print 'n_estimators', n_estimators, train_score, valid_score
        train_score, valid_score = ti.tune_rdforest(Xtrain, train_labels, Xvalid, valid_labels, n_estimators=100, max_features=None)
        print train_score, valid_score
        # ti.train_rdforest(wXtrain, wtrain_labels, Xtest, n_estimators=100)
