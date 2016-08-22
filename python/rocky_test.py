from task import Task
import feature
dataset = 'device_model'
booster = 'embedding'
version = 1

task = Task(dataset, booster, version)
if booster is 'gblinear':
    params = {
        'gblinear_alpha': 0,
        'gblinear_lambda': 100,
        'random_state': 0
    }
    num_round = 200
    early_stop_round = 10
    task.tune(params=params, num_round=num_round, early_stop_round=early_stop_round, save_log=True, save_model=False,
              dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, save_model=False, save_submission=False)
elif booster == 'gbtree':
    params = {
        'eta': 0.1,
        'max_depth': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gbtree_alpha': 0,
        'gbtree_lambda': 0,
        'random_state': 0
    }
    num_round = 1000
    early_stop_round = 50
    task.tune(params=params, num_round=num_round, early_stop_round=early_stop_round, save_log=True, save_model=False,
              dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, save_model=False, save_submission=False)
elif booster == 'mlp':
    layer_sizes = [task.space, 64, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    init_path = None
    layer_drops = [0.5, 1]
    opt_algo = 'gd'
    learning_rate = 0.2
    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }
    num_round = 3000
    early_stop_round = 20
    batch_size = 10000
    task.tune(params=params, batch_size=batch_size, num_round=num_round, early_stop_round=early_stop_round,
              verbose=True, save_log=True, save_model=True, dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=False,
    #            save_submission=False)
elif booster == 'mnn':
    layer_sizes = [task.sub_spaces, [32] * len(task.sub_spaces), task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    init_path = None
    layer_drops = [0.5, 1]
    opt_algo = 'gd'
    learning_rate = 0.1
    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }
    num_round = 500
    early_stop_round = 50
    batch_size = 10000
    task.tune(params=params, batch_size=batch_size, num_round=num_round, early_stop_round=early_stop_round,
              verbose=True, save_log=True, save_model=False, dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True,
    #            batch_size=batch_size, save_model=False, save_submission=False)
elif booster == 'embedding':
    model_path = '../model/device_model_multi_layer_perceptron_1.bin'
    # feature_to_predict = 'phone_brand'
    fea_tmp = feature.multi_feature(name=dataset, dtype='f')
    fea_tmp.load()
    data = fea_tmp.get_value()

    num_class = 64
    # pm.predict_with_mlpmodel(model_path, data_to_predict)
    layer_sizes = [task.space, num_class]
    layer_activates = [None]
    layer_inits = [('w0', 'b0')]
    init_path = model_path
    layer_drops = [1]
    opt_algo = 'gd'
    learning_rate = 0.1
    batch_size = 10000
    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }
    task = Task(dataset, booster, version, eval_metric='relu_mse', num_class=num_class)
    task.predict_mlpmodel(data, params=params, batch_size=batch_size )



# ti.init_constant(dataset='concat_6', booster='text_convolutional_neural_network', version=0, random_state=0)
# if __name__ == '__main__':
#     if ti.BOOSTER == 'gblinear':
#         dtrain = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
#         dvalid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
#         dtrain_complete = xgb.DMatrix(ti.PATH_TRAIN)
#         dtest = xgb.DMatrix(ti.PATH_TEST)
#
#         early_stopping_round = 1
#         train_score, valid_score = ti.tune_gblinear(dtrain, dvalid, gblinear_alpha=0, gblinear_lambda=1000,
#                                                     verbose_eval=True, early_stopping_rounds=early_stopping_round)
#         # train_score, valid_score = ti.tune_gblinear(dtrain, dvalid, 0.001, 10, True,
#         #                                             early_stopping_rounds=early_stopping_round, dtest=dtest)
#         print train_score, valid_score
#         # ti.train_gblinear(dtrain_complete, dtest, 0.001, 10, 2)
#
#         # for gblinear_alpha in [0]:
#         #     for gblinear_lambda in [100]:
#         #         for gblinear_lambda_bias in [0]:
#         #             train_score, valid_score = ti.tune_gblinear(dtrain, dvalid, gblinear_alpha, gblinear_lambda,
#         #                                                         gblinear_lambda_bias=gblinear_lambda_bias,
#         #                                                         verbose_eval=True,
#         #                                                         early_stopping_rounds=early_stopping_round)
#         #             # print gblinear_alpha, gblinear_lambda, train_score, valid_score
#         #             # write_log('%f\t%f\t%f\t%f\n' % (gblinear_alpha, gblinear_lambda, train_score, valid_score))
#     elif ti.BOOSTER == 'gbtree':
#         dtrain = xgb.DMatrix(ti.PATH_TRAIN_TRAIN)
#         dvalid = xgb.DMatrix(ti.PATH_TRAIN_VALID)
#         # dtrain_complete = xgb.DMatrix(ti.PATH_TRAIN)
#         dtest = xgb.DMatrix(ti.PATH_TEST)
#
#         max_depth = 3
#         eta = 0.1
#         subsample = 0.7
#         colsample_bytree = 0.7
#         early_stopping_round = 50
#
#         train_score, valid_score = ti.tune_gbtree(dtrain, dvalid, eta=eta, max_depth=max_depth, subsample=subsample,
#                                                   colsample_bytree=colsample_bytree, verbose_eval=True,
#                                                   early_stopping_rounds=early_stopping_round, dtest=dtest)
#         # train_score, valid_score = tune_gbtree(dtrain, dvalid, 0.1, 4, 0.7, 0.7, True, dtest)
#         # print train_score, valid_score
#         # train_gbtree(dtrain_complete, dtest, 0.1, 3, 0.7, 0.7, 300)
#
#         # start_time = time.time()
#         # colsample_bytree = 0.7
#         # for max_depth in [3]:
#         #     for eta in [0.1]:
#         #         for subsample in [0.7]:
#         #             for colsample_bytree in [0.7]:
#         #                 train_score, valid_score = ti.tune_gbtree(dtrain, dvalid, eta, max_depth, subsample,
#         #                                                           colsample_bytree, verbose_eval=True,
#         #                                                           early_stopping_rounds=early_stopping_round)
#         #                 # print max_depth, eta, subsample, train_score, valid_score, time.time() - start_time
#     elif ti.BOOSTER == 'logistic_regression':
#         pass
#     elif ti.BOOSTER == 'factorization_machine':
#         dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
#         dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
#         learning_rate = 0.01
#         # gd, ftrl, adagrad, adadelta
#         opt_algo = 'adagrad'
#         factor_order = 10
#         l1_w = 0
#         l1_v = 0
#         l2_w = 0
#         l2_v = 0
#         l2_b = 0
#         num_round = 200
#         batch_size = 10000
#         early_stopping_round = 10
#         for factor_order in [128, 256]:
#             ti.tune_factorization_machine(dtrain_train, dtrain_valid, factor_order, opt_algo, learning_rate, l1_w=l1_w,
#                                           l1_v=l1_v, l2_w=l2_w, l2_v=l2_v, l2_b=l2_b, num_round=num_round,
#                                           batch_size=batch_size, early_stopping_round=early_stopping_round,
#                                           verbose=True,
#                                           save_log=True)
#     elif ti.BOOSTER == 'multi_layer_perceptron':
#         dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
#         dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
#         # dtrain = ti.read_feature(open(ti.PATH_TRAIN), -1, False)
#         # dtest = ti.read_feature(open(ti.PATH_TEST), -1, False)
#         layer_sizes = [ti.SPACE, 128, ti.NUM_CLASS]
#         layer_activates = ['relu', None]
#         layer_inits = [('normal', 'zero'), ('normal', 'zero')]
#         layer_pool = [None, None, None]
#         drops = [0.5, 1]
#         opt_algo = 'gd'
#         learning_rate = 0.1
#         num_round = 500
#         early_stopping_round = 10
#         batch_size = 10000
#         init_path = None
#
#         # for n in [1000, 800, 600, 400]:
#         # for learning_rate in [0.4, 0.3, 0.2, 0.1]:
#         # for opt_algo in ['gd']:
#         ti.tune_multi_layer_perceptron(train_data=dtrain_train,
#                                        valid_data=dtrain_valid,
#                                        layer_sizes=layer_sizes,
#                                        layer_activates=layer_activates,
#                                        layer_inits=layer_inits,
#                                        opt_algo=opt_algo,
#                                        learning_rate=learning_rate,
#                                        drops=drops,
#                                        num_round=num_round,
#                                        batch_size=batch_size,
#                                        early_stopping_round=early_stopping_round,
#                                        verbose=True,
#                                        save_log=True,
#                                        save_model=True,
#                                        init_path=None,
#                                        layer_pool=layer_pool)
#
#         # ti.train_multi_layer_perceptron(dtrain, dtest, layer_sizes=layer_sizes, layer_activates=layer_activates,
#         #                                 layer_inits=layer_inits, opt_algo=opt_algo, learning_rate=learning_rate,
#         #                                 drops=drops,
#         #                                 num_round=num_round, batch_size=10000)
#     elif ti.BOOSTER == 'multiplex_neural_network':
#         dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
#         dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
#         layer_sizes = [ti.SUB_SPACES, None, ti.NUM_CLASS]
#         layer_activates = ['relu', None]
#         layer_inits = [('normal', 'zero'), ('normal', 'zero')]
#         drops = [0.5, 1]
#         opt_algo = 'gd'
#         learning_rate = 0.1
#         num_round = 500
#         batch_size = 10000
#         for alpha in [1, 2, 4, 8, 16, 32, 64]:
#             # layer_sizes = [ti.SUB_SPACES, None, ti.NUM_CLASS]
#             # layer_sizes[1] = map(lambda x: int(alpha * log(x)), layer_sizes[0])
#             layer_sizes[1] = [32] * len(ti.SUB_SPACES)
#             print layer_sizes
#             ti.tune_multiplex_neural_network(train_data=dtrain_train,
#                                              valid_data=dtrain_valid,
#                                              layer_sizes=layer_sizes,
#                                              layer_activates=layer_activates,
#                                              opt_algo=opt_algo,
#                                              learning_rate=learning_rate,
#                                              drops=drops,
#                                              num_round=num_round,
#                                              batch_size=batch_size,
#                                              early_stopping_round=10,
#                                              verbose=True,
#                                              save_log=True,
#                                              save_model=True,
#                                              layer_inits=layer_inits)
#     elif ti.BOOSTER == 'convolutional_neural_network':
#         dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
#         dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
#         layer_sizes = [ti.SUB_SPACES, [32] * len(ti.SUB_SPACES), ti.NUM_CLASS]
#         kernel_size = [[4, 4, 1, 4]]
#         layer_activates = ['relu', None]
#         layer_inits = [('normal', 'zero'), ('normal', 'zero')]
#         drops = [1, 1]
#         layer_pools = [[1, 2, 2, 1]]
#         opt_algo = 'gd'
#         learning_rate = 0.1
#         num_round = 1000
#         batch_size = 10000
#
#         # cnn_model = convolutional_neural_network(name=ti.TAG,
#         #                                          eval_metric='softmax_log_loss',
#         #                                          layer_sizes=layer_sizes,
#         #                                          layer_activates=layer_activates,
#         #                                          kernel_size=kernel_size,
#         #                                          opt_algo='gd',
#         #                                          learning_rate=0.1,
#         #                                          init_path=None,
#         #                                          layer_inits=layer_inits,
#         #                                          layer_pools=layer_pools)
#         # train_indices, train_values, train_labels = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), 100)
#         # batch_indices, batch_values, batch_shape = ti.libsvm_2_csr(train_indices, train_values, spaces=ti.SUB_SPACES,
#         #                                                            multiplex=True)
#         # # batch_loss, batch_y, batch_y_prob = cnn_model.train(batch_indices, batch_values, batch_shape, train_labels,
#         # #                                                     drops=drops)
#         # # feed_dict = {cnn_model.drops: drops}
#         # # for i in range(len(cnn_model.x)):
#         # #     feed_dict[cnn_model.x[i]] = (batch_indices[i], batch_values[i], batch_shape[i])
#         # # l = cnn_model.run(cnn_model.l, feed_dict)
#         # # print l.shape
#         # l, y, y_prob = cnn_model.train(batch_indices, batch_values, batch_shape, train_labels, drops=drops)
#         # print l
#         # print y.shape
#         # print y_prob.shape
#
#         version = 7
#         for kw in range(2, len(ti.SUB_SPACES) + 1):
#             for kh in [8]:
#                 for kd in [4]:
#                     ti.init_constant(dataset='concat_6', booster='convolutional_neural_network', version=version,
#                                      random_state=0)
#                     version += 1
#                     kernel_size = [[kw, kh, 1, kd]]
#                     ti.tune_convolutional_neural_network(train_data=dtrain_train,
#                                                          valid_data=dtrain_valid,
#                                                          layer_sizes=layer_sizes,
#                                                          layer_activates=layer_activates,
#                                                          opt_algo=opt_algo,
#                                                          learning_rate=learning_rate,
#                                                          drops=drops,
#                                                          num_round=num_round,
#                                                          batch_size=batch_size,
#                                                          early_stopping_round=10,
#                                                          verbose=True,
#                                                          save_log=True,
#                                                          save_model=True,
#                                                          layer_inits=layer_inits,
#                                                          kernel_size=kernel_size,
#                                                          layer_pools=layer_pools)
#     elif ti.BOOSTER == 'text_convolutional_neural_network':
#         layer_sizes = [ti.SUB_SPACES, [32] * len(ti.SUB_SPACES), ti.NUM_CLASS]
#         kernel_depth = 128
#         layer_activates = ['relu', None]
#         layer_inits = [('normal', 'zero'), ('normal', 'zero')]
#         kernel_inits = [('normal', 0.1)] * len(ti.BOOSTER)
#         drops = [0.5, 0.7]
#         opt_algo = 'gd'
#         learning_rate = 0.1
#         num_round = 1000
#         batch_size = 1000
#         # cnn_model = text_convolutional_neural_network(name=ti.TAG,
#         #                                               eval_metric='softmax_log_loss',
#         #                                               layer_sizes=layer_sizes,
#         #                                               layer_activates=layer_activates,
#         #                                               kernel_depth=kernel_depth,
#         #                                               opt_algo='gd',
#         #                                               learning_rate=0.1,
#         #                                               init_path=None,
#         #                                               layer_inits=layer_inits,
#         #                                               kernel_inits=kernel_inits)
#         # train_indices, train_values, train_labels = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), 100)
#         # batch_indices, batch_values, batch_shape = ti.libsvm_2_csr(train_indices, train_values, spaces=ti.SUB_SPACES,
#         #                                                            multiplex=True)
#         # # batch_loss, batch_y, batch_y_prob = cnn_model.train(batch_indices, batch_values, batch_shape, train_labels,
#         # #                                                     drops=drops)
#         # # feed_dict = {cnn_model.drops: drops}
#         # # for i in range(len(cnn_model.x)):
#         # #     feed_dict[cnn_model.x[i]] = (batch_indices[i], batch_values[i], batch_shape[i])
#         # # l = cnn_model.run(cnn_model.y, feed_dict)
#         # # print l.shape
#         # l, y, y_prob = cnn_model.train(batch_indices, batch_values, batch_shape, train_labels, drops=drops)
#         # print l
#         # print y.shape
#         # print y_prob.shape
#         # print y
#         # print y_prob
#
#         dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
#         dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
#         # version = 1
#         # for batch_size in [128, 256, 512, 1024]:
#         # ti.init_constant(dataset='concat_6', booster='convolutional_neural_network', version=version,
#         #                  random_state=0)
#         # version += 1
#         for drops in [[0.5, 0.7], [0.5, 0.5], [0.7, 0.5], [0.7, 0.7]]:
#             ti.tune_convolutional_neural_network_text(train_data=dtrain_train,
#                                                       valid_data=dtrain_valid,
#                                                       layer_sizes=layer_sizes,
#                                                       layer_activates=layer_activates,
#                                                       kernel_depth=kernel_depth,
#                                                       opt_algo=opt_algo,
#                                                       learning_rate=learning_rate,
#                                                       layer_inits=layer_inits,
#                                                       kernel_inits=kernel_inits,
#                                                       drops=drops,
#                                                       verbose=True,
#                                                       save_log=True,
#                                                       save_model=False,
#                                                       num_round=num_round,
#                                                       batch_size=batch_size,
#                                                       early_stopping_round=10,
#                                                       init_path=None)
#     elif ti.BOOSTER == 'average':
#         pass
