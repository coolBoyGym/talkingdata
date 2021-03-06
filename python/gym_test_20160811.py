from task import Task
import feature
import numpy as np

dataset = 'concat_21_new'
booster = 'net2net_mlp'
version = 2

task = Task(dataset, booster, version)

if booster is 'gblinear':
    params = {
        'gblinear_alpha': 0,
        'gblinear_lambda': 100,
        'random_state': 0
    }
    num_round = 200
    early_stop_round = 10
    task.tune(params=params, num_round=num_round, early_stop_round=early_stop_round, save_model=False,
              dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, save_model=False, save_submission=False)
elif booster == 'gbtree':
    params = {
        'eta': 0.1,
        'max_depth': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gbtree_alpha': 0,
        'gbtree_lambda': 10240,
        'random_state': 0
    }
    num_round = 2000
    early_stop_round = 50
    task.tune(params=params, num_round=num_round, early_stop_round=early_stop_round, save_model=False,
              dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, save_model=False, save_submission=False)
# elif booster == 'mlp':
#     layer_sizes = [task.space, 100, 128, task.num_class]
#     layer_activates = ['relu', 'relu', None]
#     layer_inits = [('res:w0', 'res:b0'), ('res:pass', 'zero'), ('res:w1', 'res:b1')]
#     init_path = '../model/concat_7_norm_mlp_2.bin'
#     layer_drops = [0.5, 0.75, 1]
#     opt_algo = 'adam'
#     learning_rate = 0.00001
#     params = {
#         'layer_sizes': layer_sizes,
#         'layer_activates': layer_activates,
#         'layer_drops': layer_drops,
#         'layer_inits': layer_inits,
#         'init_path': init_path,
#         'opt_algo': opt_algo,
#         'learning_rate': learning_rate,
#     }
#     num_round = 1000
#     early_stop_round = 5
#     batch_size = 10000
    # for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
    # for batch_size in [1000]:
    # task.tune(params=params, batch_size=batch_size, num_round=num_round, early_stop_round=early_stop_round,
    #           verbose=True, save_log=True, save_model=True, dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=False,
    #            save_submission=False)
# elif booster == 'mlp':
#     layer_sizes = [task.space, 100, task.num_class]
#     layer_activates = ['relu', None]
#     layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
#     layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    # init_path = '../model/ensemble_5_mlp_3.bin'
    # init_path = None
    # layer_drops = [0.5, 0.9]
    # opt_algo = 'adam'
    # learning_rate = 0.00001
    # opt_algo = 'gd'
    # learning_rate = 0.15
    # params = {
    #     'layer_sizes': layer_sizes,
    #     'layer_activates': layer_activates,
    #     'layer_drops': layer_drops,
    #     'layer_inits': layer_inits,
    #     'init_path': init_path,
    #     'opt_algo': opt_algo,
    #     'learning_rate': learning_rate,
    # }
    # num_round = 2000
    # early_stop_round = 5
    # batch_size = -1
    # for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
    # for batch_size in [1000]:
    # task.tune(params=params, batch_size=batch_size, num_round=num_round, early_stop_round=early_stop_round,
    #           verbose=True, save_log=True, save_model=True, dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=False,
    #            save_submission=False)

elif booster == 'mlp':
    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    dtest = task.load_data(task.path_test)

    layer_sizes = [task.space, 156, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('net2:w0', 'net2:b0'), ('net2:w1', 'net2:b1')]
    init_path = '../model/concat_7_norm_mlp_201.bin'
    # layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    # init_path = None
    layer_drops = [0.5, 1]
    layer_l2 = [0.0001, 0.0001]
    opt_algo = 'adam'
    learning_rate = 1e-5
    # opt_algo = 'gd'
    # learning_rate = 0.1
    random_seed = 0x89AB

    batch_size = -1
    num_round = 1500
    early_stop_round = 20

    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_l2': layer_l2,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
        'random_seed': random_seed,
    }

    # 0x0123, 0x4567, 0x89AB, 0xCDEF, 0xFEDC, 0xBA98, 0x7654, 0x3210
    task.tune(dtrain=dtrain, dvalid=dvalid, params=params, batch_size=batch_size, num_round=num_round,
              early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=dtest,
              save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=False,
    #            save_submission=True)


elif booster == 'mnn':
    layer_sizes = [task.sub_spaces, map(int, np.log(task.sub_spaces) * 5), task.num_class]
    # print 'task.sub_space =', task.sub_spaces
    layer_activates = ['relu', None]
    layer_inits = [('tnormal', 'zero'), ('tnormal', 'zero')]
    init_path = None
    layer_drops = [0.8, 1]
    opt_algo = 'adam'
    learning_rate = 0.0001
    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_l2': [0.1, 0.3],
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }
    batch_size = 1000
    num_round = 43
    early_stop_round = 10
    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    dtest = task.load_data(task.path_test)
    for batch_size in [1000]:
        # for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
        # for batch_size in [128, 256, 512]:
        # for opt_algo in ['gd', 'adagrad', 'adadelta', 'ftrl', 'adam']:
        # task.tune(dtrain=dtrain, dvalid=dvalid, params=params, batch_size=batch_size, num_round=num_round,
        #           early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=None,
        #           save_feature=False)
        task.train(params=params, num_round=num_round, verbose=True,
                   batch_size=batch_size, save_model=False, save_submission=True)

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
    task.predict_mlpmodel(data, params=params, batch_size=batch_size)

elif booster == 'net2net_mlp':
    params_1 = {
        'layer_sizes': [task.space, 64, task.num_class],
        'layer_activates': ['relu', None],
        'layer_drops': [0.5, 1],
        'layer_l2': [0.0001, 0.0001],
        'layer_inits': [('res:w0', 'res:b0'), ('res:w1', 'res:b1')],
        'init_path': '../model/concat_1_mlp_100.bin',
        'opt_algo': 'gd',
        'learning_rate': 0.1,
    }

    params_2 = {
        'layer_sizes': [task.space, 128, task.num_class],
        'layer_activates': ['relu', None],
        'layer_drops': [0.5, 1],
        'layer_l2': [0.0001, 0.0001],
        'layer_inits': [('net2:w0', 'net2:b0'), ('net2:w1', 'net2:b1')],
        'init_path': '../model/concat_6_mlp_100.bin',
        'opt_algo': 'adam',
        'learning_rate': 1e-5,
        'random_seed': 0x0123
    }
    batch_size = 1000

    num_round = 2900
    early_stop_round = 20

    for sss in [100, 116, 138]:
        params_2['layer_sizes'] = [task.space, sss, task.num_class]
        task.net2net_mlp(params_1=params_1, params_2=params_2, batch_size=batch_size, num_round=num_round,
                         early_stop_round=early_stop_round,
                         verbose=True, save_log=True, save_model=True, split_cols=2)
        task.upgrade_version()
    # task.net2net_mlp(params_1=params_1, params_2=params_2, batch_size=batch_size, num_round=num_round,
    #                  early_stop_round=early_stop_round,
    #                  verbose=True, save_log=True, save_model=True, split_cols=2)
    # task.net2net_mlp_train(params_1=params_1, params_2=params_2, batch_size=batch_size, num_round=num_round,
    #                        verbose=True, save_model=True, split_cols=2, save_submission=True)
