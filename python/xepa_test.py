import numpy as np

from task import Task

dataset = 'concat_23_freq'
booster = 'net2net_mnn'
version = 0

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
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gbtree_alpha': 0,
        'gbtree_lambda': 1024,
        'random_state': 0
    }
    num_round = 2000
    early_stop_round = 50
    task.tune(params=params, num_round=num_round, early_stop_round=early_stop_round, save_model=False,
              dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, save_model=False, save_submission=False)
elif booster == 'mlp':
    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    dtest = task.load_data(task.path_test)

    params = {
        'layer_sizes': [task.space, 64, task.num_class],
        'layer_activates': ['relu', None],
        'layer_drops': [0.5, 1],
        'layer_l2': [0.001, 0.001],
        'layer_inits': [('net2:w0', 'net2:b0'), ('net2:w1', 'net2:b1')],
        'init_path': '../model/concat_1_mlp_100.bin',
        'opt_algo': 'gd',
        'learning_rate': 0,
        'random_seed': None,
    }
    batch_size = -1
    num_round = 1
    early_stop_round = 50

    # params['learning_rate'] = learning_rate
    # for l2 in [0, 0.00001, 0.0001, 0.001]:
    # for seed in [0x0000, 0x0123, 0x4567, 0x89AB, 0xCDEF, 0xFFFF]:
    #     params['random_seed'] = seed
    #     print seed
    # print l2
    # params['layer_l2'] = [l2, l2]
    task.tune(dtrain, dvalid, params=params, batch_size=batch_size, num_round=num_round,
              early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=dtest,
              save_feature=True)
    task.upgrade_version()
    # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_log=True,
    #            save_model=True, save_submission=True)
    # task.kfold(n_fold=5, n_repeat=10, params=params, batch_size=batch_size, num_round=num_round,
    #            early_stop_round=early_stop_round, verbose=True)
elif booster == 'mnn':
    params = {
        'layer_sizes': [task.sub_spaces, map(int, np.log(task.sub_spaces) * 4), task.num_class],
        'layer_activates': ['relu', None],
        'layer_drops': [0.5, 1],
        'layer_l2': [0.01, 0.01],
        'layer_inits': [('normal', 'zero'), ('normal', 'zero')],
        'init_path': None,
        'opt_algo': 'adam',
        'learning_rate': 1e-3,
    }
    batch_size = -1
    num_round = 450
    early_stop_round = 10
    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    # dtest = task.load_data(task.path_test)
    print params
    task.tune(dtrain=dtrain, dvalid=dvalid, params=params, batch_size=batch_size, num_round=num_round,
              early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=None,
              save_feature=False)
    task.upgrade_version()
    # task.train(params=params, num_round=num_round, verbose=True,
    #            batch_size=batch_size, save_model=False, save_submission=False)
elif booster == 'net2net_mlp':
    params = {
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
    num_round = 2
    early_stop_round = 10

    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    dtest = task.load_data(task.path_test)

    task.net2net_tune('concat_1_mlp_100', 2, dtrain, dvalid, params, batch_size=batch_size, num_round=num_round,
                      early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=dtest,
                      save_feature=True)
    task.net2net_train('../output/concat_1_mlp_1001.submission', 2, params=params, batch_size=batch_size,
                       num_round=num_round, verbose=True, save_log=True, save_model=True, save_submission=True)
elif booster == 'net2net_mnn':
    params = {
        'layer_sizes': [task.sub_spaces, map(lambda x: min(x, 256), task.sub_spaces), task.num_class],
        'layer_activates': ['relu', None],
        'layer_drops': [0.5, 1],
        'layer_l2': [0.001, 0.001],
        'layer_inits': [('normal', 'zero'), ('normal', 'zero')],
        'init_path': None,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'random_seed': None
    }
    batch_size = 1000
    num_round = 2000
    early_stop_round = 10
    fea_name = 'concat_1_ensemble_mlp_1024'
    sub_file = '../output/concat_1_ensemble_mlp_1024.submission'

    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    dtest = task.load_data(task.path_test)

    task.net2net_tune(fea_name, 2, dtrain, dvalid, params, batch_size=batch_size, num_round=num_round,
                      early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=dtest,
                      save_feature=True)
