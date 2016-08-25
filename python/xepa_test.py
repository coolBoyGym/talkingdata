from task import Task
import numpy as np

dataset = 'ensemble_7'
booster = 'mnn'
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
    batch_size = -1
    num_round = 1000
    early_stop_round = 10

    params = {
        'layer_sizes': [task.space, 128, 64, task.num_class],
        'layer_activates': ['relu', 'relu', None],
        'layer_drops': [0.5, 1, 1],
        'layer_l2': [0, 0, 0],
        'layer_inits': [('res:w0', 'res:b0'), ('res:w1', 'res:b1'), ('res:pass', 'zero')],
        'init_path': '../model/concat_6_mlp_100.bin',
        'opt_algo': 'gd',
        'learning_rate': 1e-2,
    }

    # params['learning_rate'] = learning_rate
    print params
    task.tune(dtrain=dtrain, dvalid=dvalid, params=params, batch_size=batch_size, num_round=num_round,
              early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=dtest,
              save_feature=True)
    # task.upgrade_version()
    # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=False,
    #            save_submission=True)
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
    num_round = 1000
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
elif booster == 'tcnn':
    params = {
        'layer_sizes': [task.sub_spaces, 100, task.num_class],
        'layer_activates': ['relu', None],
        'layer_inits': [('tnormal', 'zero'), ('tnormal', 'zero')],
        'kernel_depth': 100,
        'kernel_inits': ('tnormal', 'zero'),
        'init_path': None,
        'layer_l2': [0.001, 0.001],
        'kernel_l2': 0.001,
        'layer_drops': [0.5, 1],
        'opt_algo': 'adam',
        'learning_rate': 1e-3
    }
    batch_size = 1024
    num_round = 1000
    early_stop_round = 10
    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    print params
    task.tune(dtrain=dtrain, dvalid=dvalid, params=params, batch_size=batch_size, num_round=num_round,
              early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=None,
              save_feature=False)
