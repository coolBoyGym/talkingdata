import numpy as np

from task import Task

dataset = 'concat_6'
booster = 'mlp'
version = 141

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
    # dtrain = task.load_data(task.path_train_train)
    # dvalid = task.load_data(task.path_train_valid)
    # dtest = task.load_data(task.path_test)

    layer_sizes = [task.space, 100, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
    init_path = '../model/concat_6_mlp_100.bin'
    layer_drops = [0.5, 1]
    layer_l2 = [0, 0]
    opt_algo = 'adam'
    learning_rate = 0.0001
    batch_size = 1024
    num_round = 470
    early_stop_round = 10

    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_l2': layer_l2,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }

    for learning_rate in [1e-5]:
        for batch_size in [4000]:
            params['learning_rate'] = learning_rate
            print params
            # task.tune(dtrain=dtrain, dvalid=dvalid, params=params, batch_size=batch_size, num_round=num_round,
            #           early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=dtest,
            #           save_feature=True)
            # task.upgrade_version()
            task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=False,
                       save_submission=True)
elif booster == 'mnn':
    layer_sizes = [task.sub_spaces, None, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    init_path = None
    layer_l2 = [0, 0]
    layer_drops = [0.5, 1]
    opt_algo = 'gd'
    learning_rate = 0.1
    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_l2': layer_l2,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }
    batch_size = -1
    num_round = 1000
    early_stop_round = 50
    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    # for batch_size in [1024, 2048, 4096, 8192]:
    # for opt_algo in ['gd', 'adagrad', 'adadelta', 'ftrl', 'adam']:
    for alpha in [16, 32, 64]:
        params['layer_sizes'][1] = map(int, np.log(task.sub_spaces) * alpha)
        print params['layer_sizes']
        task.tune(dtrain=dtrain, dvalid=dvalid, params=params, batch_size=batch_size, num_round=num_round,
                  early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=None,
                  save_feature=False)
        # task.train(params=params, num_round=num_round, verbose=True,
        #            batch_size=batch_size, save_model=False, save_submission=False)
