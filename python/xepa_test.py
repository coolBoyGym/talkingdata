from task import Task

dataset = 'concat_6'
booster = 'mlp'
version = 110

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

    layer_sizes = [task.space, 100, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    # init_path = '../model/concat_6_mlp_100.bin'
    init_path = None
    layer_drops = [0.5, 1]
    opt_algo = 'adam'
    learning_rate = 1e-4
    batch_size = -1
    num_round = 2000
    early_stop_round = 10

    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }
    # version = 200
    for learning_rate in [1e-4, 1e-5, 1e-6]:
        params['learning_rate'] = learning_rate
        task.upgrade_version()
        print params
        task.tune(dtrain=dtrain, dvalid=dvalid, params=params, batch_size=batch_size, num_round=num_round,
                  early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=dtest,
                  save_feature=True)
        # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=False,
        #            save_submission=True)
elif booster == 'mnn':
    layer_sizes = [task.sub_spaces, [64] * len(task.sub_spaces), task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    init_path = None
    layer_drops = [0.5, 0.5]
    opt_algo = 'adagrad'
    learning_rate = 0.01
    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }
    batch_size = 10000
    num_round = 1000
    early_stop_round = 50
    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    # for batch_size in [1024, 2048, 4096, 8192]:
    for opt_algo in ['gd', 'adagrad', 'adadelta', 'ftrl', 'adam']:
        task.tune(dtrain=dtrain, dvalid=dvalid, params=params, batch_size=batch_size, num_round=num_round,
                  early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=False, dtest=None,
                  save_feature=False)
        # task.train(params=params, num_round=num_round, verbose=True,
        #            batch_size=batch_size, save_model=False, save_submission=False)
