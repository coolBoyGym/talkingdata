from task import Task

dataset = 'concat_22_128'
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
    layer_sizes = [task.space, 128, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    init_path = None
    layer_drops = [0.5, 1]
    opt_algo = 'gd'
    learning_rate = 0.5
    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }
    num_round = 1000
    early_stop_round = 50
    for batch_size in [1024, 2048, 4096, 8192, 16384]:
        task.tune(params=params, batch_size=batch_size, num_round=num_round, early_stop_round=early_stop_round,
                  verbose=True, save_log=True, save_model=False, dtest=None, save_feature=False)
        # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=False,
        #            save_submission=False)
elif booster == 'mnn':
    layer_sizes = [task.sub_spaces, [32] * len(task.sub_spaces), task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    init_path = None
    layer_drops = [0.5, 1]
    opt_algo = 'gd'
    learning_rate = 0.5
    params = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }
    num_round = 1000
    early_stop_round = 50
    for batch_size in [1024, 2048, 4096, 8192, 16384]:
        task.tune(params=params, batch_size=batch_size, num_round=num_round, early_stop_round=early_stop_round,
                  verbose=True, save_log=True, save_model=False, dtest=None, save_feature=False)
        # task.train(params=params, num_round=num_round, verbose=True,
        #            batch_size=batch_size, save_model=False, save_submission=False)
