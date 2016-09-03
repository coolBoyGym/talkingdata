from task import Task

dataset = 'installed_app'
booster = 'net2net_mlp'
version = 1

task = Task(dataset, booster, version)
if booster is 'gblinear':
    params = {
        'gblinear_alpha': 0,
        'gblinear_lambda': 7.5,
        'random_state': 0
    }
    num_round = 200
    early_stop_round = 10
    task.tune(params=params, num_round=num_round, early_stop_round=early_stop_round, save_log=True, save_model=False,
              dtest=None, save_feature=True)
    # task.train(params=params, num_round=num_round, verbose=True, save_model=False, save_submission=False)
elif booster == 'gbtree':
    params = {
        'eta': 0.07,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'gbtree_alpha': 0,
        'gbtree_lambda': 0,
        'random_state': 0
    }
    num_round = 1000
    early_stop_round = 50
    task.tune(params=params, num_round=num_round, early_stop_round=early_stop_round, save_log=True, save_model=False,
              dtest=None, save_feature=True)
    # task.train(params=params, num_round=num_round, verbose=True, save_model=False, save_submission=False)
elif booster == 'mlp':
    layer_sizes = [task.space, 32, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('net2:w0', 'net2:b0'), ('net2:w1', 'net2:b1')]
    # layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    init_path = '../model/installed_app_mlp_1.bin'
    # init_path = None
    layer_drops = [0.5, 1]
    layer_l2 = [0.0001, 0.0001]
    opt_algo = 'adam'
    learning_rate = 1e-5
    batch_size = -1
    num_round = 5000
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
        'random_seed': 0
    }
    #
    # layer_sizes = [task.space, 64, 128, task.num_class]
    # layer_activates = ['relu', 'relu', None]
    # layer_inits = [('res:w0', 'res:b0'), ('res:pass', 'zero'), ('res:w1', 'res:b1')]
    # init_path = '../model/ensemble_6_mlp_1.bin'
    # layer_drops = [0.5, 0.75, 1]
    # opt_algo = 'adam'
    # learning_rate = 0.0001
    # params = {
    #     'layer_sizes': layer_sizes,
    #     'layer_activates': layer_activates,
    #     'layer_drops': layer_drops,
    #     'layer_inits': layer_inits,
    #     'init_path': init_path,
    #     'opt_algo': opt_algo,
    #     'learning_rate': learning_rate,
    # }

    # for sd in [0x4444, 0x5555, 0x6666, 0x7777]:
    #     params['random_seed'] = sd
    #     print sd
    # for ls in [80, 90, 100]:
    #     params['layer_sizes'][1] = ls
    task.tune(params=params, batch_size=batch_size, num_round=num_round, early_stop_round=early_stop_round,
              verbose=True, save_log=True, save_model=True, dtest=None, save_feature=True)
    # task.upgrade_version()

    # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=True,
    #            save_submission=True)
elif booster == 'predict':
    layer_sizes = [task.space, 64, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
    init_path = '../model/concat_6_net2net_mlp_11.bin'
    layer_drops = [0.5, 1]
    layer_l2 = [0, 0]
    opt_algo = 'gd'
    learning_rate = 0.1
    batch_size = 10000
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
    task.predict(params=params, batch_size=batch_size, save_feature=True)
elif booster == 'net2net_mlp':
    params = {
        'layer_sizes': [task.space, 100, task.num_class],
        'layer_activates': ['relu', None],
        'layer_drops': [0.5, 1],
        'layer_l2': [0.0001, 0.0001],
        # 'layer_inits': [('net2:w0', 'net2:b0'), ('net2:w1', 'net2:b1')],
        'layer_inits': [('normal', 'zero'), ('normal', 'zero'), ('normal', 'zero')],
        'init_path': None,
        # 'init_path': '../model/concat_6_embedding_net2net_mlp_1.bin',
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'random_seed': None
    }

    batch_size = 1000
    num_round = 2000
    early_stop_round = 10

    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    dtest = task.load_data(task.path_test)

    task.net2net_tune('concat_1_ensemble_mlp_1024', 0, dtrain, dvalid, params, batch_size=batch_size,
                      num_round=num_round,
                      early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=dtest,
                      save_feature=True)
    # task.net2net_train('../output/concat_1_mlp_1001.submission', 2, params=params, batch_size=batch_size,
    #                    num_round=num_round, verbose=True, save_log=True, save_model=True, save_submission=True)

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
    model_path = '../model/installed_app_label_net2net_mlp_1.bin'
    # feature_to_predict = 'phone_brand'
    # pm.predict_with_mlpmodel(model_path, data_to_predict)
    embedding_size = 32
    params = {
        'layer_sizes': [task.space, embedding_size],
        'layer_activates': [None],
        'layer_drops': [1],
        'layer_inits': [('w0', 'b0')],
        'init_path': model_path,
        'opt_algo': 'gd',
        'learning_rate': 0,
    }
    batch_size = 10000
    task = Task(dataset, booster, version, input_type='sparse', eval_metric='relu_mse', num_class=embedding_size)
    # task.predict_mlpmodel(data, params=params, batch_size=batch_size)
    task.net2net_predict(params=params, batch_size=batch_size)
