from task import Task

dataset = 'concat_100'
# dataset = 'concat_100'
booster = 'net2net_mlp'
version = 1009

task = Task(dataset, booster, version)

if booster == 'net2net_mlp':
    params = {
        'layer_sizes': [task.space, 148, task.num_class],   # best
        'layer_activates': ['relu', None],
        'layer_drops': [0.5, 1],      # best
        'layer_l2': [0.0001, 0.0001],  # best
        'layer_inits': [('net2:w0', 'net2:b0'), ('net2:w1', 'net2:b1')],
        'init_path': '../model/concat_6_mlp_100.bin',
        # 'init_path': '../model/concat_100_net2net_mlp_52.bin',
        'opt_algo': 'adam',
        'learning_rate': 1e-4,
        # 'opt_algo': 'gd',
        # 'learning_rate': 0,
        'random_seed': 0x0123  # best
    }

    batch_size = 1000  # best
    num_round = 2500
    early_stop_round = 10

    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    dtest = task.load_data(task.path_test)

    # for sss in [1024, ]:
    #     batch_size = sss
    #     for lll in [1e-4]:
    #         params['learning_rate'] = lll
    #         task.net2net_tune('concat_1_ensemble_mlp_1024', 2, dtrain, dvalid, params, batch_size=batch_size, num_round=num_round,
    #                           early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=False, dtest=dtest,
    #                           save_feature=False)
    #
    #         task.upgrade_version()
    # task.net2net_tune('concat_1_ensemble_mlp_1024', 2, dtrain, dvalid, params, batch_size=batch_size,
    #                   num_round=num_round, early_stop_round=early_stop_round, verbose=True,
    #                   save_log=True, save_model=False, dtest=dtest, save_feature=False)
    for s in [24, 26, 27, 28, 30]:
        num_round = s
        task.net2net_train('../output/concat_1_ensemble_mlp_1024.submission', 2, params=params, batch_size=batch_size,
                           num_round=num_round, verbose=True, save_log=True, save_model=True, save_submission=True)
        task.upgrade_version()
    # task.net2net_predict(split_col=2, params=params, batch_size=10000)

