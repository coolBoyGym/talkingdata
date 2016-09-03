from task import Task

dataset = 'concat_25_diff'
booster = 'net2net_mlp'
version = 1

task = Task(dataset, booster, version)

if booster == 'net2net_mlp':
    params = {
        'layer_sizes': [task.space, 128, task.num_class],
        'layer_activates': ['relu', None],
        'layer_drops': [0.5, 1],
        'layer_l2': [0.0001, 0.0001],
        'layer_inits': [('net2:w0', 'net2:b0'), ('net2:w1', 'net2:b1')],
        'init_path': '../model/concat_25_diff_net2net_mlp_1.bin',
        # 'init_path': '../model/concat_6_mlp_100.bin',
        'opt_algo': 'gd',
        # 'opt_algo': 'adam',
        # 'learning_rate': 1e-5,
        'learning_rate': 0,
        'random_seed': 0x0123
    }

    batch_size = 1000
    num_round = 1
    early_stop_round = 10

    # dtrain = task.load_data(task.path_train_train)
    # dvalid = task.load_data(task.path_train_valid)
    # dtest = task.load_data(task.path_test)

    # task.net2net_mlp(dtrain=dtrain, dvalid=dvalid, params_1=params_1, params_2=params_2, batch_size=batch_size,
    #                  num_round=num_round, early_stop_round=early_stop_round, verbose=True, save_log=True,
    #                  save_model=True, split_cols=2, dtest=dtest, save_feature=True)
    # task.upgrade_version()
    # task.net2net_mlp_train(params_1=params_1, params_2=params_2, batch_size=batch_size, num_round=num_round,
    #                        verbose=True, save_model=True, split_cols=2, save_submission=True)

    # for sss in [138]:
    #     params['layer_sizes'] = [task.space, sss, task.num_class]
    #     for lll in [0x0123]:
    #         params['random_seed'] = lll

    # task.net2net_tune('concat_1_ensemble_mlp_1024', 2, dtrain, dvalid, params, batch_size=batch_size, num_round=num_round,
    #                   early_stop_round=early_stop_round, verbose=True, save_log=False, save_model=False, dtest=dtest,
    #                   save_feature=True)

            # task.upgrade_version()
    # task.net2net_train('../output/concat_1_ensemble_mlp_1024.submission', 2, params=params, batch_size=batch_size,
    #                    num_round=num_round, verbose=True, save_log=True, save_model=True, save_submission=True)
    task.net2net_predict(split_col=2, params=params, batch_size=10000)