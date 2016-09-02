from task import Task

dataset = 'concat_25'
booster = 'net2net_mlp'
version = 1

task = Task(dataset, booster, version)

if booster == 'net2net_mlp':
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

    # for sss in [100, 116, 138]:
    #     params_2['layer_sizes'] = [task.space, sss, task.num_class]
    #     task.net2net_mlp(params_1=params_1, params_2=params_2, batch_size=batch_size, num_round=num_round,
    #                      early_stop_round=early_stop_round,
    #                      verbose=True, save_log=True, save_model=True, split_cols=2)
    #     task.upgrade_version()
    task.net2net_mlp(params_1=params_1, params_2=params_2, batch_size=batch_size, num_round=num_round,
                     early_stop_round=early_stop_round,
                     verbose=True, save_log=True, save_model=True, split_cols=2)
    # task.net2net_mlp_train(params_1=params_1, params_2=params_2, batch_size=batch_size, num_round=num_round,
    #                        verbose=True, save_model=True, split_cols=2, save_submission=True)
