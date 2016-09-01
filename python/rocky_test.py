from task import Task
import feature

dataset = 'concat_6'
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
    layer_sizes = [task.space, 64, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
    # layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    init_path = '../model/concat_1_ensemble_mlp_1.bin'
    # init_path = None
    layer_drops = [0.5, 1]
    layer_l2 = [0.0001, 0.0001]
    opt_algo = 'gd'
    learning_rate = 0
    batch_size = -1
    num_round = 1
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

    task.tune(params=params, batch_size=batch_size, num_round=num_round, early_stop_round=early_stop_round,
              verbose=True, save_log=True, save_model=True, dtest=None, save_feature=True)
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
    layer_sizes = [task.space, 64, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
    init_path = '../model/concat_1_mlp_100.bin'
    layer_drops = [0.5, 1]
    layer_l2 = [0.0001, 0.0001]
    opt_algo = 'gd'
    learning_rate = 0.2
    params_1 = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_l2': layer_l2,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
    }

    layer_sizes = [task.space, 128, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('net2:w0', 'net2:b0'), ('res:w1', 'res:b1')]
    init_path = '../model/concat_6_mlp_100.bin'
    layer_drops = [0.5, 1]
    layer_l2 = [0.0001, 0.0001]
    opt_algo = 'adam'
    learning_rate = 0.00001
    batch_size = -1
    num_round = 3000
    early_stop_round = 20
    params_2 = {
        'layer_sizes': layer_sizes,
        'layer_activates': layer_activates,
        'layer_drops': layer_drops,
        'layer_l2': layer_l2,
        'layer_inits': layer_inits,
        'init_path': init_path,
        'opt_algo': opt_algo,
        'learning_rate': learning_rate,
        # 'random_seed': 0xFFFF
    }
    # task.net2net_mlp(params_1=params_1, params_2=params_2, batch_size=batch_size, num_round=num_round,
    #                  early_stop_round=early_stop_round,
    #                  verbose=True, save_log=True, save_model=True, split_cols=2)
    # task.net2net_mlp_train(params_1=params_1, params_2=params_2, batch_size=batch_size, num_round=num_round, verbose=True,
    #                        save_model=True, split_cols=2, save_submission=True)
    task.fea_net_mlp(fea_name='concat_1_ensemble_mlp_1', params_2=params_2, batch_size=batch_size, num_round=num_round,
                                       early_stop_round=early_stop_round,
                                       verbose=True, save_log=True, save_model=True, split_cols=2)

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
    model_path = '../model/device_model_multi_layer_perceptron_1.bin'
    # feature_to_predict = 'phone_brand'
    fea_tmp = feature.MultiFeature(name=dataset, dtype='f')
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
