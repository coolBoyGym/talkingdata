from task import Task
import feature

dataset = 'concat_15'
booster = 'mlp'
version = 2

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
# elif booster == 'mlp':
#     layer_sizes = [task.space, 100, 128, task.num_class]
#     layer_activates = ['relu', 'relu', None]
#     layer_inits = [('res:w0', 'res:b0'), ('res:pass', 'zero'), ('res:w1', 'res:b1')]
#     init_path = '../model/concat_7_norm_mlp_2.bin'
#     layer_drops = [0.5, 0.75, 1]
#     opt_algo = 'adam'
#     learning_rate = 0.00001
#     params = {
#         'layer_sizes': layer_sizes,
#         'layer_activates': layer_activates,
#         'layer_drops': layer_drops,
#         'layer_inits': layer_inits,
#         'init_path': init_path,
#         'opt_algo': opt_algo,
#         'learning_rate': learning_rate,
#     }
#     num_round = 1000
#     early_stop_round = 5
#     batch_size = 10000
    # for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
    # for batch_size in [1000]:
    # task.tune(params=params, batch_size=batch_size, num_round=num_round, early_stop_round=early_stop_round,
    #           verbose=True, save_log=True, save_model=True, dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=False,
    #            save_submission=False)
elif booster == 'mlp':
    layer_sizes = [task.space, 100, task.num_class]
    layer_activates = ['relu', None]
    layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
    # layer_inits = [('normal', 'zero'), ('normal', 'zero')]
    init_path = '../model/concat_15_mlp_1.bin'
    # init_path = None
    layer_drops = [0.5, 1]
    opt_algo = 'adam'
    learning_rate = 0.001
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
    early_stop_round = 5
    batch_size = -1
    # for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
    # for batch_size in [1000]:
    task.tune(params=params, batch_size=batch_size, num_round=num_round, early_stop_round=early_stop_round,
              verbose=True, save_log=True, save_model=True, dtest=None, save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True, batch_size=batch_size, save_model=False,
    #            save_submission=False)

elif booster == 'mnn':
    layer_sizes = [task.sub_spaces, [64, 64, 256, 64], task.num_class]
    # print 'task.sub_space =', task.sub_spaces
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
    batch_size = 1024
    num_round = 1000
    early_stop_round = 50
    dtrain = task.load_data(task.path_train_train)
    dvalid = task.load_data(task.path_train_valid)
    for batch_size in [1024]:
        # for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
        # for batch_size in [128, 256, 512]:
        # for opt_algo in ['gd', 'adagrad', 'adadelta', 'ftrl', 'adam']:
        task.tune(dtrain=dtrain, dvalid=dvalid, params=params, batch_size=batch_size, num_round=num_round,
                  early_stop_round=early_stop_round, verbose=True, save_log=True, save_model=True, dtest=None,
                  save_feature=False)
    # task.train(params=params, num_round=num_round, verbose=True,
    #            batch_size=batch_size, save_model=False, save_submission=False)

elif booster == 'embedding':
    model_path = '../model/device_model_multi_layer_perceptron_1.bin'
    # feature_to_predict = 'phone_brand'
    fea_tmp = feature.multi_feature(name=dataset, dtype='f')
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