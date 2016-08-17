import train_impl as ti
import numpy as np
from scipy.sparse import csr_matrix
import xgboost as xgb
import feature_factory as ff
from model_impl import opt_property

def random_sample(train_data, number):
    train_indices, train_values, train_labels = train_data
    random_indices = np.random.randint(0, len(train_indices)-1, number)
    sample_indices = train_indices[random_indices]
    sample_values = train_values[random_indices]
    sample_labels = train_labels[random_indices]
    return sample_indices, sample_values, sample_labels

def ensemble_model(train_data, valid_data, test_data, name, model_list):
    train_indices, train_values, train_labels = train_data
    valid_indices, valid_values, valid_labels = valid_data
    test_indices, test_values, test_labels = test_data

    valid_csr_indices, valid_csr_values, valid_csr_shape = ti.libsvm_2_csr(valid_indices, valid_values)
    valid_csr = csr_matrix((valid_csr_values, (valid_csr_indices[:, 0], valid_csr_indices[:, 1])), shape=valid_csr_shape)
    valid_labels = ti.label_2_group_id(valid_labels)
    d_valid = xgb.DMatrix(valid_csr, label=valid_labels)

    test_csr_indices, test_csr_values, test_csr_shape = ti.libsvm_2_csr(test_indices, test_values)
    test_csr = csr_matrix((test_csr_values, (test_csr_indices[:, 0], test_csr_indices[:, 1])), shape=test_csr_shape)
    test_labels = ti.label_2_group_id(test_labels)
    d_test = xgb.DMatrix(test_csr, label=test_labels)

    features_for_ensemble = []
    for model in model_list:
        for i in range(model_list[model]):
            if model == 'gblinear':
                ti.BOOSTER = 'gblinear'
                sample_data = random_sample(train_data, len(train_indices))
                sample_indices, sample_values, sample_labels = sample_data
                train_csr_indices, train_csr_values, train_csr_shape = ti.libsvm_2_csr(sample_indices, sample_values)
                train_csr = csr_matrix((train_csr_values, (train_csr_indices[:, 0], train_csr_indices[:, 1])),
                                       shape=train_csr_shape)
                sample_labels = ti.label_2_group_id(sample_labels, num_class=12)
                d_train = xgb.DMatrix(train_csr, label=sample_labels)
                feature_model = ti.ensemble_gblinear(d_train, d_valid, gblinear_alpha=0, gblinear_lambda=10,
                                                     verbose_eval=True, early_stopping_rounds=8, dtest=d_test)
                features_for_ensemble.append(feature_model)
            elif model == 'gbtree':
                ti.BOOSTER = 'gbtree'
                sample_data = random_sample(train_data, len(train_indices))
                sample_indices, sample_values, sample_labels = sample_data
                train_csr_indices, train_csr_values, train_csr_shape = ti.libsvm_2_csr(sample_indices, sample_values)
                train_csr = csr_matrix((train_csr_values, (train_csr_indices[:, 0], train_csr_indices[:, 1])),
                                       shape=train_csr_shape)
                sample_labels = ti.label_2_group_id(sample_labels, num_class=12)
                d_train = xgb.DMatrix(train_csr, label=sample_labels)
                feature_model = ti.ensemble_gbtree(d_train, d_valid, 0.1, 3, 0.8, 0.6, verbose_eval=True,
                                                   early_stopping_rounds=20, dtest=d_test)
                features_for_ensemble.append(feature_model)
            elif model == 'mlp':
                ti.BOOSTER = 'multi_layer_perceptron'
                layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
                layer_activates = ['relu', 'relu', None]
                drops = [0.5, 0.5]
                learning_rate = 0.2
                num_round = 1000
                opt_prop = opt_property('gd', learning_rate)
                sample_data = random_sample(train_data, len(train_indices))

                ti.ensemble_multi_layer_perceptron(sample_data, valid_data, layer_sizes, layer_activates, opt_prop,
                                               drops, num_round=num_round, batch_size=10000, early_stopping_round=10,
                                               verbose=True, save_log=False, dtest=None)



    feature_ensembled = ff.ensemble_concat_feature(name, features_for_ensemble)
    return feature_ensembled


if __name__ == '__main__':
    ti.init_constant('concat_6', booster=None, version=1, random_state=0)
    dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1, False)
    dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1, False)
    dtest = ti.read_feature(open(ti.PATH_TEST), -1, False)

    feature_ensembled = ensemble_model(dtrain_train, dtrain_valid, dtest, 'ensmeble_test', {'gblinear':2, 'gbtree':2})
    print feature_ensembled
    feature_ensembled.dump()