import train_impl as ti
import feature_factory as ff
if __name__ == '__main__':
    ti.init_constant('bagofapps', booster=None, version=1, random_state=0)
    dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
    dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
    dtest = ti.read_feature(open(ti.PATH_TEST), -1)

    features_for_ensemble = ti.ensemble_model(dtrain_train, dtrain_valid, dtest,
                                          {'mlp': 1, 'randomforest': 1, 'gblinear': 1, 'gbtree': 1})
    feature_ensembled = ff.ensemble_concat_feature('ensemble_test', features_for_ensemble)
    print feature_ensembled
    feature_ensembled.dump()
