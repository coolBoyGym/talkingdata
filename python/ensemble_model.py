import train_impl as ti

if __name__ == '__main__':
    ti.init_constant('concat_6', booster=None, version=1, random_state=0)
    dtrain_train = ti.read_feature(open(ti.PATH_TRAIN_TRAIN), -1)
    dtrain_valid = ti.read_feature(open(ti.PATH_TRAIN_VALID), -1)
    dtest = ti.read_feature(open(ti.PATH_TEST), -1)

    feature_ensembled = ti.ensemble_model(dtrain_train, dtrain_valid, dtest, 'ensmeble_test',
                                          {'gblinear': 2, 'gbtree': 2})
    print feature_ensembled
    feature_ensembled.dump()
