import cPickle as pkl
import time

import numpy as np

import feature

data_app_events = '../data/raw/app_events.csv'
data_app_labels = '../data/raw/app_labels.csv'
data_events = '../data/raw/events.csv'
data_gender_age_test = '../data/raw/gender_age_test.csv'
data_gender_age_train = '../data/raw/gender_age_train.csv'
data_label_categories = '../data/raw/label_categories.csv'
data_phone_brand_device_model = '../data/raw/phone_brand_device_model.csv'
data_sample_submission = '../data/raw/sample_submission.csv'

print 'loading data...'

start_time = time.time()

dict_device = pkl.load(open('../data/dict_id_device.pkl', 'rb'))
dict_device_event = pkl.load(open('../data/dict_device_event.pkl', 'rb'))
dict_app_event = pkl.load(open('../data/dict_app_event.pkl', 'rb'))
dict_app = pkl.load(open('../data/dict_id_app.pkl', 'rb'))
dict_device_brand_model = pkl.load(open('../data/dict_device_brand_model.pkl', 'rb'))
dict_brand = pkl.load(open('../data/dict_id_brand.pkl', 'rb'))
dict_model = pkl.load(open('../data/dict_id_model.pkl', 'rb'))

print 'finish in %d sec' % (time.time() - start_time)


def read_data():
    groups = ['F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42', 'F43+', 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38',
              'M39+']
    group_id = {}
    for i, v in enumerate(groups):
        group_id[v] = i

    train_data = np.loadtxt(data_gender_age_train, delimiter=',', skiprows=1, usecols=[0, 3],
                            dtype=[('device_id', np.int64), ('group', 'S10')])
    test_data = np.loadtxt(data_gender_age_test, delimiter=',', skiprows=1, dtype=np.int64)

    train_device_id = map(lambda d: dict_device[d], train_data['device_id'])
    train_label = map(lambda d: group_id[d], train_data['group'])
    test_device_id = map(lambda d: dict_device[d], test_data)
    return train_device_id, train_label, test_device_id


def gather_device_id():
    train_device_id, _, test_device_id = read_data()
    print len(train_device_id)
    train_size = len(train_device_id)

    device_id = train_device_id
    device_id.extend(test_device_id)
    device_id = np.array(device_id)
    print device_id, device_id.shape

    np.savetxt('../feature/device_id', device_id, header='%d' % train_size, fmt='%d')


def make_feature():
    device_id = np.loadtxt('../feature/device_id', dtype=np.int64, skiprows=1)

    fea_phone_brand = feature.one_hot_feature(name='phone_brand', dtype='d', space=len(dict_brand))
    fea_device_model = feature.one_hot_feature(name='device_model', dtype='d', space=len(dict_model))
    fea_installed_app = feature.multi_feature(name='installed_app', dtype='d', space=len(dict_app))
    fea_active_app = feature.multi_feature(name='active_app', dtype='d', space=len(dict_app))
    fea_installed_app_norm = feature.multi_feature(name='installed_app_norm', dtype='f', space=len(dict_app))
    fea_active_app_norm = feature.multi_feature(name='active_app_norm', dtype='f', space=len(dict_app))

    fea_phone_brand.process(device_id=device_id, dict_device_brand_model=dict_device_brand_model)
    fea_phone_brand.dump()

    fea_device_model.process(device_id=device_id, dict_device_brand_model=dict_device_brand_model)
    fea_device_model.dump()

    fea_installed_app.process(device_id=device_id, dict_device_event=dict_device_event, dict_app_event=dict_app_event)
    fea_installed_app.dump()

    indices, values = fea_installed_app.get_value()
    fea_installed_app_norm.process(indices=indices, values=values)
    fea_installed_app_norm.dump()

    fea_active_app.process(device_id=device_id, dict_device_event=dict_device_event, dict_app_event=dict_app_event)
    fea_active_app.dump()

    indices, values = fea_active_app.get_value()
    fea_active_app_norm.process(indices=indices, values=values)
    fea_active_app_norm.dump()


make_feature()
