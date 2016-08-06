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

print 'finish in %d sec' % (time.time() - start_time)


def read_data():
    dict_device = pkl.load(open('../data/dict_id_device.pkl', 'rb'))

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


# fea_phone_brand = feature.one_hot_feature(name='phone_brand', dtype='d', space=len(dict_brand))
# fea_device_model = feature.one_hot_feature(name='device_model', dtype='d', space=len(dict_model))
# fea_installed_app = feature.multi_feature(name='installed_app', dtype='d', space=len(dict_app))
# fea_active_app = feature.multi_feature(name='active_app', dtype='d', space=len(dict_app))
# fea_installed_app_norm = feature.multi_feature(name='installed_app_norm', dtype='f', space=len(dict_app))
# fea_active_app_norm = feature.multi_feature(name='active_app_norm', dtype='f', space=len(dict_app))


fea_phone_brand = feature.one_hot_feature(name='phone_brand')
fea_device_model = feature.one_hot_feature(name='device_model')
fea_installed_app = feature.multi_feature(name='installed_app')
fea_active_app = feature.multi_feature(name='active_app')
fea_installed_app_norm = feature.multi_feature(name='installed_app_norm')
fea_active_app_norm = feature.multi_feature(name='active_app_norm')


def make_feature():
    dict_device_brand_model = pkl.load(open('../data/dict_device_brand_model.pkl', 'rb'))
    dict_device_event = pkl.load(open('../data/dict_device_event.pkl', 'rb'))
    dict_app_event = pkl.load(open('../data/dict_app_event.pkl', 'rb'))
    # dict_brand = pkl.load(open('../data/dict_id_brand.pkl', 'rb'))
    # dict_model = pkl.load(open('../data/dict_id_model.pkl', 'rb'))
    # dict_app = pkl.load(open('../data/dict_id_app.pkl', 'rb'))

    device_id = np.loadtxt('../feature/device_id', dtype=np.int64, skiprows=1)

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


# make_feature()


def concat_feature(name, fea_list):
    extra = ','.join([fea.get_name() for fea in fea_list])
    print 'concat feature list', extra

    print 'loading features...'
    start_time = time.time()

    spaces = []
    for fea in fea_list:
        fea.load()
        spaces.append(fea.get_space())

    print 'finish in %d sec' % (time.time() - start_time)
    print 'spaces', str(spaces)

    fea_concat = feature.multi_feature(name=name, dtype='f')

    collect_indices = []
    collect_values = []

    for i in range(len(fea_list)):
        fea = fea_list[i]
        concat_indices, concat_values = fea.get_value()
        concat_indices += sum(spaces[:i + 1])
        collect_indices.append(concat_indices)
        collect_values.append(concat_values)

    concat_indices = []
    concat_values = []
    for i in range(fea_list[0].get_size()):
        tmp_indices = []
        tmp_values = []
        for j in range(len(fea_list)):
            tmp_indices.extend(feature.get_array(collect_indices[j][i]))
            tmp_values.extend(feature.get_array(collect_values[j][i]))
        concat_indices.append(np.array(tmp_indices))
        concat_values.append(np.array(tmp_values))

    concat_indices = np.array(concat_indices)
    concat_values = np.array(concat_values)

    print concat_indices
    print concat_values

    fea_concat.set_value(indices=concat_indices, values=concat_values)
    max_indices = map(feature.get_max, concat_indices)
    len_indices = map(lambda x: len(x), concat_values)
    fea_concat.set_space(max(max_indices) + 1)
    fea_concat.set_rank(max(len_indices))
    fea_concat.set_size(len(concat_indices))

    fea_concat.dump(extra=extra)


# concat_feature('concat_3', [fea_phone_brand, fea_device_model, fea_installed_app_norm, fea_active_app_norm])

fea_concat_1 = feature.multi_feature(name='concat_1')
fea_concat_1.load()
print fea_concat_1.get_value()
print fea_concat_1.get_name()
print fea_concat_1.get_feature_type()
print fea_concat_1.get_data_type()
print fea_concat_1.get_space()
print fea_concat_1.get_rank()
print fea_concat_1.get_size()

fea_concat_2 = feature.multi_feature(name='concat_2')
fea_concat_2.load()
print fea_concat_2.get_value()

fea_concat_3 = feature.multi_feature(name='concat_3')
fea_concat_3.load()
print fea_concat_3.get_value()
