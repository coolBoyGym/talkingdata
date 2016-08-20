import cPickle as pkl
import time

import numpy as np
from scipy.sparse import csr_matrix

import feature
import tf_idf
from train_impl import csr_matrix_2_libsvm, libsvm_2_csr_matrix

data_app_events = '../data/raw/app_events.csv'
data_app_labels = '../data/raw/app_labels.csv'
data_events = '../data/raw/events.csv'
data_gender_age_test = '../data/raw/gender_age_test.csv'
data_gender_age_train = '../data/raw/gender_age_train.csv'
data_label_categories = '../data/raw/label_categories.csv'
data_phone_brand_device_model = '../data/raw/phone_brand_device_model.csv'
data_sample_submission = '../data/raw/sample_submission.csv'


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
    return np.array(train_device_id), np.array(train_label), np.array(test_device_id)


def gather_device_id():
    train_device_id, train_label, test_device_id = read_data()
    train_size = len(train_device_id)
    test_size = len(test_device_id)
    shuffled_indices = np.arange(train_size)
    for i in range(7):
        np.random.shuffle(shuffled_indices)
    data = np.zeros([train_size + test_size, 2])
    data[:train_size, 0] = train_device_id[shuffled_indices]
    data[:train_size, 1] = train_label[shuffled_indices]
    data[train_size:, 0] = test_device_id
    print train_size, test_size, data.shape
    np.savetxt('../feature/device_id', data, header='%d,%d' % (train_size, test_size), fmt='%d,%d')


def get_subset_size():
    with open('../feature/device_id', 'r') as fin:
        header = next(fin).strip().split(' ')[1].split(',')
        train_size = int(header[0])
        test_size = int(header[1])
    return train_size, test_size


def gather_event_id():
    device_data = np.loadtxt('../feature/device_id', skiprows=1, dtype=np.int64, delimiter=',')
    dict_device_event = pkl.load(open('../data/dict_device_event.pkl'))
    train_device_size, test_device_size = get_subset_size()
    event_data = []
    for did, dlabel in device_data[:train_device_size]:
        if did in dict_device_event:
            eids = map(lambda x: x[0], dict_device_event[did])
            event_data.extend(map(lambda x: [did, x, dlabel], eids))
    train_size = len(event_data)
    for did, dlabel in device_data[train_device_size:]:
        if did in dict_device_event:
            eids = map(lambda x: x[0], dict_device_event[did])
            event_data.extend(map(lambda x: [did, x, dlabel], eids))
    event_data = np.array(event_data)
    test_size = len(event_data) - train_size
    print train_size, test_size, event_data.shape
    np.savetxt('../feature/event_id', event_data, header='%d,%d' % (train_size, test_size), fmt='%d,%d,%d')


fea_phone_brand = feature.one_hot_feature(name='phone_brand', dtype='d')
fea_device_model = feature.one_hot_feature(name='device_model', dtype='d')

fea_installed_app = feature.multi_feature(name='installed_app', dtype='d')
fea_active_app = feature.multi_feature(name='active_app', dtype='d')
fea_installed_app_label = feature.multi_feature(name='installed_app_label', dtype='d')
fea_installed_app_label_num = feature.multi_feature(name='installed_app_label_num', dtype='d')
fea_active_app_label = feature.multi_feature(name='active_app_label', dtype='d')
fea_active_app_label_num = feature.multi_feature(name='active_app_label_num', dtype='d')
fea_device_long_lat = feature.multi_feature(name='device_long_lat', dtype='f')
fea_device_event_num = feature.num_feature(name='device_event_num', dtype='d')
fea_device_day_event_num = feature.multi_feature(name='device_day_event_num', dtype='d')
fea_device_hour_event_num = feature.multi_feature(name='device_hour_event_num', dtype='d')
fea_device_day_hour_event_num = feature.multi_feature(name='device_day_hour_event_num', dtype='d')
fea_device_weekday_event_num = feature.multi_feature(name='device_weekday_event_num', dtype='d')

fea_installed_app_freq = feature.multi_feature(name='installed_app_freq', dtype='f')
fea_active_app_freq = feature.multi_feature(name='active_app_freq', dtype='f')
fea_installed_app_label_freq = feature.multi_feature(name='installed_app_label_freq', dtype='f')
fea_active_app_label_freq = feature.multi_feature(name='active_app_label_freq', dtype='f')
fea_device_long_lat_norm = feature.multi_feature(name='device_long_lat_norm', dtype='f')
fea_device_event_num_norm = feature.num_feature(name='device_event_num_norm', dtype='f')
fea_device_day_event_num_norm = feature.multi_feature(name='device_day_event_num_norm', dtype='f')
fea_device_day_event_num_freq = feature.multi_feature(name='device_day_event_num_freq', dtype='f')
fea_device_hour_event_num_norm = feature.multi_feature(name='device_hour_event_num_norm', dtype='f')
fea_device_hour_event_num_freq = feature.multi_feature(name='device_hour_event_num_freq', dtype='f')
fea_device_day_hour_event_num_norm = feature.multi_feature(name='device_day_hour_event_num_norm', dtype='f')
fea_device_weekday_event_num_norm = feature.multi_feature(name='device_weekday_event_num_norm', dtype='f')
fea_device_weekday_event_num_freq = feature.multi_feature(name='device_weekday_event_num_freq', dtype='f')

fea_installed_app_tfidf = feature.multi_feature(name='installed_app_tfidf', dtype='f')
fea_installed_app_label_tfidf = feature.multi_feature(name='installed_app_label_tfidf', dtype='f')

# new features about app label category
fea_active_app_label_category = feature.multi_feature(name='active_app_label_category', dtype='d')
fea_active_app_label_category_num = feature.multi_feature(name='active_app_label_category_num', dtype='d')
fea_active_app_label_diff_hour_category = feature.multi_feature(name='active_app_label_diff_hour_category', dtype='d')
fea_active_app_label_diff_hour_category_num = feature.multi_feature(name='active_app_label_diff_hour_category_num',
                                                                    dtype='d')
fea_active_app_label_each_hour_category = feature.multi_feature(name='active_app_label_each_hour_category', dtype='d')
fea_active_app_label_each_hour_category_num = feature.multi_feature(name='active_app_label_each_hour_category_num',
                                                                    dtype='d')

# new features about app label cluster
fea_active_app_label_cluster_40 = feature.multi_feature(name='active_app_label_cluster_40', dtype='d')
fea_active_app_label_cluster_100 = feature.multi_feature(name='active_app_label_cluster_100', dtype='d')
fea_active_app_label_cluster_270 = feature.multi_feature(name='active_app_label_cluster_270', dtype='d')
fea_active_app_label_cluster_40_num = feature.multi_feature(name='active_app_label_cluster_40_num', dtype='d')

# new feature about tf-idf
fea_active_app_label_category_num_tfidf = feature.multi_feature(name='active_app_label_category_num_tfidf', dtype='f')
fea_active_app_label_num_tfidf = feature.multi_feature(name='active_app_label_num_tfidf', dtype='f')
fea_active_app_label_cluster_40_num_tfidf = feature.multi_feature(name='active_app_label_cluster_40_num_tfidf',
                                                                  dtype='f')

"""
event features
"""
fea_event_time = feature.multi_feature(name='event_time', dtype='d')
fea_event_longitude = feature.num_feature(name='event_longitude', dtype='f')
fea_event_longitude_norm = feature.num_feature(name='event_longitude_norm', dtype='f')
fea_event_latitude = feature.num_feature(name='event_latitude', dtype='f')
fea_event_latitude_norm = feature.num_feature(name='event_latitude_norm', dtype='f')
fea_event_phone_brand = feature.one_hot_feature(name='event_phone_brand', dtype='d')
fea_event_installed_app = feature.multi_feature(name='event_installed_app', dtype='d')
fea_event_installed_app_norm = feature.multi_feature(name='event_installed_app_norm', dtype='f')

"""
concat features
"""
fea_concat_1 = feature.multi_feature(name='concat_1', dtype='f')
fea_concat_1_norm = feature.multi_feature(name='concat_1_norm', dtype='f')
fea_concat_2 = feature.multi_feature(name='concat_2', dtype='f')
fea_concat_2_norm = feature.multi_feature(name='concat_2_norm', dtype='f')
fea_concat_3 = feature.multi_feature(name='concat_3', dtype='f')
fea_concat_3_norm = feature.multi_feature(name='concat_3_norm', dtype='f')
fea_concat_4 = feature.multi_feature(name='concat_4', dtype='f')
fea_concat_4_norm = feature.multi_feature(name='concat_4_norm', dtype='f')
fea_concat_5 = feature.multi_feature(name='concat_5', dtype='f')
fea_concat_5_norm = feature.multi_feature(name='concat_5_norm', dtype='f')
fea_concat_6 = feature.multi_feature(name='concat_6', dtype='d')
fea_concat_6_norm = feature.multi_feature(name='concat_6_norm', dtype='f')
fea_concat_7 = feature.multi_feature(name='concat_7', dtype='d')
fea_concat_7_norm = feature.multi_feature(name='concat_7_norm', dtype='f')
fea_concat_8 = feature.multi_feature(name='concat_8', dtype='d')
fea_concat_8_norm = feature.multi_feature(name='concat_8_norm', dtype='f')
fea_concat_9 = feature.multi_feature(name='concat_9', dtype='d')
fea_concat_9_norm = feature.multi_feature(name='concat_9_norm', dtype='f')
fea_ensemble_test = feature.multi_feature(name='ensmeble_test', dtype='f')

# fea_concat_1 = feature.multi_feature(name='concat_1', dtype='f')
# fea_concat_2 = feature.multi_feature(name='concat_2', dtype='f')
# fea_concat_3 = feature.multi_feature(name='concat_3', dtype='f')
# fea_concat_4 = feature.multi_feature(name='concat_4', dtype='f')
# fea_concat_5 = feature.multi_feature(name='concat_5', dtype='f')
# fea_concat_6 = feature.multi_feature(name='concat_6', dtype='f')

"""
model outputs, for ensemble use
"""
fea_concat_1_gblinear_1 = feature.multi_feature(name='concat_1_gblinear_1', dtype='f')
fea_concat_1_gbtree_1 = feature.multi_feature(name='concat_1_gbtree_1', dtype='f')
fea_concat_2_gblinear_1 = feature.multi_feature(name='concat_2_gblinear_1', dtype='f')
fea_concat_2_gbtree_1 = feature.multi_feature(name='concat_2_gbtree_1', dtype='f')
fea_concat_2_norm_gblinear_1 = feature.multi_feature(name='concat_2_norm_gblinear_1', dtype='f')
fea_concat_2_norm_gbtree_1 = feature.multi_feature(name='concat_2_norm_gbtree_1', dtype='f')
fea_concat_3_gblinear_1 = feature.multi_feature(name='concat_3_gblinear_1', dtype='f')
fea_concat_3_gbtree_1 = feature.multi_feature(name='concat_3_gbtree_1', dtype='f')
fea_concat_3_norm_gblinear_1 = feature.multi_feature(name='concat_3_norm_gblinear_1', dtype='f')
fea_concat_3_norm_gbtree_1 = feature.multi_feature(name='concat_3_norm_gbtree_1', dtype='f')
fea_concat_4_gblinear_1 = feature.multi_feature(name='concat_4_gblinear_1', dtype='f')
fea_concat_4_gbtree_1 = feature.multi_feature(name='concat_4_gbtree_1', dtype='f')
fea_concat_4_norm_gblinear_1 = feature.multi_feature(name='concat_4_norm_gblinear_1', dtype='f')
fea_concat_4_norm_gbtree_1 = feature.multi_feature(name='concat_4_norm_gbtree_1', dtype='f')
fea_concat_5_gblinear_1 = feature.multi_feature(name='concat_5_gblinear_1', dtype='f')
fea_concat_5_gbtree_1 = feature.multi_feature(name='concat_5_gbtree_1', dtype='f')
fea_concat_5_norm_gblinear_1 = feature.multi_feature(name='concat_5_norm_gblinear_1', dtype='f')
fea_concat_5_norm_gbtree_1 = feature.multi_feature(name='concat_5_norm_gbtree_1', dtype='f')
fea_concat_6_gbtree_1 = feature.multi_feature(name='concat_6_gbtree_1', dtype='f')


def make_feature():
    print 'loading data...'
    start_time = time.time()

    # dict_device_brand_model = pkl.load(open('../data/dict_device_brand_model.pkl', 'rb'))
    dict_device_event = pkl.load(open('../data/dict_device_event.pkl', 'rb'))
    dict_app_event = pkl.load(open('../data/dict_app_event.pkl', 'rb'))
    dict_app_label = pkl.load(open('../data/dict_app_label.pkl', 'rb'))
    # dict_event = pkl.load(open('../data/dict_event.pkl', 'rb'))
    # dict_brand = pkl.load(open('../data/dict_id_brand.pkl', 'rb'))
    # dict_model = pkl.load(open('../data/dict_id_model.pkl', 'rb'))
    # dict_app = pkl.load(open('../data/dict_id_app.pkl', 'rb'))
    # dict_label_category_group = pkl.load(open('../data/dict_label_category_group_number.pkl', 'rb'))
    dict_label_cluster_40 = pkl.load(open('../data/dict_label_cluster_40.pkl', 'rb'))
    # dict_label_cluster_100 = pkl.load(open('../data/dict_label_cluster_100.pkl', 'rb'))
    # dict_label_cluster_270 = pkl.load(open('../data/dict_label_cluster_270.pkl', 'rb'))

    # for i in range(3):
    #     print dict_device_event[i]
    #     print "\nA new one!\n"
    #
    # exit(0)

    device_id = np.loadtxt('../feature/device_id', dtype=np.int64, skiprows=1, delimiter=',', usecols=[0])

    print 'finish in %d sec' % (time.time() - start_time)

    # fea_phone_brand.process(device_id=device_id, dict_device_brand_model=dict_device_brand_model)
    # fea_phone_brand.dump()
    #
    # fea_device_model.process(device_id=device_id, dict_device_brand_model=dict_device_brand_model)
    # fea_device_model.dump()
    #
    # fea_installed_app.process(device_id=device_id, dict_device_event=dict_device_event, dict_app_event=dict_app_event)
    # fea_installed_app.dump()
    #
    # # indices, values = fea_installed_app.get_value()
    # # fea_installed_app_norm.process(indices=indices, values=values)
    # # fea_installed_app_norm.dump()
    #
    # fea_active_app.process(device_id=device_id, dict_device_event=dict_device_event, dict_app_event=dict_app_event)
    # fea_active_app.dump()
    #
    # # indices, values = fea_active_app.get_value()
    # # fea_active_app_norm.process(indices=indices, values=values)
    # # fea_active_app_norm.dump()
    #
    # fea_device_long_lat.process(device_id=device_id, dict_device_event=dict_device_event)
    # fea_device_long_lat.dump()
    #
    # fea_device_long_lat_norm.process(device_id=device_id, dict_device_event=dict_device_event)
    # fea_device_long_lat_norm.dump()
    #
    # fea_installed_app_freq.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                dict_app_event=dict_app_event)
    # fea_installed_app_freq.dump()
    #
    # fea_active_app_freq.process(device_id=device_id, dict_device_event=dict_device_event,
    #                             dict_app_event=dict_app_event)
    # fea_active_app_freq.dump()
    #
    # fea_device_event_num.process(device_id=device_id, dict_device_event=dict_device_event)
    # fea_device_event_num.dump()
    #
    # indices, values = fea_device_event_num.get_value()
    # fea_device_event_num_norm.process(indices=indices, values=values)
    # fea_device_event_num_norm.dump()
    #
    # fea_device_day_event_num.process(device_id=device_id, dict_device_event=dict_device_event)
    # fea_device_day_event_num.dump()
    #
    # indices, values = fea_device_day_event_num.get_value()
    # fea_device_day_event_num_norm.process(indices=indices, values=values)
    # fea_device_day_event_num_norm.dump()
    #
    # fea_weekday_event_num.process(device_id=device_id, dict_device_event=dict_device_event)
    # fea_weekday_event_num.dump()
    #
    # indices, values = fea_weekday_event_num.get_value()
    # fea_weekday_event_num_norm.process(indices=indices, values=values)
    # fea_weekday_event_num_norm.dump()
    #
    # event_id = np.loadtxt('../feature/event_id', dtype=np.int64, skiprows=1, usecols=[1], delimiter=',')
    #
    # fea_event_time.process(event_id=event_id, dict_event=dict_event)
    # fea_event_time.dump()
    #
    # fea_event_longitude.process(event_id=event_id, dict_event=dict_event)
    # fea_event_longitude.dump()
    #
    # indices, values = fea_event_longitude.get_value()
    # print np.max(values), np.min(values)
    #
    # fea_event_longitude_norm.process(indices=indices, values=values)
    # fea_event_longitude_norm.dump()
    #
    # fea_event_latitude.process(event_id=event_id, dict_event=dict_event)
    # fea_event_latitude.dump()
    #
    # indices, values = fea_event_latitude.get_value()
    # print np.max(values), np.min(values)
    #
    # fea_event_latitude_norm.process(indices=indices, values=values)
    # fea_event_latitude_norm.dump()
    # fea_event_phone_brand.process(event_id=event_id, dict_event=dict_event,
    #                               dict_device_brand_model=dict_device_brand_model)
    # fea_event_phone_brand.dump()
    #
    # fea_event_installed_app.process(event_id=event_id, dict_app_event=dict_app_event)
    # fea_event_installed_app.dump()
    #
    # indices, values = fea_event_installed_app.get_value()
    # fea_event_installed_app_norm.process(indices=indices, values=values)
    # fea_event_installed_app_norm.dump()
    # fea_installed_app_label.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                 dict_app_event=dict_app_event, dict_app_label=dict_app_label)
    # fea_installed_app_label.dump()
    #
    # fea_active_app_label.process(device_id=device_id, dict_device_event=dict_device_event,
    #                              dict_app_event=dict_app_event, dict_app_label=dict_app_label)
    # fea_active_app_label.dump()
    #
    # fea_installed_app_label_freq.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                      dict_app_event=dict_app_event, dict_app_label=dict_app_label)
    # fea_installed_app_label_freq.dump()

    # fea_installed_app_label_num.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                     dict_app_event=dict_app_event, dict_app_label=dict_app_label)
    # fea_installed_app_label_num.dump()

    # fea_active_app_label_freq.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                   dict_app_event=dict_app_event, dict_app_label=dict_app_label)
    # fea_active_app_label_freq.dump()

    # fea_active_app_label_num.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                  dict_app_event=dict_app_event, dict_app_label=dict_app_label)
    # fea_active_app_label_num.dump()

    # fea_device_hour_event_num.process(device_id=device_id, dict_device_event=dict_device_event)
    # fea_device_hour_event_num.dump()
    #
    # indices, values = fea_device_hour_event_num.get_value()
    # fea_device_hour_event_num_norm.process(indices=indices, values=values)
    # fea_device_hour_event_num_norm.dump()

    # fea_device_day_hour_event_num.process(device_id=device_id, dict_device_event=dict_device_event)
    # fea_device_day_hour_event_num.dump()
    #
    # indices, values = fea_device_day_hour_event_num.get_value()
    # fea_device_day_hour_event_num_norm.process(indices=indices, values=values)
    # fea_device_day_hour_event_num_norm.dump()

    # fea_device_weekday_event_num_freq.process(device_id=device_id, dict_device_event=dict_device_event)
    # fea_device_weekday_event_num_freq.dump()

    # fea_device_hour_event_num_freq.process(device_id=device_id, dict_device_event=dict_device_event)
    # fea_device_hour_event_num_freq.dump()

    # fea_active_app_label_category.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                       dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                       dict_label_category_group=dict_label_category_group)
    # fea_active_app_label_category.dump()
    # fea_active_app_label_category_num.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                           dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                           dict_label_category_group=dict_label_category_group)
    # fea_active_app_label_category_num.dump()
    #
    # fea_active_app_label_diff_hour_category.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                                 dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                                 dict_label_category_group=dict_label_category_group)
    # fea_active_app_label_diff_hour_category.dump()
    #
    # fea_active_app_label_diff_hour_category_num.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                                     dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                                     dict_label_category_group=dict_label_category_group)
    # fea_active_app_label_diff_hour_category_num.dump()

    # fea_active_app_label_each_hour_category.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                                 dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                                 dict_label_category_group=dict_label_category_group)
    # fea_active_app_label_each_hour_category.dump()
    # fea_active_app_label_each_hour_category_num.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                                     dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                                     dict_label_category_group=dict_label_category_group)
    # fea_active_app_label_each_hour_category_num.dump()

    # fea_active_app_label_cluster_40.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                         dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                         dict_label_cluster_40=dict_label_cluster_40)
    # fea_active_app_label_cluster_40.dump()

    # fea_active_app_label_cluster_100.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                          dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                          dict_label_cluster_100=dict_label_cluster_100)
    # fea_active_app_label_cluster_100.dump()

    # fea_active_app_label_cluster_270.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                          dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                          dict_label_cluster_270=dict_label_cluster_270)
    # fea_active_app_label_cluster_270.dump()
    fea_active_app_label_cluster_40_num.process(device_id=device_id, dict_device_event=dict_device_event,
                                                dict_app_event=dict_app_event, dict_app_label=dict_app_label,
                                                dict_label_cluster_40=dict_label_cluster_40)
    fea_active_app_label_cluster_40_num.dump()


def ensemble_concat_feature(name, fea_list):
    spaces = []
    for fea in fea_list:
        spaces.append(fea.get_space())
    print 'spaces', str(spaces)

    fea_concat = feature.multi_feature(name=name)

    collect_indices = []
    collect_values = []

    for i in range(len(fea_list)):
        fea = fea_list[i]
        concat_indices, concat_values = fea.get_value()
        concat_indices += sum(spaces[:i])
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
        concat_values.append(tmp_values)

    concat_indices = np.array(concat_indices)
    concat_values = np.array(concat_values)

    fea_concat.set_value(indices=concat_indices, values=concat_values)
    max_indices = map(feature.get_max, concat_indices)
    len_indices = map(lambda x: len(x), concat_values)
    fea_concat.set_space(max(max_indices) + 1)
    fea_concat.set_rank(max(len_indices))
    fea_concat.set_size(len(concat_indices))

    return fea_concat


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

    fea_concat = feature.multi_feature(name=name)

    collect_indices = []
    collect_values = []

    for i in range(len(fea_list)):
        fea = fea_list[i]
        concat_indices, concat_values = fea.get_value()
        concat_indices += sum(spaces[:i])
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
        concat_values.append(tmp_values)

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


def csr_2_feature(name, csr_mat, reorder=False):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    values = csr_mat.data
    libsvm_indices = []
    libsvm_values = []
    for i in range(csr_mat.shape[0]):
        libsvm_indices.append(indices[indptr[i]:indptr[i + 1]])
        libsvm_values.append(values[indptr[i]:indptr[i + 1]])
    libsvm_indices, libsvm_values = np.array(libsvm_indices), np.array(libsvm_values)
    fea_csr = feature.multi_feature(name=name, dtype='f')
    fea_csr.set_value(indices=libsvm_indices, values=libsvm_values)
    max_indices = map(feature.get_max, libsvm_indices)
    len_indices = map(lambda x: len(x), libsvm_values)
    fea_csr.set_space(max(max_indices) + 1)
    fea_csr.set_rank(max(len_indices))
    fea_csr.set_size(len(libsvm_indices))
    if reorder:
        fea_csr.reorder()
    fea_csr.dump()


def padding_zero(line, space):
    line_space = int(line.strip().split()[-1].split(':')[0]) + 1
    if line_space < space:
        return line.strip() + ' %d:0\n' % (space - 1)
    else:
        return line.strip() + '\n'


def split_dataset(name, cv_rate, zero_pad=False):
    train_label = np.loadtxt('../feature/device_id', delimiter=',', dtype=np.int64, skiprows=1, usecols=[1])

    with open('../feature/' + name, 'r') as data_in:
        header = next(data_in)
        space = int(header.strip().split()[2])
        train_size, test_size = get_subset_size()

        with open('../input/' + name + '.train', 'w') as train_out:
            first_line = next(data_in)
            if zero_pad:
                first_line = padding_zero(first_line, space)
            train_out.write('%d %s' % (train_label[0], first_line))
            for i in range(1, train_size):
                train_out.write('%d %s' % (train_label[i], next(data_in)))

        with open('../input/' + name + '.test', 'w') as test_out:
            first_line = next(data_in)
            if zero_pad:
                first_line = padding_zero(first_line, space)
            test_out.write('0 %s' % first_line)
            for line in data_in:
                test_out.write('0 %s' % line)

    valid_size = int(train_size * cv_rate)
    with open('../input/' + name + '.train', 'r') as train_in:
        with open('../input/' + name + '.train.valid', 'w') as valid_out:
            first_line = next(train_in)
            if zero_pad:
                first_line = padding_zero(first_line, space)
            valid_out.write(first_line)
            for i in range(1, valid_size):
                valid_out.write(next(train_in))

        with open('../input/' + name + '.train.train', 'w') as train_out:
            first_line = next(train_in)
            if zero_pad:
                first_line = padding_zero(first_line, space)
            train_out.write(first_line)
            for line in train_in:
                train_out.write(line)

    print 'train_size', train_size, 'valid_size', valid_size, 'test_size', test_size


# def get_hour_event_num_distribution(device_id, dict_device_event):
#     tmp = {}
#     for did in device_id:
#         if did in dict_device_event:
#             hours = map(lambda x: get_time(x[1], ['hour'])[0], dict_device_event[did])
#             for d in hours:
#                 if d in tmp:
#                     tmp[d] += 1
#                 else:
#                     tmp[d] = 1
#
#     sum_tmp = 0.0
#     for k in tmp.keys():
#         sum_tmp += tmp[k]
#     res = {}
#     for i in range(len(tmp)):
#         res[i] = tmp[i] / sum_tmp
#
#     sorted_tmp = sorted(res.keys())
#     indices = sorted_tmp
#     values = map(lambda x: res[x], sorted_tmp)
#     return indices, tmp, values

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def process_keras_data(keras_data_name):
    train_data_csr = load_sparse_csr('../data/' + keras_data_name + '_train_csr.npz')
    test_data_csr = load_sparse_csr('../data/' + keras_data_name + '_test_csr.npz')
    train_indices, train_values = csr_matrix_2_libsvm(train_data_csr)
    test_indices, test_values = csr_matrix_2_libsvm(test_data_csr)
    train_device_id = np.loadtxt(data_gender_age_train, delimiter=',', skiprows=1, usecols=[0], dtype=np.int64)
    dict_device_id = pkl.load(open('../data/dict_id_device.pkl'))
    dict_device_train = {}
    for i in range(len(train_device_id)):
        device_id = train_device_id[i]
        did = dict_device_id[device_id]
        dict_device_train[did] = (train_indices[i], train_values[i])
    device_id = np.loadtxt('../feature/device_id', delimiter=',', skiprows=1, usecols=[0])
    fea_indices = []
    fea_values = []
    for i in range(len(train_device_id)):
        did = device_id[i]
        fea_indices.append(dict_device_train[did][0])
        fea_values.append(np.int64(dict_device_train[did][1]))
    for i in range(len(train_device_id), len(device_id)):
        fea_indices.append(test_indices[i - len(train_device_id)])
        fea_values.append(np.int64(test_values[i - len(train_device_id)]))
    fea_indices = np.array(fea_indices)
    fea_values = np.array(fea_values)
    fea_tmp = feature.multi_feature(name=keras_data_name, dtype='d')
    fea_tmp.set_value(indices=fea_indices, values=fea_values)
    max_indices = map(feature.get_max, fea_indices)
    len_indices = map(lambda x: len(x), fea_values)
    fea_tmp.set_space(max(max_indices) + 1)
    fea_tmp.set_rank(max(len_indices))
    fea_tmp.set_size(len(fea_indices))
    fea_tmp.dump()


def feature_tfidf(name):
    fea_tmp = feature.multi_feature(name=name, dtype='f')
    fea_tmp.load()
    feature_indices, feature_values = fea_tmp.get_value()
    feature_space = fea_tmp.get_space()
    csr_fea = libsvm_2_csr_matrix(feature_indices, feature_values, feature_space)
    csr_tfidf = tf_idf.tf_idf(csr_fea)
    name_out = name + '_tfidf'
    csr_2_feature(name_out, csr_tfidf, reorder=True)


if __name__ == '__main__':
    print 'processing features...'
    # gather_device_id()
    # gather_event_id()

    # make_feature()
    # fea_concat_6.load()
    # fea_concat_6.set_data_type('d')
    # fea_concat_6.dump()

    # concat_feature('concat_1', [fea_phone_brand,
    #                             fea_device_model, ])
    #
    # concat_feature('concat_2', [fea_phone_brand,
    #                             fea_device_model,
    #                             fea_device_long_lat, ])
    #
    # concat_feature('concat_3', [fea_phone_brand,
    #                             fea_device_model,
    #                             fea_device_long_lat,
    #                             fea_device_event_num,
    #                             fea_device_day_event_num,
    #                             fea_device_hour_event_num,
    #                             fea_device_day_hour_event_num, ])
    #
    # concat_feature('concat_4', [fea_phone_brand,
    #                             fea_device_model,
    #                             fea_device_long_lat,
    #                             fea_device_event_num,
    #                             fea_device_day_event_num,
    #                             fea_device_hour_event_num,
    #                             fea_device_day_hour_event_num,
    #                             fea_installed_app,
    #                             fea_active_app, ])
    #
    # concat_feature('concat_5', [fea_phone_brand,
    #                             fea_device_model,
    #                             fea_device_long_lat,
    #                             fea_device_event_num,
    #                             fea_device_day_event_num,
    #                             fea_device_hour_event_num,
    #                             fea_device_day_hour_event_num,
    #                             fea_installed_app,
    #                             fea_active_app,
    #                             fea_installed_app_label,
    #                             fea_active_app_label, ])
    #
    # concat_feature('concat_2_norm', [fea_phone_brand,
    #                                  fea_device_model,
    #                                  fea_device_long_lat_norm, ])
    #
    # concat_feature('concat_3_norm', [fea_phone_brand,
    #                                  fea_device_model,
    #                                  fea_device_long_lat_norm,
    #                                  fea_device_event_num_norm,
    #                                  fea_device_day_event_num_norm,
    #                                  fea_device_hour_event_num_norm,
    #                                  fea_device_day_hour_event_num_norm])
    #
    # concat_feature('concat_4_norm', [fea_phone_brand,
    #                                  fea_device_model,
    #                                  fea_device_long_lat_norm,
    #                                  fea_device_event_num_norm,
    #                                  fea_device_day_event_num_norm,
    #                                  fea_device_hour_event_num_norm,
    #                                  fea_device_day_hour_event_num_norm,
    #                                  fea_installed_app_freq,
    #                                  fea_active_app_freq, ])
    #
    # concat_feature('concat_5_norm', [fea_phone_brand,
    #                                  fea_device_model,
    #                                  fea_device_long_lat_norm,
    #                                  fea_device_event_num_norm,
    #                                  fea_device_day_event_num_norm,
    #                                  fea_device_hour_event_num_norm,
    #                                  fea_device_day_hour_event_num_norm,
    #                                  fea_installed_app_freq,
    #                                  fea_active_app_freq,
    #                                  fea_installed_app_label_freq,
    #                                  fea_active_app_label_freq, ])

    # concat_feature('concat_6', [fea_phone_brand,
    #                             fea_device_model,
    #                             fea_installed_app,
    #                             fea_installed_app_label])

    # concat_feature('concat_7', [fea_phone_brand,
    #                             fea_device_model,
    #                             fea_installed_app,
    #                             fea_installed_app_label,
    #                             fea_device_long_lat])

    # concat_feature('concat_7_norm', [fea_phone_brand,
    #                                  fea_device_model,
    #                                  fea_installed_app,
    #                                  fea_installed_app_label,
    #                                  fea_device_long_lat_norm])

    # concat_feature('concat_8', [fea_phone_brand,
    #                             fea_device_model,
    #                             fea_installed_app,
    #                             fea_installed_app_label,
    #                             fea_device_event_num,
    #                             fea_device_weekday_event_num,
    #                             fea_device_day_event_num,
    #                             fea_device_hour_event_num, ])
    #
    # concat_feature('concat_8_norm', [fea_phone_brand,
    #                                  fea_device_model,
    #                                  fea_installed_app,
    #                                  fea_installed_app_label,
    #                                  fea_device_event_num_norm,
    #                                  fea_device_weekday_event_num_norm,
    #                                  fea_device_day_event_num_norm,
    #                                  fea_device_hour_event_num_norm, ])

    # concat_feature('concat_9', [fea_phone_brand,
    #                             fea_device_model,
    #                             fea_installed_app,
    #                             fea_installed_app_label,
    #                             fea_device_long_lat,
    #                             fea_device_event_num,
    #                             fea_device_weekday_event_num,
    #                             fea_device_day_event_num,
    #                             fea_device_hour_event_num, ])

    # concat_feature('concat_9_norm', [fea_phone_brand,
    #                                  fea_device_model,
    #                                  fea_installed_app,
    #                                  fea_installed_app_label,
    #                                  fea_device_long_lat_norm,
    #                                  fea_device_event_num_norm,
    #                                  fea_device_weekday_event_num_norm,
    #                                  fea_device_day_event_num_norm,
    #                                  fea_device_hour_event_num_norm, ])

    # concat_feature('concat_10', [fea_phone_brand,
    #                              fea_device_model,
    #                              fea_installed_app,
    #                              fea_installed_app_label,
    #                              fea_device_hour_event_num_freq,
    #                              fea_device_weekday_event_num_freq])

    # concat_feature('concat_11', [fea_phone_brand,
    #                              fea_device_model,
    #                              fea_installed_app,
    #                              fea_installed_app_label,
    #                              fea_active_app_label_category])
    #
    # concat_feature('concat_12', [fea_phone_brand,
    #                              fea_device_model,
    #                              fea_installed_app,
    #                              fea_installed_app_label,
    #                              fea_active_app_label_diff_hour_category])

    # concat_feature('concat_13', [fea_phone_brand,
    #                              fea_device_model,
    #                              fea_installed_app,
    #                              fea_installed_app_label,
    #                              fea_active_app_label_each_hour_category])

    # concat_feature('concat_14', [fea_phone_brand,
    #                              fea_device_model,
    #                              fea_installed_app,
    #                              fea_installed_app_label,
    #                              fea_active_app_label,
    #                              fea_device_hour_event_num_freq,
    #                              fea_device_weekday_event_num_freq,
    #                              fea_device_long_lat_norm,
    #                              ])

    # concat_feature('concat_15', [fea_phone_brand,
    #                              fea_device_model,
    #                              fea_installed_app,
    #                              fea_installed_app_label,
    #                              fea_active_app_label_category_num_tfidf])

    # concat_feature('concat_16', [fea_phone_brand,
    #                              fea_device_model,
    #                              fea_installed_app,
    #                              fea_installed_app_label,
    #                              fea_active_app_label_cluster_40])

    # concat_feature('concat_16_tfidf', [fea_phone_brand,
    #                                    fea_device_model,
    #                                    fea_installed_app,
    #                                    fea_installed_app_label,
    #                                    fea_active_app_label_cluster_40_num_tfidf])

    # concat_feature('concat_16_2', [fea_phone_brand,
    #                                fea_device_model,
    #                                fea_installed_app,
    #                                fea_installed_app_label,
    #                                fea_active_app_label_cluster_100])

    # concat_feature('concat_16_3', [fea_phone_brand,
    #                                fea_device_model,
    #                                fea_installed_app,
    #                                fea_installed_app_label,
    #                                fea_active_app_label_cluster_270])

    # concat_feature('concat_17', [fea_phone_brand,
    #                              fea_device_model,
    #                              fea_installed_app,
    #                              fea_installed_app_label,
    #                              fea_active_app_label_num_tfidf])

    # concat_feature('ensemble_4', [fea_concat_1_gbtree_1,
    #                               fea_concat_1_gblinear_1,
    #                               fea_concat_2_gbtree_1,
    #                               fea_concat_2_gblinear_1,
    #                               fea_concat_2_norm_gbtree_1,
    #                               fea_concat_2_norm_gblinear_1,
    #                               fea_concat_3_gbtree_1,
    #                               fea_concat_3_gblinear_1,
    #                               fea_concat_3_norm_gbtree_1,
    #                               fea_concat_3_norm_gblinear_1,
    #                               fea_concat_4_gbtree_1,
    #                               fea_concat_4_gblinear_1,
    #                               fea_concat_4_norm_gbtree_1,
    #                               fea_concat_4_norm_gblinear_1,
    #                               fea_concat_5_gbtree_1,
    #                               fea_concat_5_gblinear_1,
    #                               # fea_concat_5_norm_gbtree_1,
    #                               # fea_concat_5_norm_gblinear_1,
    #                               fea_concat_6_gbtree_1,
    #                               fea_concat_6])
    #

    # concat_feature('concat_6_tfidf', [fea_phone_brand,
    #                                   fea_device_model,
    #                                   fea_installed_app_tfidf,
    #                                   fea_installed_app_label_tfidf])

    # concat_feature('concat_20', [fea_phone_brand,
    #                              fea_device_model,
    #                              fea_installed_app,
    #                              fea_installed_app_label,
    #                              fea_active_app,
    #                              fea_active_app_label])

    # concat_feature('concat_21', [fea_phone_brand,
    #                              fea_device_model,
    #                              fea_installed_app,
    #                              fea_installed_app_label,
    #                              fea_device_day_event_num_freq,
    #                              fea_device_hour_event_num_freq,
    #                              fea_device_weekday_event_num_freq,
    #                              fea_device_long_lat_norm,
    #                              fea_active_app_label_category,
    #                              fea_active_app_label_cluster_40])

    # split_dataset('concat_6_tfidf', 0.2, zero_pad=True)
    # split_dataset('bagofapps', 0.2, zero_pad=True)
    # split_dataset('concat_12', 0.2, zero_pad=True)
    # make_feature()
    # split_dataset('concat_13', 0.2, zero_pad=True)
    # split_dataset('concat_20', 0.2, zero_pad=True)
    # make_feature()

    # feature_tfidf('active_app_label_diff_hour_category_num')
    # feature_tfidf('installed_app')
    # feature_tfidf('installed_app_label')

    # feature_tfidf('active_app_label_cluster_40_num')

    # dict_device_event = pkl.load(open('../data/dict_device_event.pkl', 'rb'))
    # device_id = np.loadtxt('../feature/device_id', dtype=np.int64, skiprows=1, delimiter=',', usecols=[0])
    #
    # indices, res, num = get_hour_event_num_distribution(device_id=device_id, dict_device_event=dict_device_event)
    # print indices
    # print res
    # print num

    # train_data_csr = load_sparse_csr('../input/bagofapps_train_csr.npz')
    # test_data_csr = load_sparse_csr('../input/bagofapps_test_csr.npz')
    # train_label = np.load('../input/bagofapps_train_label.npy')
