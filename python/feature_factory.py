import cPickle as pkl
import time

import numpy as np
from scipy.sparse import csr_matrix

import feature
import tf_idf
import utils

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


fea_phone_brand_embedding = feature.MultiFeature(name='phone_brand_embedding_1', dtype='f')
fea_device_model_embedding = feature.MultiFeature(name='device_model_embedding_1', dtype='f')
fea_installed_app_embedding = feature.MultiFeature(name='installed_app_embedding_1', dtype='f')
fea_installed_app_label_embedding = feature.MultiFeature(name='installed_app_label_embedding_1', dtype='f')

fea_phone_brand = feature.OneHotFeature(name='phone_brand', dtype='d')
fea_device_model = feature.OneHotFeature(name='device_model', dtype='d')

fea_installed_app = feature.MultiFeature(name='installed_app', dtype='d')
fea_active_app = feature.MultiFeature(name='active_app', dtype='d')
fea_active_app_num = feature.MultiFeature(name='active_app_num', dtype='d')
fea_installed_app_label = feature.MultiFeature(name='installed_app_label', dtype='d')
fea_installed_app_label_num = feature.MultiFeature(name='installed_app_label_num', dtype='d')
fea_active_app_label = feature.MultiFeature(name='active_app_label', dtype='d')
fea_active_app_label_num = feature.MultiFeature(name='active_app_label_num', dtype='d')
fea_device_long_lat = feature.MultiFeature(name='device_long_lat', dtype='f')
fea_device_event_num = feature.NumFeature(name='device_event_num', dtype='d')
fea_device_day_event_num = feature.MultiFeature(name='device_day_event_num', dtype='d')
fea_device_day_event_num_new = feature.MultiFeature(name='device_day_event_num_new', dtype='d')
fea_device_hour_event_num = feature.MultiFeature(name='device_hour_event_num', dtype='d')
fea_device_day_hour_event_num = feature.MultiFeature(name='device_day_hour_event_num', dtype='d')
fea_device_weekday_event_num = feature.MultiFeature(name='device_weekday_event_num', dtype='d')

fea_installed_app_freq = feature.MultiFeature(name='installed_app_freq', dtype='f')
fea_active_app_freq = feature.MultiFeature(name='active_app_freq', dtype='f')
fea_installed_app_label_freq = feature.MultiFeature(name='installed_app_label_freq', dtype='f')
fea_active_app_label_freq = feature.MultiFeature(name='active_app_label_freq', dtype='f')
fea_device_long_lat_norm = feature.MultiFeature(name='device_long_lat_norm', dtype='f')
fea_device_event_num_norm = feature.NumFeature(name='device_event_num_norm', dtype='f')
fea_device_day_event_num_norm = feature.MultiFeature(name='device_day_event_num_norm', dtype='f')
fea_device_day_event_num_freq = feature.MultiFeature(name='device_day_event_num_freq', dtype='f')
fea_device_day_event_num_freq_new = feature.MultiFeature(name='device_day_event_num_freq_new', dtype='f')
fea_device_hour_event_num_norm = feature.MultiFeature(name='device_hour_event_num_norm', dtype='f')
fea_device_hour_event_num_freq = feature.MultiFeature(name='device_hour_event_num_freq', dtype='f')
fea_device_day_hour_event_num_norm = feature.MultiFeature(name='device_day_hour_event_num_norm', dtype='f')
fea_device_weekday_event_num_norm = feature.MultiFeature(name='device_weekday_event_num_norm', dtype='f')
fea_device_weekday_event_num_freq = feature.MultiFeature(name='device_weekday_event_num_freq', dtype='f')

fea_installed_app_tfidf = feature.MultiFeature(name='installed_app_tfidf', dtype='f')
fea_installed_app_label_tfidf = feature.MultiFeature(name='installed_app_label_tfidf', dtype='f')
fea_active_app_num_tfidf = feature.MultiFeature(name='active_app_num_tfidf', dtype='f')

# new features about app label category
fea_active_app_label_category = feature.MultiFeature(name='active_app_label_category', dtype='d')
fea_active_app_label_category_num = feature.MultiFeature(name='active_app_label_category_num', dtype='d')
fea_active_app_label_diff_hour_category = feature.MultiFeature(name='active_app_label_diff_hour_category', dtype='d')
fea_active_app_label_diff_hour_category_num = feature.MultiFeature(name='active_app_label_diff_hour_category_num',dtype='d')
fea_active_app_label_diff_hour_category_freq = feature.MultiFeature(name='active_app_label_diff_hour_category_freq',dtype='f')
fea_active_app_label_each_hour_category = feature.MultiFeature(name='active_app_label_each_hour_category', dtype='d')
fea_active_app_label_each_hour_category_num = feature.MultiFeature(name='active_app_label_each_hour_category_num',dtype='d')
fea_active_app_label_each_hour_category_freq = feature.MultiFeature(name='active_app_label_each_hour_category_freq', dtype='f')

# new features about app label cluster
fea_active_app_label_cluster_40 = feature.MultiFeature(name='active_app_label_cluster_40', dtype='d')
fea_active_app_label_cluster_100 = feature.MultiFeature(name='active_app_label_cluster_100', dtype='d')
fea_active_app_label_cluster_270 = feature.MultiFeature(name='active_app_label_cluster_270', dtype='d')
fea_active_app_label_cluster_40_num = feature.MultiFeature(name='active_app_label_cluster_40_num', dtype='d')

# new feature about tf-idf
fea_active_app_label_category_num_tfidf = feature.MultiFeature(name='active_app_label_category_num_tfidf', dtype='f')
fea_active_app_label_num_tfidf = feature.MultiFeature(name='active_app_label_num_tfidf', dtype='f')
fea_active_app_label_cluster_40_num_tfidf = feature.MultiFeature(name='active_app_label_cluster_40_num_tfidf',
                                                                 dtype='f')
fea_model_cluster_1 = feature.MultiFeature(name='model_cluster_1', dtype='f')

"""
embedding features
"""

fea_installed_app_w2v_8 = feature.MultiFeature(name='installed_app_w2v_8', dtype='f')
fea_installed_app_label_w2v_8 = feature.MultiFeature(name='installed_app_label_w2v_8', dtype='f')
fea_installed_app_w2v_16 = feature.MultiFeature(name='installed_app_w2v_16', dtype='f')
fea_installed_app_label_w2v_16 = feature.MultiFeature(name='installed_app_label_w2v_16', dtype='f')
fea_installed_app_w2v_32 = feature.MultiFeature(name='installed_app_w2v_32', dtype='f')
fea_installed_app_label_w2v_32 = feature.MultiFeature(name='installed_app_label_w2v_32', dtype='f')
fea_installed_app_w2v_64 = feature.MultiFeature(name='installed_app_w2v_64', dtype='f')
fea_installed_app_label_w2v_64 = feature.MultiFeature(name='installed_app_label_w2v_64', dtype='f')
fea_installed_app_w2v_128 = feature.MultiFeature(name='installed_app_w2v_128', dtype='f')
fea_installed_app_label_w2v_128 = feature.MultiFeature(name='installed_app_label_w2v_128', dtype='f')

"""
event features
"""
fea_event_time = feature.MultiFeature(name='event_time', dtype='d')
fea_event_longitude = feature.NumFeature(name='event_longitude', dtype='f')
fea_event_longitude_norm = feature.NumFeature(name='event_longitude_norm', dtype='f')
fea_event_latitude = feature.NumFeature(name='event_latitude', dtype='f')
fea_event_latitude_norm = feature.NumFeature(name='event_latitude_norm', dtype='f')
fea_event_phone_brand = feature.OneHotFeature(name='event_phone_brand', dtype='d')
fea_event_installed_app = feature.MultiFeature(name='event_installed_app', dtype='d')
fea_event_installed_app_norm = feature.MultiFeature(name='event_installed_app_norm', dtype='f')

"""
concat features
"""
fea_concat_1 = feature.MultiFeature(name='concat_1', dtype='f')
fea_concat_1_norm = feature.MultiFeature(name='concat_1_norm', dtype='f')
fea_concat_2 = feature.MultiFeature(name='concat_2', dtype='f')
fea_concat_2_norm = feature.MultiFeature(name='concat_2_norm', dtype='f')
fea_concat_3 = feature.MultiFeature(name='concat_3', dtype='f')
fea_concat_3_norm = feature.MultiFeature(name='concat_3_norm', dtype='f')
fea_concat_4 = feature.MultiFeature(name='concat_4', dtype='f')
fea_concat_4_norm = feature.MultiFeature(name='concat_4_norm', dtype='f')
fea_concat_5 = feature.MultiFeature(name='concat_5', dtype='f')
fea_concat_5_norm = feature.MultiFeature(name='concat_5_norm', dtype='f')
fea_concat_6 = feature.MultiFeature(name='concat_6', dtype='d')
fea_concat_6_norm = feature.MultiFeature(name='concat_6_norm', dtype='f')
fea_concat_7 = feature.MultiFeature(name='concat_7', dtype='d')
fea_concat_7_norm = feature.MultiFeature(name='concat_7_norm', dtype='f')
fea_concat_8 = feature.MultiFeature(name='concat_8', dtype='d')
fea_concat_8_norm = feature.MultiFeature(name='concat_8_norm', dtype='f')
fea_concat_9 = feature.MultiFeature(name='concat_9', dtype='d')
fea_concat_9_norm = feature.MultiFeature(name='concat_9_norm', dtype='f')
fea_ensemble_test = feature.MultiFeature(name='ensmeble_test', dtype='f')


"""
model outputs, for ensemble use
"""
fea_concat_1_gblinear_1 = feature.MultiFeature(name='concat_1_gblinear_1', dtype='f')
fea_concat_1_gbtree_1 = feature.MultiFeature(name='concat_1_gbtree_1', dtype='f')
fea_concat_2_gblinear_1 = feature.MultiFeature(name='concat_2_gblinear_1', dtype='f')
fea_concat_2_gbtree_1 = feature.MultiFeature(name='concat_2_gbtree_1', dtype='f')
fea_concat_2_norm_gblinear_1 = feature.MultiFeature(name='concat_2_norm_gblinear_1', dtype='f')
fea_concat_2_norm_gbtree_1 = feature.MultiFeature(name='concat_2_norm_gbtree_1', dtype='f')
fea_concat_3_gblinear_1 = feature.MultiFeature(name='concat_3_gblinear_1', dtype='f')
fea_concat_3_gbtree_1 = feature.MultiFeature(name='concat_3_gbtree_1', dtype='f')
fea_concat_3_norm_gblinear_1 = feature.MultiFeature(name='concat_3_norm_gblinear_1', dtype='f')
fea_concat_3_norm_gbtree_1 = feature.MultiFeature(name='concat_3_norm_gbtree_1', dtype='f')
fea_concat_4_gblinear_1 = feature.MultiFeature(name='concat_4_gblinear_1', dtype='f')
fea_concat_4_gbtree_1 = feature.MultiFeature(name='concat_4_gbtree_1', dtype='f')
fea_concat_4_norm_gblinear_1 = feature.MultiFeature(name='concat_4_norm_gblinear_1', dtype='f')
fea_concat_4_norm_gbtree_1 = feature.MultiFeature(name='concat_4_norm_gbtree_1', dtype='f')
fea_concat_5_gblinear_1 = feature.MultiFeature(name='concat_5_gblinear_1', dtype='f')
fea_concat_5_gbtree_1 = feature.MultiFeature(name='concat_5_gbtree_1', dtype='f')
fea_concat_5_norm_gblinear_1 = feature.MultiFeature(name='concat_5_norm_gblinear_1', dtype='f')
fea_concat_5_norm_gbtree_1 = feature.MultiFeature(name='concat_5_norm_gbtree_1', dtype='f')
fea_concat_6_gbtree_1 = feature.MultiFeature(name='concat_6_gbtree_1', dtype='f')
fea_concat_6_embedding_64_mlp_for_ensemble = feature.MultiFeature(name='concat_6_embedding_64_mlp_for_ensemble',
                                                                  dtype='f')
fea_concat_6_mlp_136 = feature.MultiFeature(name='concat_6_mlp_136', dtype='f')
fea_concat_6_mlp_for_ensemble = feature.MultiFeature(name='concat_6_mlp_for_ensemble', dtype='f')
fea_concat_6_tfidf_gbtree_1 = feature.MultiFeature(name='concat_6_tfidf_gbtree_1', dtype='f')
fea_concat_7_norm_for_ensemble = feature.MultiFeature(name='concat_7_norm_for_ensemble', dtype='f')
fea_concat_15_mlp_6 = feature.MultiFeature(name='concat_15_mlp_6', dtype='f')
fea_concat_16_mlp_3 = feature.MultiFeature(name='concat_16_mlp_3', dtype='f')
fea_concat_20_mlp_for_ensemble = feature.MultiFeature(name='concat_20_mlp_for_ensemble', dtype='f')
fea_concat_21_mlp_for_ensemble = feature.MultiFeature(name='concat_21_mlp_for_ensemble', dtype='f')
fea_ensemble_5 = feature.MultiFeature(name='ensemble_5', dtype='f')

fea_concat_6_predict_1 = feature.MultiFeature(name='concat_6_predict_1', dtype='f')
fea_concat_6_mlp_143 = feature.MultiFeature(name='concat_6_mlp_143', dtype='f')

fea_concat_1_gbtree_1 = feature.MultiFeature(name='concat_1_gbtree_1', dtype='f')
fea_concat_1_gblinear_1 = feature.MultiFeature(name='concat_1_gblinear_1', dtype='f')
fea_concat_1_mlp_100 = feature.MultiFeature(name='concat_1_mlp_1', dtype='f')


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
    dict_label_category_group = pkl.load(open('../data/dict_label_category_group_number.pkl', 'rb'))
    # dict_label_cluster_40 = pkl.load(open('../data/dict_label_cluster_40.pkl', 'rb'))
    # dict_label_cluster_100 = pkl.load(open('../data/dict_label_cluster_100.pkl', 'rb'))
    # dict_label_cluster_270 = pkl.load(open('../data/dict_label_cluster_270.pkl', 'rb'))

    # for i in range(3):
    #     print dict_device_event[i]
    #     print "\nA new one!\n"
    #
    # exit(0)

    device_id = np.loadtxt('../feature/device_id', dtype=np.int64, skiprows=1, delimiter=',', usecols=[0])

    print 'finish in %d sec' % (time.time() - start_time)

    # event_id = np.loadtxt('../feature/event_id', dtype=np.int64, skiprows=1, usecols=[1], delimiter=',')

    # fea_active_app_label_cluster_40_num.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                             dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                             dict_label_cluster_40=dict_label_cluster_40)
    # fea_active_app_label_cluster_40_num.dump()
    # fea_active_app_num = feature.MultiFeature('active_app_num', dtype='d')
    # fea_active_app_num.process(device_id=device_id, dict_device_event=dict_device_event, dict_app_event=dict_app_event)
    # fea_active_app_num.dump()
    # fea_active_app_label_each_hour_category_freq.process(device_id=device_id, dict_device_event=dict_device_event,
    #                                                      dict_app_event=dict_app_event, dict_app_label=dict_app_label,
    #                                                      dict_label_category_group=dict_label_category_group)
    # fea_active_app_label_each_hour_category_freq.dump()
    fea_active_app_label_diff_hour_category_freq.process(device_id=device_id, dict_device_event=dict_device_event,
                                                         dict_app_event=dict_app_event, dict_app_label=dict_app_label,
                                                         dict_label_category_group=dict_label_category_group)
    fea_active_app_label_diff_hour_category_freq.dump()


def ensemble_concat_feature(name, fea_list):
    spaces = []
    for fea in fea_list:
        spaces.append(fea.get_space())
    print 'spaces', str(spaces)

    fea_concat = feature.MultiFeature(name=name)

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
            tmp_indices.extend(utils.wrap_array(collect_indices[j][i]))
            tmp_values.extend(utils.wrap_array(collect_values[j][i]))
        concat_indices.append(np.array(tmp_indices))
        concat_values.append(tmp_values)

    concat_indices = np.array(concat_indices)
    concat_values = np.array(concat_values)

    fea_concat.set_value(indices=concat_indices, values=concat_values)
    max_indices = map(utils.general_max, concat_indices)
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

    fea_concat = feature.MultiFeature(name=name)

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
            tmp_indices.extend(utils.wrap_array(collect_indices[j][i]))
            tmp_values.extend(utils.wrap_array(collect_values[j][i]))
        concat_indices.append(np.array(tmp_indices))
        concat_values.append(tmp_values)

    concat_indices = np.array(concat_indices)
    concat_values = np.array(concat_values)

    print concat_indices
    print concat_values

    fea_concat.set_value(indices=concat_indices, values=concat_values)
    max_indices = map(utils.general_max, concat_indices)
    len_indices = map(lambda x: len(x), concat_values)
    fea_concat.set_space(max(max_indices) + 1)
    fea_concat.set_rank(max(len_indices))
    fea_concat.set_size(len(concat_indices))

    fea_concat.dump(extra=extra)


def padding_zero(line, space):
    line_space = int(line.strip().split()[-1].split(':')[0]) + 1
    if line_space < space:
        return line.strip() + ' %d:0\n' % (space - 1)
    else:
        return line.strip() + '\n'


def split_dataset(name, cv_rate=0.2, zero_pad=True):
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


def extract_test_preds(name):
    train_size, test_size = get_subset_size()
    test_preds = []
    with open('../feature/' + name, 'r') as data_in:
        next(data_in)
        for i in range(train_size):
            next(data_in)
        for line in data_in:
            test_preds.append(map(lambda x: float(x.split(':')[1]), line.strip().split(' ')))
    test_preds = np.array(test_preds)
    utils.make_submission('../output/' + name + '.submission', test_preds)


def average_submissions(name, path_submissions):
    model_preds = None
    for ps in path_submissions:
        preds_i = np.loadtxt(ps, delimiter=',', dtype=np.float64, skiprows=1, usecols=range(1, 13))
        if model_preds is None:
            model_preds = preds_i
        else:
            model_preds += preds_i
    model_preds /= len(path_submissions)
    utils.make_submission('../output/' + name + '.submission', model_preds)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def feature_tfidf(name):
    fea_tmp = feature.MultiFeature(name=name, dtype='f')
    fea_tmp.load()
    feature_indices, feature_values = fea_tmp.get_value()
    feature_space = fea_tmp.get_space()
    csr_fea = utils.libsvm_2_csr(feature_indices, feature_values, feature_space)
    csr_tfidf = tf_idf.tf_idf(csr_fea)
    name_out = name + '_tfidf'
    utils.csr_2_feature(name_out, csr_tfidf, reorder=True)


def feature_w2v_embedding(fea_raw, model_w2v, order, name):
    raw_indices, raw_values = fea_raw.get_value()
    indices = []
    values = []
    for i in range(len(raw_indices)):
        w2v = np.zeros([order])
        cnt = 0
        for j in range(len(raw_indices[i])):
            try:
                w2v += model_w2v[str(raw_indices[i][j])]
                cnt += 1
            except Exception:
                continue
        if cnt > 0:
            w2v /= cnt
            indices.append(range(order))
            values.append(w2v)
        else:
            indices.append([])
            values.append([])
    indices = np.array(indices)
    values = np.array(values)
    fea_out = feature.MultiFeature(name=name, dtype='f')
    fea_out.set_value(indices=indices, values=values)
    max_indices = map(utils.general_max, indices)
    len_indices = map(lambda x: len(x), values)
    fea_out.set_space(max(max_indices) + 1)
    fea_out.set_rank(max(len_indices))
    fea_out.set_size(len(indices))
    fea_out.dump()


if __name__ == '__main__':
    print 'processing features...'

    # make_feature()
    # feature_tfidf('active_app_label_diff_hour_category_num')
    # feature_tfidf('installed_app')
    #
    split_dataset('concat_100', zero_pad=True)
    split_dataset('concat_1_ensemble_mlp_1024')
    # make_feature()

    # concat_feature('concat_100', [fea_phone_brand, fea_device_model, fea_installed_app, fea_installed_app_label,
    #                               fea_device_long_lat_norm, fea_active_app_freq, fea_active_app_label_freq,
    #                               fea_active_app_label_category, fea_active_app_label_cluster_40,
    #                               fea_active_app_label_diff_hour_category_freq,
    #                               fea_device_day_event_num_freq, fea_device_hour_event_num_freq, fea_device_weekday_event_num_freq])

