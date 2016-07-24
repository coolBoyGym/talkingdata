import cPickle as pkl

import numpy as np
import seaborn as sns

sns.set_style('darkgrid')

data_app_events = '../data/app_events.csv'
data_app_labels = '../data/app_labels.csv'
# ['app_id', 'label_id']
# (459943, 2)
# field app_id unique values: 113211
# field label_id unique values: 507
# app owning labels: max: 25	min: 1	avg: 4.058369
# label owning apps: max: 56902	min: 1	avg: 906.216963

data_events = '../data/events.csv'
data_gender_age_test = '../data/gender_age_test.csv'
data_gender_age_train = '../data/gender_age_train.csv'
data_label_categories = '../data/label_categories.csv'
# label_id unique, category not unique ('unknown', '', etc.)
# (930,)

data_phone_brand_device_model = '../data/phone_brand_device_model.csv'
# 131 brands, 1666 models
# brand owning models: max: 194	min: 1	avg: 12.717557

data_sample_submission = '../data/sample_submission.csv'


# label stat
# label id in app_label: %d 507 label id in label_category: %d 930
# app & category 507
# app < category True

# app in app_label 113211 # app in app_event 19237
# event <= label True
# only build index for app having events

# device stat
# device_id: phone 186716 event 60865 train 74645 test 112071
# has device info. train & phone 74645 test & phone 112071
# has event info. train & event 23309 test & event 35194
# no device info. train - phone 0 test - phone 0
# no event info. train - event 51336 test - event 76877
# train & test 0
# phone & event 58503
# has device and event train & (phone & event) 23309 test & (phone & event) 35194
# phone | event 189078
# no device and no event train - (phone | event) 0 test - (phone | event) 0
# phone - event 128213
# has device no event train & (phone - event) 51336 test & (phone - event) 76877
# event - phone 2362
# has event no device train & (event - phone) 0 test & (event - phone) 0
# phone == (train | test) True

# event stat
# event in events 3252950 # event in app_events 1488096
# app <= events True
# is_installed always True
# max: 33426, min: 1, mean: 54.021451891356001

def make_label_id():
    label_a = set(np.loadtxt(data_app_labels, skiprows=1, delimiter=',', usecols=[1], dtype=np.int64))
    label_c = set(np.loadtxt(data_label_categories, skiprows=1, delimiter=',', usecols=[0], dtype=np.int64))

    print 'label id in app_label: %d', len(label_a), 'label id in label_category: %d', len(label_c)
    print 'app & category', len(label_a & label_c)

    print 'app < category', label_a <= label_c

    cnt = 0
    dict = {}
    for l in label_a:
        dict[l] = cnt
        cnt += 1

    pkl.dump(dict, open('../data/dict_id_label.pkl', 'wb'))

    with open('../data/id_label.csv', 'wb') as fout:
        fout.write('lid,label_id\n')

        for k, v in dict.iteritems():
            fout.write('%d,%d\n' % (v, k))


def make_app_id():
    app_l = set(np.loadtxt(data_app_labels, skiprows=1, delimiter=',', usecols=[0], dtype=np.int64))
    app_e = set(np.loadtxt(data_app_events, skiprows=1, delimiter=',', usecols=[1], dtype=np.int64))

    print '# app in app_label', len(app_l), '# app in app_event', len(app_e)
    print 'event <= label', app_e <= app_l
    print 'only build index for app having events'

    cnt = 0
    dict = {}
    for a in app_e:
        dict[a] = cnt
        cnt += 1

    pkl.dump(dict, open('../data/dict_id_app.pkl', 'wb'))

    with open('../data/id_app.csv', 'wb') as fout:
        fout.write('aid,app_id\n')

        for k, v in dict.iteritems():
            fout.write('%d,%d\n' % (v, k))


def aggregate_app_label():
    data = np.loadtxt(data_app_labels, delimiter=',', skiprows=1, dtype=np.int64)
    print data.shape

    dict_app = pkl.load(open('../data/dict_id_app.pkl'))
    dict_label = pkl.load(open('../data/dict_id_label.pkl'))

    dict_app_label = {}
    for i in range(len(data)):
        app_id, label_id = data[i]
        if app_id in dict_app:
            aid = dict_app[app_id]
            lid = dict_label[label_id]

            if aid in dict_app_label:
                dict_app_label[aid].add(lid)
            else:
                dict_app_label[aid] = {lid}

    pkl.dump(dict_app_label, open('../data/dict_app_label.pkl', 'wb'))


# def stat_app_label():
#     dict1 = pkl.load(open('../data/app_label_dict1.pkl', 'rb'))
#     counter = np.array([[k, len(v)] for k, v in dict1.iteritems()])
#
#     print 'app owning labels: max: %d\tmin: %d\tavg: %f' % (
#         counter[:, 1].max(), counter[:, 1].min(), counter[:, 1].mean())
#
#     dict2 = pkl.load(open('../data/app_label_dict2.pkl', 'rb'))
#     counter = np.array([[k, len(v)] for k, v in dict2.iteritems()])
#
#     print 'label owning apps: max: %d\tmin: %d\tavg: %f' % (
#         counter[:, 1].max(), counter[:, 1].min(), counter[:, 1].mean())


# def stat_label_category():
#     data = np.loadtxt(data_label_categories, delimiter=',', skiprows=1,
#                       dtype=[('label_id', np.int64), ('category', 'S100')])
#
#     print data.shape


def encode(d):
    d[1] = d[1].encode('utf-8')
    return d


def make_brand_model_id():
    with open(data_phone_brand_device_model, 'r') as fin:
        next(fin)

        brands = {}
        models = {}
        for line in fin:
            line = line.decode('utf-8')
            _, b, m = line.strip().split(',')

            m = '-'.join([b, m])

            if b not in brands.keys():
                brands[b] = len(brands)

            if m not in models.keys():
                models[m] = len(models)

    pkl.dump(brands, open('../data/dict_id_brand.pkl', 'wb'))
    pkl.dump(models, open('../data/dict_id_model.pkl', 'wb'))

    brands = [[v, k] for k, v in brands.iteritems()]
    brands = sorted(brands, key=lambda d: d[0])
    brands = map(encode, brands)
    with open('../data/id_brand.csv', 'w') as fout:
        fout.write('brand_id,brand_name\n')
        for brand_id, brand_name in brands:
            fout.write('%d,%s\n' % (brand_id, brand_name))

    models = [[v, k] for k, v in models.iteritems()]
    models = sorted(models, key=lambda d: d[0])
    models = map(encode, models)
    with open('../data/id_model.csv', 'w') as fout:
        fout.write('model_id,model_name,brand_id\n')
        for model_id, model_name in models:
            fout.write('%d,%s\n' % (model_id, model_name))


def aggregate_brand():
    data = np.loadtxt('../data/id_model.csv', dtype=np.int64, delimiter=',', skiprows=1, usecols=[0, 2])

    dict = {}
    for m, b in data:
        if b in dict:
            dict[b].add(m)
        else:
            dict[b] = {m}

    counter = np.array([[k, len(v)] for k, v in dict.iteritems()])
    print 'label owning apps: max: %d\tmin: %d\tavg: %f' % (
        counter[:, 1].max(), counter[:, 1].min(), counter[:, 1].mean())

    pkl.dump(dict, open('../data/dict_brand_model.pkl', 'wb'))


def make_device_id():
    phone = set(np.loadtxt(data_phone_brand_device_model, skiprows=1, delimiter=',', dtype=np.int64, usecols=[0]))
    event = set(np.loadtxt(data_events, skiprows=1, delimiter=',', dtype=np.int64, usecols=[1]))
    train = set(np.loadtxt(data_gender_age_train, skiprows=1, delimiter=',', dtype=np.int64, usecols=[0]))
    test = set(np.loadtxt(data_gender_age_test, skiprows=1, delimiter=',', dtype=np.int64, usecols=[0]))

    print '# device_id: phone', len(phone), 'event', len(event), 'train', len(train), 'test', len(test)
    print 'has device info.', 'train & phone', len(train & phone), 'test & phone', len(test & phone)
    print 'has event info.', 'train & event', len(train & event), 'test & event', len(test & event)
    print 'no device info.', 'train - phone', len(train - phone), 'test - phone', len(test - phone)
    print 'no event info.', 'train - event', len(train - event), 'test - event', len(test - event)
    print 'train & test', len(train & test)
    print 'phone & event', len(phone & event)
    print 'has device and event', 'train & (phone & event)', len(
        train & (phone & event)), 'test & (phone & event)', len(test & (phone & event))
    print 'phone | event', len(phone | event)
    print 'no device and no event', 'train - (phone | event)', len(
        train - (phone | event)), 'test - (phone | event)', len(test - (phone | event))
    print 'phone - event', len(phone - event)
    print 'has device no event', 'train & (phone - event)', len(train & (phone - event)), 'test & (phone - event)', len(
        test & (phone - event))
    print 'event - phone', len(event - phone)
    print 'has event no device', 'train & (event - phone)', len(train & (event - phone)), 'test & (event - phone)', len(
        test & (event - phone))

    print 'phone == (train | test)', phone == (train | test)

    with open('../data/id_device.csv', 'w') as fout:
        fout.write('did,device_id\n')

        cnt = 0
        for d in phone:
            fout.write('%d,%d\n' % (cnt, d))
            cnt += 1

    data = np.loadtxt('../data/id_device.csv', skiprows=1, delimiter=',', dtype=np.int64)
    dict = {}

    for k, v in data:
        dict[v] = k

    pkl.dump(dict, open('../data/dict_id_device.pkl', 'wb'))


def build_index_brand_model():
    dict_device = pkl.load(open('../data/dict_id_device.pkl', 'rb'))
    dict_brand = pkl.load(open('../data/dict_id_brand.pkl', 'rb'))
    dict_model = pkl.load(open('../data/dict_id_model.pkl', 'rb'))

    data = []
    with open(data_phone_brand_device_model, 'r') as fin:
        next(fin)

        for line in fin:
            line = line.decode('utf-8')
            d, b, m = line.strip().split(',')
            m = '-'.join([b, m])

            d = int(d)
            did = dict_device[d]
            bid = dict_brand[b]
            mid = dict_model[m]

            data.append([did, bid, mid])

    data = np.array(data)
    print data, data.shape

    dict_device_brand_model = {}
    dict_brand_device = {}
    dict_model_device = {}

    for did, bid, mid in data:
        if did in dict_device_brand_model:
            if (bid, mid) != dict_device_brand_model[did]:
                print 'corruption'
                print 'device already exists', did, dict_device_brand_model[did]
                print 'current value', did, bid, mid

        dict_device_brand_model[did] = (bid, mid)

        if bid in dict_brand_device:
            dict_brand_device[bid].add(did)
        else:
            dict_brand_device[bid] = {did}

        if mid in dict_model_device:
            dict_model_device[mid].add(did)
        else:
            dict_model_device[mid] = {did}

    pkl.dump(dict_device_brand_model, open('../data/dict_device_brand_model.pkl', 'wb'))
    pkl.dump(dict_brand_device, open('../data/dict_brand_device.pkl', 'wb'))
    pkl.dump(dict_model_device, open('../data/dict_model_device.pkl', 'wb'))


def make_event_id():
    event_e = set(np.loadtxt(data_events, skiprows=1, delimiter=',', dtype=np.int64, usecols=[0]))
    event_a = set(np.loadtxt(data_app_events, skiprows=1, delimiter=',', dtype=np.int64, usecols=[0]))
    print '# event in events', len(event_e), '# event in app_events', len(event_a)
    print 'app <= events', event_a <= event_e


def aggregate_device_event():
    dict_device = pkl.load(open('../data/dict_id_device.pkl', 'rb'))
    data_e = np.loadtxt(data_events, skiprows=1, delimiter=',',
                        dtype=[('event_id', np.int64), ('device_id', np.int64), ('timestamp', 'S100'),
                               ('longitude', np.float64), ('latitude', np.float64)])

    print data_e
    print data_e.shape

    dict_device_event = {}
    no_id_device = set()
    for eid, device_id, timestamp, longitude, latitude in data_e:
        # print eid, device_id, timestamp, longitude, latitude
        if device_id not in dict_device:
            no_id_device.add(device_id)
            continue

        did = dict_device[device_id]
        if did in dict_device_event:
            dict_device_event[did].append((eid, timestamp, longitude, latitude))
        else:
            dict_device_event[did] = [(eid, timestamp, longitude, latitude)]

    print 'device id not in device_dict', len(no_id_device)
    for k in dict_device_event.keys():
        dict_device_event[k] = sorted(dict_device_event[k], key=lambda d: d[1])

    print dict_device_event.keys()[:10]
    print dict_device_event.values()[:10]

    pkl.dump(dict_device_event, open('../data/dict_device_event.pkl', 'wb'))


def aggregate_app_event():
    dict_app = pkl.load(open('../data/dict_id_app.pkl', 'rb'))
    data_a = np.loadtxt(data_app_events, skiprows=1, delimiter=',', dtype=np.int64)

    print data_a
    print data_a.shape

    dict_app_event = {}
    for eid, app_id, is_installed, is_active in data_a:
        aid = dict_app[app_id]

        if eid in dict_app_event:
            if is_installed:
                dict_app_event[eid][0].add(aid)
            if is_active:
                dict_app_event[eid][1].add(aid)
        else:
            if is_installed:
                dict_app_event[eid] = ({aid}, set())
            if is_active:
                dict_app_event[eid] = (set(), {aid})

    print dict_app_event.keys()[:10]
    print dict_app_event.values()[:10]

    pkl.dump(dict_app_event, open('../data/dict_app_event.pkl', 'wb'))


if __name__ == '__main__':
    aggregate_app_event()
