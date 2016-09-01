import cPickle as pkl
import re
import time
from Queue import PriorityQueue

import numpy as np
from scipy.sparse import csr_matrix

import feature
from tf_idf import tf_idf

data_app_events = '../data/raw/app_events.csv'
data_app_labels = '../data/raw/app_labels.csv'
# ['app_id', 'label_id']
# (459943, 2)
# field app_id unique values: 113211
# field label_id unique values: 507
# app owning labels: max: 25	min: 1	avg: 4.058369
# label owning apps: max: 56902	min: 1	avg: 906.216963

data_events = '../data/raw/events.csv'
data_gender_age_test = '../data/raw/gender_age_test.csv'
data_gender_age_train = '../data/raw/gender_age_train.csv'
data_label_categories = '../data/raw/label_categories.csv'
# label_id unique, category not unique ('unknown', '', etc.)
# (930,)

data_phone_brand_device_model = '../data/raw/phone_brand_device_model.csv'
# 131 brands, 1666 models
# brand owning models: max: 194	min: 1	avg: 12.717557

data_sample_submission = '../data/raw/sample_submission.csv'


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
    # data_app_labels = '../data/raw/app_labels.csv'
    # each app is connected with one set, the items in the set are app labels this app has.
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


def aggregate_label_category():
    # data_label_categories = '../data/raw/label_categories.csv'
    data = np.loadtxt(data_label_categories, delimiter=',', skiprows=1, dtype=[('label_id', int), ('category', 'S100')])
    # print data.shape

    dict_label = pkl.load(open('../data/dict_id_label.pkl'))
    dict_label_category = {}

    for i in range(len(data)):
        label_id, label_category = data[i]
        if label_id in dict_label:
            lid = dict_label[label_id]
            dict_label_category[lid] = change_group_name_2_number(change_category_2_group(label_category))

    pkl.dump(dict_label_category, open('../data/dict_label_category_group_number.pkl', 'wb'))


def change_group_name_2_number(x):
    if x == 'Games':
        return 0
    elif x == 'Property':
        return 1
    elif x == 'Industry tag':
        return 2
    elif x == 'Custom':
        return 3
    elif x == 'Tencent':
        return 4
    elif x == 'Other':
        return 5
    elif x == 'Finance':
        return 6
    elif x == 'Fun':
        return 7
    elif x == 'Services':
        return 8
    elif x == 'Family':
        return 9
    elif x == 'Productivity':
        return 10
    elif x == 'Religion':
        return 11
    elif x == 'Video':
        return 12
    elif x == 'Travel':
        return 13
    elif x == 'Education':
        return 14
    elif x == 'Vitality':
        return 15
    elif x == 'Shopping':
        return 16
    elif x == 'Sports':
        return 17
    elif x == 'Music':
        return 18
    else:
        return 5


def change_category_2_group(x):
    if re.search('([gG]am)|([pP]oker)|([cC]hess)|([pP]uzz)|([bB]all)|([pP]ursu)|([fF]ight)|([sS]imulat)|([sS]hoot)',
                 x) is not None:
        return ('Games')
    # Then I went through existing abbreviations like RPG, MMO and so on
    elif re.search('(RPG)|(SLG)|(RAC)|(MMO)|(MOBA)', x) is not None:
        return ('Games')
    # Still small list of items left which is not covered by regex
    elif x in ['billards', 'World of Warcraft', 'Tower Defense', 'Tomb', 'Ninja', 'Europe and Fantasy', 'Senki',
               'Shushan', 'Lottery ticket', 'majiang', 'tennis', 'Martial arts']:
        return ('Games')
    elif x in ['Property Industry 2.0', 'Property Industry new', 'Property Industry 1.0']:
        return ('Property')
    elif re.search('([eE]state)', x) is not None:
        return ('Property')
    elif re.search(
            '([fF]amili)|([mM]othe)|([fF]athe)|(bab)|([rR]elative)|([pP]regnan)|([pP]arent)|([mM]arriag)|([lL]ove)',
            x) is not None:
        return ('Family')
    elif re.search('([fF]un)|([cC]ool)|([tT]rend)|([cC]omic)|([aA]nima)|([pP]ainti)|\
                 ([fF]iction)|([pP]icture)|(joke)|([hH]oroscope)|([pP]assion)|([sS]tyle)|\
                 ([cC]ozy)|([bB]log)', x) is not None:
        return ('Fun')
    elif x in ['Parkour avoid class', 'community', 'Enthusiasm', 'cosplay', 'IM']:
        return ('Fun')
    elif x == 'Personal Effectiveness 1' or x == 'Personal Effectiveness':
        return ('Productivity')
    elif re.search(
            '([iI]ncome)|([pP]rofitabil)|([lL]iquid)|([rR]isk)|([bB]ank)|([fF]uture)|([fF]und)|([sS]tock)|([sS]hare)',
            x) is not None:
        return ('Finance')
    elif re.search('([fF]inanc)|([pP]ay)|(P2P)|([iI]nsura)|([lL]oan)|([cC]ard)|([mM]etal)|\
                  ([cC]ost)|([wW]ealth)|([bB]roker)|([bB]usiness)|([eE]xchange)', x) is not None:
        return ('Finance')
    elif x in ['High Flow', 'Housekeeping', 'Accounting', 'Debit and credit', 'Recipes', 'Heritage Foundation',
               'IMF', ]:
        return ('Finance')
    elif x == 'And the Church':
        return ('Religion')
    elif re.search('([sS]ervice)', x) is not None:
        return ('Services')
    elif re.search('([aA]viation)|([aA]irlin)|([bB]ooki)|([tT]ravel)|\
                  ([hH]otel)|([tT]rain)|([tT]axi)|([rR]eservati)|([aA]ir)|([aA]irport)', x) is not None:
        return ('Travel')
    elif re.search('([jJ]ourne)|([tT]ransport)|([aA]ccommodat)|([nN]avigat)|([tT]ouris)|([fF]light)|([bB]us)',
                   x) is not None:
        return ('Travel')
    elif x in ['High mobility', 'Destination Region', 'map', 'Weather', 'Rentals']:
        return ('Travel')
    elif re.search('([cC]ustom)', x) is not None:
        return ('Custom')
    elif x in ['video', 'round', 'the film', 'movie']:
        return ('Video')
    elif x in ['Smart Shopping', 'online malls', 'online shopping by group, like groupon', 'takeaway ordering',
               'online shopping, price comparing', 'Buy class', 'Buy', 'shopping sharing',
               'Smart Shopping 1', 'online shopping navigation']:
        return ('Shopping')
    elif re.search('([eE]ducati)|([rR]ead)|([sS]cienc)|([bB]ooks)', x) is not None:
        return ('Education')
    elif x in ['literature', 'Maternal and child population', 'psychology', 'exams', 'millitary and wars', 'news',
               'foreign language', 'magazine and journal', 'dictionary', 'novels', 'art and culture',
               'Entertainment News',
               'College Students', 'math', 'Western Mythology', 'Technology Information', 'study abroad',
               'Chinese Classical Mythology']:
        return ('Education')
    elif x in ['vitality', '1 vitality']:
        return ('Vitality')
    elif x in ['sports and gym', 'Health Management', 'Integrated Living', 'Medical', 'Free exercise', 'A beauty care',
               'fashion', 'fashion outfit', 'lose weight', 'health', 'Skin care applications', 'Wearable Health']:
        return ('Vitality')
    elif x in ['sports', 'Sports News']:
        return ('Sports')
    elif x == 'music':
        return ('Music')
    elif re.search('([hH]otel)', x) is not None:
        return ('Travel')
    elif x in ['1 free', 'The elimination of class', 'unknown', 'free', 'comfortable', 'Cozy 1', 'other',
               'Total Cost 1', 'Classical 1', 'Quality 1', 'classical', 'quality', 'Car Owners', 'Noble 1',
               'Pirated content', 'Securities', 'professional skills', 'Jobs', 'Reputation', 'Simple 1', '1 reputation',
               'Condition of the vehicles', 'magic', 'Internet Securities', 'weibo', 'Housing Advice', 'notes', 'farm',
               'Nature 1', 'Total Cost', 'Sea Amoy', 'show', 'Car', 'pet raising up', 'dotal-lol', 'Express',
               'radio', 'Occupational identity', 'Utilities', 'Trust', 'Contacts', 'Simple', 'Automotive News',
               'Sale of cars', 'File Editor', 'network disk', 'class managemetn', 'management', 'natural',
               'Points Activities', 'Decoration', 'store management', 'Maternal and child supplies', 'Tour around',
               'coupon', 'User Community', 'Vermicelli', 'noble', 'poetry', 'Antique collection', 'Reviews',
               'Scheduling', 'Beauty Nail', 'shows', 'Hardware Related', 'Smart Home', 'Sellers',
               'Desktop Enhancements',
               'library', 'entertainment', 'Calendar', 'Ping', 'System Tools', 'KTV', 'Behalf of the drive',
               'household products', 'Information', 'Man playing favorites', 'App Store', 'Engineering Drawing',
               'Academic Information', 'Appliances', 'Peace - Search', 'Make-up application', 'WIFI', 'phone',
               'Doctors',
               'Smart Appliances', 'reality show', 'Harem', 'trickery', 'Jin Yong', 'effort', 'Xian Xia', 'Romance',
               'tribe', 'email', 'mesasge', 'Editor', 'Clock', 'search', 'Intelligent hardware', 'Browser',
               'Furniture']:
        return ('Other')
    else:
        return x


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


def build_event_dict():
    dict_device = pkl.load(open('../data/dict_id_device.pkl', 'rb'))
    data_e = np.loadtxt(data_events, skiprows=1, delimiter=',',
                        dtype=[('event_id', np.int64), ('device_id', np.int64), ('timestamp', 'S100'),
                               ('longitude', np.float64), ('latitude', np.float64)])

    print data_e
    print data_e.shape

    dict_events = {}
    for eid, device_id, timestamp, longitude, latitude in data_e:
        if device_id not in dict_device:
            print 'not in'
            continue

        did = dict_device[device_id]
        dict_events[eid] = (did, timestamp, longitude, latitude)

    print dict_events.keys()[:10]
    print dict_events.values()[:10]

    pkl.dump(dict_events, open('../data/dict_event.pkl', 'wb'))


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


def count_app_coocur():
    dict_app = pkl.load(open('../data/dict_id_app.pkl', 'rb'))
    dict_app_event = pkl.load(open('../data/dict_app_event.pkl', 'rb'))
    app_coocur = np.zeros([len(dict_app), len(dict_app)], dtype=np.int32)
    for _, aids in dict_app_event.values():
        for ai in aids:
            for aj in aids:
                if ai != aj:
                    app_coocur[ai, aj] += 1
    rows = []
    cols = []
    data = []
    for i in range(len(app_coocur)):
        for j in range(len(app_coocur[i])):
            if app_coocur[i, j] > 0:
                rows.append(i)
                cols.append(j)
                data.append(app_coocur[i, j])
    app_coocur_csr = csr_matrix((data, (rows, cols)))
    pkl.dump(app_coocur_csr, open('../data/app_coocur.pkl', 'wb'))
    app_coocur_tfidf = tf_idf(app_coocur)
    pkl.dump(app_coocur_tfidf, open('../data/app_coocur_tfidf.pkl', 'wb'))


def count_label_coocur():
    dict_label = pkl.load(open('../data/dict_id_label.pkl', 'rb'))
    dict_app_event = pkl.load(open('../data/dict_app_event.pkl', 'rb'))
    dict_app_label = pkl.load(open('../data/dict_app_label.pkl', 'rb'))
    label_coocur = np.zeros([len(dict_label), len(dict_label)], dtype=np.int32)
    print len(dict_app_event)
    cnt = 0
    start_time = time.time()
    for _, aids in dict_app_event.values():
        cnt += 1
        if cnt % 10000 == 0:
            print cnt, time.time() - start_time
            start_time = time.time()
        tmp = set()
        for ai in aids:
            tmp |= dict_app_label[ai]
        for li in tmp:
            for lj in tmp:
                if li != lj:
                    label_coocur[li, lj] += 1
    rows = []
    cols = []
    data = []
    for i in range(len(label_coocur)):
        for j in range(len(label_coocur[i])):
            if label_coocur[i, j] > 0:
                rows.append(i)
                cols.append(j)
                data.append(label_coocur[i, j])
    label_coocur_csr = csr_matrix((data, (rows, cols)))
    pkl.dump(label_coocur_csr, open('../data/label_coocur.pkl', 'wb'))
    label_coocur_tfidf = tf_idf(label_coocur_csr)
    pkl.dump(label_coocur_tfidf, open('../data/label_coocur_tfidf.pkl', 'wb'))


class edge:
    def __init__(self, cost, node1, node2):
        self.cost = cost
        self.node1 = node1
        self.node2 = node2

    def __cmp__(self, other):
        return self.cost < other.cost


def find_parent(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find_parent(parent, parent[i])
    return parent[i]


def join_parent(parent, i, j):
    j_parent = find_parent(parent, j)
    parent[j_parent] = find_parent(parent, i)


def coocur_cluster(path, threshold=0.0):
    coocur_tfidf = pkl.load(open(path, 'rb'))
    coocur_tfidf += coocur_tfidf.transpose()
    coocur_tfidf /= 2
    csr_indices = coocur_tfidf.indices
    csr_indptr = coocur_tfidf.indptr
    csr_data = coocur_tfidf.data
    q = PriorityQueue()
    print 'make heap...'
    parent = np.arange((coocur_tfidf.shape[0]))
    for i in range(coocur_tfidf.shape[0]):
        for ij in range(csr_indptr[i], csr_indptr[i + 1]):
            j = csr_indices[ij]
            d = csr_data[ij]
            if i < j and d > threshold:
                q.put(edge(d, i, j))
    print 'find and join...'
    while not q.empty():
        next_edge = q.get()
        node1 = next_edge.node1
        node2 = next_edge.node2
        n1_parent = find_parent(parent, node1)
        n2_parent = find_parent(parent, node2)
        if n1_parent == n2_parent:
            continue
        elif n2_parent == node2:
            join_parent(parent, node1, node2)
        elif n1_parent == node1:
            join_parent(parent, node2, node1)
        else:
            join_parent(parent, node1, node2)
    clusters = []
    for i in range(coocur_tfidf.shape[0]):
        son_i = np.where(parent == i)[0]
        if len(son_i) > 0:
            clusters.append(son_i)
    clusters = np.array(clusters)
    print len(clusters)
    print clusters
    dict_label_cluster = {}
    cnt = 0
    for c in clusters:
        for l in c:
            dict_label_cluster[l] = cnt
        cnt += 1
    pkl.dump(dict_label_cluster, open('../data/dict_label_cluster_500.pkl', 'wb'))


def aggregate_model_cluster(name):
    fea_tmp = feature.MultiFeature(name=name, dtype='f')
    fea_tmp.load()
    indices, values = fea_tmp.get_value()
    models = np.reshape(indices[:, 0], [-1])
    preds = values[:, 1:]
    model_centers = {}
    for mid in models:
        if mid not in model_centers:
            model_ind = np.where(models == mid)[0]
            model_centers[mid] = np.mean(preds[model_ind], axis=0)
    pkl.dump(model_centers, open('../data/' + name + '.pkl', 'wb'))


if __name__ == '__main__':
    # build_event_dict()
    # aggregate_label_category()
    # count_app_coocur()
    # count_label_coocur()
    # path = '../data/label_coocur_tfidf.pkl'
    # for t in [0.032]:
    #     coocur_cluster(path, t)
    aggregate_model_cluster('model_cluster_1')
