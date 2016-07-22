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


def aggregate_app_label():
    """
    aggregate the tow column data into a dict, key is app_id, value is set of label_id
    """
    with open(data_app_labels) as fin:
        fields = next(fin).strip().split(',')
        print fields

    data = np.loadtxt(data_app_labels, delimiter=',', skiprows=1, dtype=np.int64)
    print data.shape

    for i in range(data.shape[1]):
        print 'field %s unique values: %d' % (fields[i], len(np.unique(data[:, i])))

    dict1 = {}
    dict2 = {}

    for i in range(len(data)):
        app_id, label_id = data[i]
        if app_id in dict1:
            dict1[app_id].add(label_id)
        else:
            dict1[app_id] = {label_id}

        if label_id in dict2:
            dict2[label_id].add(app_id)
        else:
            dict2[label_id] = {app_id}

    pkl.dump(dict1, open('../data/app_label_dict1.pkl', 'wb'))
    pkl.dump(dict2, open('../data/app_label_dict2.pkl', 'wb'))


def stat_app_label():
    dict1 = pkl.load(open('../data/app_label_dict1.pkl', 'rb'))
    counter = np.array([[k, len(v)] for k, v in dict1.iteritems()])

    print 'app owning labels: max: %d\tmin: %d\tavg: %f' % (
        counter[:, 1].max(), counter[:, 1].min(), counter[:, 1].mean())

    dict2 = pkl.load(open('../data/app_label_dict2.pkl', 'rb'))
    counter = np.array([[k, len(v)] for k, v in dict2.iteritems()])

    print 'label owning apps: max: %d\tmin: %d\tavg: %f' % (
        counter[:, 1].max(), counter[:, 1].min(), counter[:, 1].mean())


def stat_label_category():
    data = np.loadtxt(data_label_categories, delimiter=',', skiprows=1,
                      dtype=[('label_id', np.int64), ('category', 'S100')])

    print data.shape


def encode(d):
    d[1] = d[1].encode('utf-8')
    return d


def make_brand_model_id():
    with open(data_phone_brand_device_model, 'r') as fin:
        next(fin)

        cnt = 0
        brands = {}
        models = {}
        for line in fin:
            line = line.decode('utf-8')
            _, b, m = line.strip().split(',')

            cnt += 1
            m = '-'.join([b, m])

            if b not in brands.keys():
                brands[b] = len(brands)

            if m not in models.keys():
                models[m] = [len(models), brands[b]]

    brands = [[v, k] for k, v in brands.iteritems()]
    brands = sorted(brands, key=lambda d: d[0])
    brands = map(encode, brands)
    with open('../data/brands.csv', 'w') as fout:
        fout.write('brand_id,brand_name\n')
        for brand_id, brand_name in brands:
            fout.write('%d,%s\n' % (brand_id, brand_name))

    models = [[v[0], k, v[1]] for k, v in models.iteritems()]
    models = sorted(models, key=lambda d: d[0])
    models = map(encode, models)
    with open('../data/models.csv', 'w') as fout:
        fout.write('model_id,model_name,brand_id\n')
        for model_id, model_name, brand_id in models:
            fout.write('%d,%s,%d\n' % (model_id, model_name, brand_id))


def aggregate_brand():
    data = np.loadtxt('../data/models.csv', dtype=np.int64, delimiter=',', skiprows=1, usecols=[0, 2])

    dict = {}
    for m, b in data:
        if b in dict:
            dict[b].add(m)
        else:
            dict[b] = {m}

    counter = np.array([[k, len(v)] for k, v in dict.iteritems()])
    print 'label owning apps: max: %d\tmin: %d\tavg: %f' % (
        counter[:, 1].max(), counter[:, 1].min(), counter[:, 1].mean())

    pkl.dump(dict, open('../data/brand_model_dict.pkl', 'wb'))


if __name__ == '__main__':
    aggregate_brand()
