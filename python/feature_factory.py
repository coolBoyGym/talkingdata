import cPickle as pkl

from feature_impl import *

data_app_events = '../data/app_events.csv'
data_app_labels = '../data/app_labels.csv'
data_events = '../data/events.csv'
data_gender_age_test = '../data/gender_age_test.csv'
data_gender_age_train = '../data/gender_age_train.csv'
data_label_categories = '../data/label_categories.csv'
data_phone_brand_device_model = '../data/phone_brand_device_model.csv'
data_sample_submission = '../data/sample_submission.csv'

dict_device_event = pkl.load(open('../data/dict_device_event.pkl', 'rb'))
dict_app_event = pkl.load(open('../data/dict_app_event.pkl', 'rb'))
dict_app = pkl.load(open('../data/dict_id_app.pkl', 'rb'))
dict_device = pkl.load(open('../data/dict_id_device.pkl', 'rb'))
dict_device_brand_model = pkl.load(open('../data/dict_device_brand_model.pkl', 'rb'))
dict_brand = pkl.load(open('../data/dict_id_brand.pkl', 'rb'))
dict_model = pkl.load(open('../data/dict_id_model.pkl', 'rb'))

groups = ['F23-', 'F24-26', 'F27-28', 'F29-32', 'F33-42', 'F43+', 'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38',
          'M39+']
group_id = {}
for i, v in enumerate(groups):
    group_id[v] = i

# train_data = np.loadtxt(data_gender_age_train, delimiter=',', skiprows=1,
#                         dtype=[('device_id', np.int64), ('gender', 'S10'), ('age', np.int64), ('group', 'S10')])

test_data = np.loadtxt(data_gender_age_test, delimiter=',', skiprows=1, dtype=np.int64)

device_id = np.array(map(lambda d: dict_device[d], test_data))
# gender = np.array(map(lambda d: int(d == 'M'), train_data['gender']))
# age = train_data['age']
# group = np.array(map(lambda d: group_id[d], train_data['group']))

# print train_data
print device_id
# print gender
# print age
# print group

feature_brand = map(lambda d: dict_device_brand_model[d][0], device_id)
feature_model = map(lambda d: dict_device_brand_model[d][1], device_id)

feature_installed_app = multi_feature('installed_app', np.int64, installed_app(), space=len(dict_app)).get_value(
    dict_device_event=dict_device_event, dict_app_event=dict_app_event,
    device_id=device_id)
feature_active_app = multi_feature('installed_app', np.int64, active_app(), space=len(dict_app)).get_value(
    dict_device_event=dict_device_event, dict_app_event=dict_app_event,
    device_id=device_id)

# print feature_installed_app
# print feature_active_app

with open('../data/test_installed_active.csv', 'w') as fout:
    off1 = len(dict_brand)
    off2 = off1 + len(dict_model)
    off3 = off2 + len(dict_app)
    off4 = off3 + len(dict_app)
    for i in range(len(device_id)):
        fout.write('%d:%d %d:%d %s %s\n' % (feature_brand[i], 1, off1 + feature_model[i], 1,
                                            ' '.join([':'.join([str(ind + off2), str(val)]) for ind, val in
                                                      feature_installed_app[i]]),
                                            ' '.join([':'.join([str(ind + off3), str(val)]) for ind, val in
                                                      feature_active_app[i]])))
