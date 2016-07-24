import numpy as np

from feature import *


class installed_app(value_function):
    def __init__(self):
        value_function.__init__(self, 'num', 'list')

    def apply(self, dict_device_event, dict_app_event, device_id):
        res = []
        for did in device_id:
            events = dict_device_event[did]
            tmp = set()
            for e in events:
                eid = e[0]
                if eid in dict_app_event:
                    tmp = tmp | dict_app_event[eid][0]
            tmp = list(tmp)
            res.append(np.vstack((tmp, np.ones((len(tmp)), dtype=np.int64))))

        return res


class active_app(value_function):
    def __init__(self):
        value_function.__init__(self, 'num', 'list')

    def apply(self, dict_device_event, dict_app_event, device_id):
        res = []
        for did in device_id:
            events = dict_device_event[did]
            tmp = set()
            for e in events:
                eid = e[0]
                if eid in dict_app_event:
                    tmp = tmp | dict_app_event[eid][1]
            tmp = list(tmp)
            res.append(np.vstack((tmp, np.ones((len(tmp)), dtype=np.int64))))

        return res


import cPickle as pkl

dict_device_event = pkl.load(open('../data/dict_device_event.pkl', 'rb'))
dict_app_event = pkl.load(open('../data/dict_app_event.pkl', 'rb'))

# device_id = [-6401643145415154744, 1782450055857303792]
device_id = dict_device_event.keys()[:10]

feature_app = seq_feature('installed_app', np.int64, installed_app())
print feature_app.get_value(dict_device_event=dict_device_event, dict_app_event=dict_app_event, device_id=device_id)
feature_app = seq_feature('installed_app', np.int64, active_app())
print feature_app.get_value(dict_device_event=dict_device_event, dict_app_event=dict_app_event, device_id=device_id)
