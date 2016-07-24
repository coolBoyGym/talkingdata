import numpy as np

from feature import *


class installed_app(value_function):
    def __init__(self):
        value_function.__init__(self, 'num', 'list')

    def apply(self, dict_device_event, dict_app_event, device_id):
        res = []
        for did in device_id:
            if did not in dict_device_event:
                res.append([])
                continue

            events = dict_device_event[did]
            tmp = set()
            for e in events:
                eid = e[0]
                if eid in dict_app_event:
                    tmp = tmp | dict_app_event[eid][0]
            res.append(np.array([[x, 1] for x in list(tmp)]))

        return np.array(res)


class active_app(value_function):
    def __init__(self):
        value_function.__init__(self, 'num', 'list')

    def apply(self, dict_device_event, dict_app_event, device_id):
        res = []
        for did in device_id:
            if did not in dict_device_event:
                res.append([])
                continue

            events = dict_device_event[did]
            tmp = set()
            for e in events:
                eid = e[0]
                if eid in dict_app_event:
                    tmp = tmp | dict_app_event[eid][1]
            res.append(np.array([[x, 1] for x in list(tmp)]))

        return np.array(res)
