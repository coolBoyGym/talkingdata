from datetime import datetime

import numpy as np


# def num_proc():
#     return [0] * 10, range(10)
#
#
# def one_hot_proc():
#     return range(10), [1] * 10
#
#
# def multi_proc():
#     return [range(x) for x in range(1, 11)], [[1] * x for x in range(1, 11)]
#
#
# def seq_proc():
#     return None, None


def phone_brand_proc(device_id, dict_device_brand_model):
    indices = map(lambda d: dict_device_brand_model[d][0], device_id)
    indices = np.array(indices)
    values = np.ones_like(indices, dtype=np.int64)
    return indices, values


def device_model_proc(device_id, dict_device_brand_model):
    indices = map(lambda d: dict_device_brand_model[d][1], device_id)
    indices = np.array(indices)
    values = np.ones_like(indices, dtype=np.int64)
    return indices, values


def installed_app_proc(device_id, dict_device_event, dict_app_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
            continue

        events = dict_device_event[did]
        tmp = set()
        for e in events:
            eid = e[0]
            if eid in dict_app_event:
                tmp = tmp | dict_app_event[eid][0]
        indices.append(sorted(tmp))
        values.append([1] * len(tmp))

    return np.array(indices), np.array(values)


def active_app_proc(device_id, dict_device_event, dict_app_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
            continue

        events = dict_device_event[did]
        tmp = set()
        for e in events:
            eid = e[0]
            if eid in dict_app_event:
                tmp = tmp | dict_app_event[eid][1]
        indices.append(sorted(tmp))
        values.append([1] * len(tmp))

    return np.array(indices), np.array(values)


def installed_app_norm_proc(indices, values):
    norm_values = []
    for v in values:
        norm_values.append(np.float64(v) / len(v))
    return indices, np.array(norm_values)


def active_app_norm_proc(indices, values):
    norm_values = []
    for v in values:
        norm_values.append(np.float64(v) / len(v))
    return indices, np.array(norm_values)


def get_time(timestamp, fetch_list):
    date_obj = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    res = []
    for f in fetch_list:
        res.append(getattr(date_obj, f))
    return res


def event_time_proc(event_id, dict_event):
    indices = []
    values = []
    for eid in event_id:
        day, hour, minute, second = get_time(dict_event[eid][1], ['day', 'hour', 'minute', 'second'])
        indices.append([day, hour, minute, second])
        values.append([1] * 4)
    indices = np.array(indices, dtype=np.int64)
    values = np.array(values, dtype=np.int64)
    spaces = np.max(indices, axis=0) + 1
    print spaces
    for i in range(4):
        indices[:, i] += sum(spaces[:i])
    return indices, values


def event_longitude_proc(event_id, dict_event):
    values = []
    for eid in event_id:
        values.append(dict_event[eid][2])
    values = np.array(values, dtype=np.float64)
    indices = np.zeros_like(values, dtype=np.int64)
    return indices, values


def event_latitude_proc(event_id, dict_event):
    values = []
    for eid in event_id:
        values.append(dict_event[eid][3])
    values = np.array(values, dtype=np.float64)
    indices = np.zeros_like(values, dtype=np.int64)
    return indices, values


def event_longitude_norm_proc(indices, values):
    values -= np.min(values)
    values /= np.max(values)
    return indices, values


def event_latitude_norm_proc(indices, values):
    values -= np.min(values)
    values /= np.max(values)
    return indices, values


def event_phone_brand_proc(event_id, dict_event, dict_device_brand_model):
    indices = map(lambda x: dict_device_brand_model[dict_event[x][0]][0], event_id)
    indices = np.array(indices)
    values = np.ones_like(indices, dtype=np.int64)
    return indices, values


def event_installed_app_proc(event_id, dict_app_event):
    indices = []
    values = []
    for eid in event_id:
        if eid in dict_app_event:
            tmp = sorted(dict_app_event[eid][0])
            indices.append(tmp)
            values.append([1] * len(tmp))
        else:
            indices.append([])
            values.append([])

    return indices, values


def event_installed_app_norm_proc(indices, values):
    norm_values = []
    for v in values:
        norm_values.append(np.float64(v) / len(v))
    return indices, np.array(norm_values)
