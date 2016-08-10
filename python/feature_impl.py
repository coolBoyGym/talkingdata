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


def installed_app_freq_proc(device_id, dict_device_event, dict_app_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
            continue
        events = dict_device_event[did]
        tmp = {}
        for e in events:
            eid = e[0]
            if eid in dict_app_event:
                for aid in dict_app_event[eid][0]:
                    if aid in tmp:
                        tmp[aid] += 1.0
                    else:
                        tmp[aid] = 1.0
        sorted_aid = sorted(tmp)
        for i in sorted_aid:
            tmp[i] /= len(events)
        indices.append(sorted_aid)
        values.append(map(lambda x: tmp[x], sorted_aid))
    return np.array(indices), np.array(values)


def active_app_freq_proc(device_id, dict_device_event, dict_app_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
            continue
        events = dict_device_event[did]
        tmp = {}
        for e in events:
            eid = e[0]
            if eid in dict_app_event:
                for aid in dict_app_event[eid][1]:
                    if aid in tmp:
                        tmp[aid] += 1.0
                    else:
                        tmp[aid] = 1.0
        sorted_aid = sorted(tmp)
        for i in sorted_aid:
            tmp[i] /= len(events)
        indices.append(sorted_aid)
        values.append(map(lambda x: tmp[x], sorted_aid))
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


def device_event_num_proc(device_id, dict_device_event):
    values = []
    for did in device_id:
        if did not in dict_device_event:
            values.append(0)
        else:
            values.append(len(dict_device_event[did]))
    indices = np.zeros_like(values, dtype=np.int64)
    return indices, values


def device_day_event_num_proc(device_id, dict_device_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
        else:
            days = map(lambda x: get_time(x[1], ['day'])[0], dict_device_event[did])
            tmp = {}
            for d in days:
                if d in tmp:
                    tmp[d] += 1
                else:
                    tmp[d] = 1
            sorted_tmp = sorted(tmp.keys())
            indices.append(sorted_tmp)
            values.append(map(lambda x: tmp[x], sorted_tmp))
    indices = np.array(indices)
    values = np.array(values)
    return indices, values


def device_event_num_norm_proc(indices, values):
    norm_values = np.float64(values) / np.max(values)
    return indices, norm_values


def device_day_event_num_norm_proc(indices, values):
    norm_values = []
    max_num = 0
    for v in values:
        if len(v) > 0:
            max_num = max(max_num, max(v))
    for v in values:
        norm_values.append(np.array(v, dtype=np.float64) / max_num)
    return indices, norm_values


def device_long_lat_proc(device_id, dict_device_event):
    values = []
    for did in device_id:
        if did not in dict_device_event:
            values.append([0] * 10)
        else:
            tmp = dict_device_event[did]
            tmp_long = map(lambda x: x[2], tmp)
            tmp_lat = map(lambda x: x[3], tmp)
            values.append([np.mean(tmp_long), np.std(tmp_long), np.max(tmp_long), np.min(tmp_long), np.median(tmp_long),
                           np.mean(tmp_lat), np.std(tmp_lat), np.max(tmp_lat), np.min(tmp_lat), np.median(tmp_lat), ])
    values = np.array(values)
    indices = np.zeros_like(values, dtype=np.int64)
    indices += range(10)
    return indices, values


def device_long_lat_norm_proc(device_id, dict_device_event):
    max_long = 0
    min_long = 0
    max_lat = 0
    min_lat = 0
    for did in device_id:
        if did in dict_device_event:
            tmp_long = map(lambda x: x[2], dict_device_event[did])
            tmp_lat = map(lambda x: x[3], dict_device_event[did])
            max_long = max(max_long, max(tmp_long))
            min_long = min(min_long, min(tmp_long))
            max_lat = max(max_lat, max(tmp_lat))
            min_lat = min(min_lat, min(tmp_lat))
    values = []
    for did in device_id:
        if did not in dict_device_event:
            values.append([0] * 10)
        else:
            tmp_long = np.array(map(lambda x: x[2], dict_device_event[did]), dtype=np.float64)
            tmp_long -= min_long
            tmp_long /= (max_long - min_long)
            tmp_lat = np.array(map(lambda x: x[3], dict_device_event[did]), dtype=np.float64)
            tmp_lat -= min_lat
            tmp_lat /= (max_lat - min_lat)
            values.append(
                [np.mean(tmp_long), np.std(tmp_long), np.max(tmp_long), np.min(tmp_long), np.median(tmp_long),
                 np.mean(tmp_lat), np.std(tmp_lat), np.max(tmp_lat), np.min(tmp_lat), np.median(tmp_lat)])
    values = np.array(values)
    indices = np.zeros_like(values, dtype=np.int64)
    indices += range(10)
    return indices, values


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
