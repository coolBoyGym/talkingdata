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
