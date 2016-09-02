from datetime import datetime

import numpy as np


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


def active_app_num_proc(device_id, dict_device_event, dict_app_event):
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
                        tmp[aid] += 1
                    else:
                        tmp[aid] = 1
        sorted_aid = sorted(tmp)
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


def installed_app_label_proc(device_id, dict_device_event, dict_app_event, dict_app_label):
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
                for aid in dict_app_event[eid][0]:
                    tmp = tmp | dict_app_label[aid]
        indices.append(sorted(tmp))
        values.append([1] * len(tmp))
    return np.array(indices), np.array(values)


def active_app_label_proc(device_id, dict_device_event, dict_app_event, dict_app_label):
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
                for aid in dict_app_event[eid][1]:
                    tmp = tmp | dict_app_label[aid]
        indices.append(sorted(tmp))
        values.append([1] * len(tmp))
    return np.array(indices), np.array(values)


def installed_app_label_freq_proc(device_id, dict_device_event, dict_app_event, dict_app_label):
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
                    for lid in dict_app_label[aid]:
                        if lid in tmp:
                            tmp[lid] += 1
                        else:
                            tmp[lid] = 1
        sorted_tmp = sorted(tmp.keys())
        total_freq = sum(tmp.values())
        indices.append(sorted_tmp)
        values.append(map(lambda x: tmp[x] * 1.0 / total_freq, sorted_tmp))
    return np.array(indices), np.array(values)


def installed_app_label_num_proc(device_id, dict_device_event, dict_app_event, dict_app_label):
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
                    for lid in dict_app_label[aid]:
                        if lid in tmp:
                            tmp[lid] += 1
                        else:
                            tmp[lid] = 1
        sorted_tmp = sorted(tmp.keys())
        # total_freq = sum(tmp.values())
        indices.append(sorted_tmp)
        values.append(map(lambda x: tmp[x], sorted_tmp))
    return np.array(indices), np.array(values)


def active_app_label_freq_proc(device_id, dict_device_event, dict_app_event, dict_app_label):
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
                    for lid in dict_app_label[aid]:
                        if lid in tmp:
                            tmp[lid] += 1
                        else:
                            tmp[lid] = 1
        sorted_tmp = sorted(tmp.keys())
        total_freq = sum(tmp.values())
        indices.append(sorted_tmp)
        values.append(map(lambda x: tmp[x] * 1.0 / total_freq, sorted_tmp))
    return np.array(indices), np.array(values)


def active_app_label_num_proc(device_id, dict_device_event, dict_app_event, dict_app_label):
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
                    for lid in dict_app_label[aid]:
                        if lid in tmp:
                            tmp[lid] += 1
                        else:
                            tmp[lid] = 1
        sorted_tmp = sorted(tmp.keys())
        # total_freq = sum(tmp.values())
        indices.append(sorted_tmp)
        values.append(map(lambda x: tmp[x], sorted_tmp))
    return np.array(indices), np.array(values)


def active_app_label_cluster_40_proc(device_id, dict_device_event, dict_app_event, dict_app_label,
                                     dict_label_cluster_40):
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
                for aid in dict_app_event[eid][1]:
                    for lid in dict_app_label[aid]:
                        lid_cluster = dict_label_cluster_40[lid]
                        # if lid_group in tmp:
                        #     tmp[lid_group] += 1
                        # else:
                        # tmp[lid_group] = 1
                        tmp.add(lid_cluster)
        # sorted_tmp = sorted(tmp.keys())
        # total_num = sum(tmp.values())
        # sorted_tmp = sorted(tmp.keys())
        indices.append(sorted(tmp))
        values.append([1] * len(tmp))
    return np.array(indices), np.array(values)


def active_app_label_cluster_40_num_proc(device_id, dict_device_event, dict_app_event, dict_app_label,
                                         dict_label_cluster_40):
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
                    for lid in dict_app_label[aid]:
                        lid_cluster = dict_label_cluster_40[lid]
                        if lid_cluster in tmp:
                            tmp[lid_cluster] += 1
                        else:
                            tmp[lid_cluster] = 1
                            # tmp.add(lid_cluster)
        sorted_tmp = sorted(tmp.keys())
        # total_num = sum(tmp.values())
        indices.append(sorted_tmp)
        values.append(map(lambda x: tmp[x], sorted_tmp))
    return np.array(indices), np.array(values)


def active_app_label_cluster_100_proc(device_id, dict_device_event, dict_app_event, dict_app_label,
                                      dict_label_cluster_100):
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
                for aid in dict_app_event[eid][1]:
                    for lid in dict_app_label[aid]:
                        lid_cluster = dict_label_cluster_100[lid]
                        # if lid_group in tmp:
                        #     tmp[lid_group] += 1
                        # else:
                        # tmp[lid_group] = 1
                        tmp.add(lid_cluster)
        # sorted_tmp = sorted(tmp.keys())
        # total_num = sum(tmp.values())
        # sorted_tmp = sorted(tmp.keys())
        indices.append(sorted(tmp))
        values.append([1] * len(tmp))
    return np.array(indices), np.array(values)


def active_app_label_cluster_270_proc(device_id, dict_device_event, dict_app_event, dict_app_label,
                                      dict_label_cluster_270):
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
                for aid in dict_app_event[eid][1]:
                    for lid in dict_app_label[aid]:
                        lid_cluster = dict_label_cluster_270[lid]
                        # if lid_group in tmp:
                        #     tmp[lid_group] += 1
                        # else:
                        # tmp[lid_group] = 1
                        tmp.add(lid_cluster)
        # sorted_tmp = sorted(tmp.keys())
        # total_num = sum(tmp.values())
        # sorted_tmp = sorted(tmp.keys())
        indices.append(sorted(tmp))
        values.append([1] * len(tmp))
    return np.array(indices), np.array(values)


def active_app_label_category_proc(device_id, dict_device_event, dict_app_event, dict_app_label,
                                   dict_label_category_group):
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
                for aid in dict_app_event[eid][1]:
                    for lid in dict_app_label[aid]:
                        lid_group = dict_label_category_group[lid]
                        # if lid_group in tmp:
                        #     tmp[lid_group] += 1
                        # else:
                        # tmp[lid_group] = 1
                        tmp.add(lid_group)
        # sorted_tmp = sorted(tmp.keys())
        # total_num = sum(tmp.values())
        # sorted_tmp = sorted(tmp.keys())
        indices.append(sorted(tmp))
        values.append([1] * len(tmp))
    return np.array(indices), np.array(values)


def active_app_label_category_num_proc(device_id, dict_device_event, dict_app_event, dict_app_label,
                                       dict_label_category_group):
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
                    for lid in dict_app_label[aid]:
                        lid_group = dict_label_category_group[lid]
                        if lid_group in tmp:
                            tmp[lid_group] += 1
                        else:
                            tmp[lid_group] = 1
                            # tmp.add(lid_group)
        sorted_tmp = sorted(tmp.keys())
        indices.append(sorted_tmp)
        values.append(map(lambda x: tmp[x], sorted_tmp))

    return np.array(indices), np.array(values)


def active_app_label_each_hour_category_proc(device_id, dict_device_event, dict_app_event, dict_app_label,
                                             dict_label_category_group):
    # 24 hours * 19 group numbers
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
            # event_hour: the accurate part of hour when this event happen
            event_hour = get_time(e[1], ['hour'])[0]
            if eid in dict_app_event:
                for aid in dict_app_event[eid][1]:
                    for lid in dict_app_label[aid]:
                        lid_group = dict_label_category_group[lid]
                        label_hour_index = event_hour * 19 + lid_group
                        # if label_hour_index in tmp:
                        #     tmp[label_hour_index] += 1
                        # else:
                        #     tmp[label_hour_index] = 1
                        tmp.add(label_hour_index)
        # sorted_tmp = sorted(tmp.keys())
        # total_num = sum(tmp.values())
        # indices.append(sorted_tmp)
        # values.append(map(lambda x: tmp[x] * 1.0 / total_num, sorted_tmp))
        indices.append(sorted(tmp))
        values.append([1] * len(tmp))

    return np.array(indices), np.array(values)


def active_app_label_each_hour_category_num_proc(device_id, dict_device_event, dict_app_event, dict_app_label,
                                                 dict_label_category_group):
    # 24 hours * 19 group numbers
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
            # event_hour: the accurate part of hour when this event happen
            event_hour = get_time(e[1], ['hour'])[0]
            if eid in dict_app_event:
                for aid in dict_app_event[eid][1]:
                    for lid in dict_app_label[aid]:
                        lid_group = dict_label_category_group[lid]
                        label_hour_index = event_hour * 19 + lid_group
                        if label_hour_index in tmp:
                            tmp[label_hour_index] += 1
                        else:
                            tmp[label_hour_index] = 1
                            # tmp.add(label_hour_index)
        sorted_tmp = sorted(tmp.keys())
        # total_num = sum(tmp.values())
        indices.append(sorted_tmp)
        values.append(map(lambda x: tmp[x], sorted_tmp))
        # indices.append(sorted(tmp))
        # values.append([1] * len(tmp))

    return np.array(indices), np.array(values)


def active_app_label_diff_hour_category_proc(device_id, dict_device_event, dict_app_event, dict_app_label,
                                             dict_label_category_group):
    # N hour parts * 19 group numbers
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
            # event_hour: the accurate part of hour when this event happen
            event_hour = change_hour_2_its_group(get_time(e[1], ['hour'])[0])
            if eid in dict_app_event:
                for aid in dict_app_event[eid][1]:
                    for lid in dict_app_label[aid]:
                        lid_group = dict_label_category_group[lid]
                        label_hour_index = event_hour * 19 + lid_group
                        # if label_hour_index in tmp:
                        #     tmp[label_hour_index] += 1
                        # else:
                        #     tmp[label_hour_index] = 1
                        tmp.add(label_hour_index)
        # sorted_tmp = sorted(tmp.keys())
        # total_num = sum(tmp.values())
        # indices.append(sorted_tmp)
        # values.append(map(lambda x: tmp[x] * 1.0 / total_num, sorted_tmp))
        indices.append(sorted(tmp))
        values.append([1] * len(tmp))

    return np.array(indices), np.array(values)


def active_app_label_diff_hour_category_num_proc(device_id, dict_device_event, dict_app_event, dict_app_label,
                                                 dict_label_category_group):
    # N hour parts * 19 group numbers
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
            # event_hour: the accurate part of hour when this event happen
            event_hour = change_hour_2_its_group(get_time(e[1], ['hour'])[0])
            if eid in dict_app_event:
                for aid in dict_app_event[eid][1]:
                    for lid in dict_app_label[aid]:
                        lid_group = dict_label_category_group[lid]
                        label_hour_index = event_hour * 19 + lid_group
                        if label_hour_index in tmp:
                            tmp[label_hour_index] += 1
                        else:
                            tmp[label_hour_index] = 1
        sorted_tmp = sorted(tmp.keys())
        indices.append(sorted_tmp)
        values.append(map(lambda x: tmp[x], sorted_tmp))

    return np.array(indices), np.array(values)


# used in active_app_label_group_diff_hour_freq_proc()
def change_hour_2_its_group(x):
    x = int(x)
    if x < 5:
        return 0
    elif x < 9:
        return 1
    elif x < 14:
        return 2
    elif x < 19:
        return 3
    else:
        return 4


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


def device_day_event_num_freq_proc(device_id, dict_device_event):
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
                    tmp[d] += 1.0
                else:
                    tmp[d] = 1.0
            sum_tmp = sum(tmp.values())
            sorted_tmp = sorted(tmp.keys())
            indices.append(sorted_tmp)
            values.append(map(lambda x: tmp[x] / sum_tmp, sorted_tmp))
    indices = np.array(indices)
    values = np.array(values)
    return indices, values


def device_weekday_event_num_proc(device_id, dict_device_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
        else:
            weekdays = map(lambda x: get_time(x[1], ['weekday'])[0](), dict_device_event[did])
            tmp = [0, 0]
            for d in weekdays:
                di = int(d < 5)
                tmp[di] += 1
            indices.append([0, 1])
            values.append(tmp)
    indices = np.array(indices)
    values = np.array(values)
    return indices, values


def device_weekday_event_num_freq_proc(device_id, dict_device_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
        else:
            weekdays = map(lambda x: get_time(x[1], ['weekday'])[0](), dict_device_event[did])
            tmp = [0.0, 0.0]
            for d in weekdays:
                di = int(d < 5)
                tmp[di] += 1
            tmp_sum = tmp[0] + tmp[1]
            tmp[0] /= tmp_sum
            tmp[1] /= tmp_sum
            indices.append([0, 1])
            values.append(tmp)
    indices = np.array(indices)
    values = np.array(values)
    return indices, values


def device_hour_event_num_proc(device_id, dict_device_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
        else:
            hours = map(lambda x: get_time(x[1], ['hour'])[0], dict_device_event[did])
            tmp = {}
            for d in hours:
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


def device_hour_event_num_freq_proc(device_id, dict_device_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
        else:
            hours = map(lambda x: get_time(x[1], ['hour'])[0], dict_device_event[did])
            tmp = {}
            for d in hours:
                if d in tmp:
                    tmp[d] += 1.0
                else:
                    tmp[d] = 1.0
            sum_tmp = sum(tmp.values())
            sorted_tmp = sorted(tmp.keys())
            indices.append(sorted_tmp)
            values.append(map(lambda x: tmp[x] / sum_tmp, sorted_tmp))
    indices = np.array(indices)
    values = np.array(values)
    return indices, values


def device_day_hour_event_num_proc(device_id, dict_device_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
        else:
            day_hours = map(lambda x: get_time(x[1], ['day', 'hour']), dict_device_event[did])
            day_hours = map(lambda x: x[0] * 24 + x[1], day_hours)
            tmp = {}
            for dh in day_hours:
                if dh in tmp:
                    tmp[dh] += 1
                else:
                    tmp[dh] = 1
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


def device_weekday_event_num_norm_proc(indices, values):
    norm_values = []
    max_num = 0
    for v in values:
        if len(v) > 0:
            max_num = max(max_num, max(v))
    for v in values:
        norm_values.append(np.array(v, dtype=np.float64) / max_num)
    return indices, norm_values


def device_hour_event_num_norm_proc(indices, values):
    norm_values = []
    max_num = 0
    for v in values:
        if len(v) > 0:
            max_num = max(max_num, max(v))
    for v in values:
        norm_values.append(np.array(v, dtype=np.float64) / max_num)
    return indices, norm_values


def device_day_hour_event_num_norm_proc(indices, values):
    norm_values = []
    max_num = 0
    for v in values:
        if len(v) > 0:
            max_num = max(max_num, max(v))
    for v in values:
        norm_values.append(np.array(v, dtype=np.float64) / max_num)
    return indices, norm_values


def device_long_lat_proc(device_id, dict_device_event):
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
        else:
            tmp = dict_device_event[did]
            tmp_long = map(lambda x: x[2], tmp)
            tmp_lat = map(lambda x: x[3], tmp)
            if abs(max(tmp_long)) < 0.01 and abs(min(tmp_long)) < 0.01 and abs(max(tmp_lat)) < 0.01 and \
                            abs(min(tmp_lat)) < 0.01:
                indices.append([])
                values.append([])
            else:
                indices.append(range(10))
                values.append(
                    [np.mean(tmp_long), np.std(tmp_long), np.max(tmp_long), np.min(tmp_long), np.median(tmp_long),
                     np.mean(tmp_lat), np.std(tmp_lat), np.max(tmp_lat), np.min(tmp_lat), np.median(tmp_lat), ])
    indices = np.array(indices)
    values = np.array(values)
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
    indices = []
    values = []
    for did in device_id:
        if did not in dict_device_event:
            indices.append([])
            values.append([])
        else:
            tmp_long = np.array(map(lambda x: x[2], dict_device_event[did]), dtype=np.float64)
            tmp_lat = np.array(map(lambda x: x[3], dict_device_event[did]), dtype=np.float64)
            if abs(max(tmp_long)) < 0.01 and abs(min(tmp_long)) < 0.01 and abs(max(tmp_lat)) < 0.01 and \
                            abs(min(tmp_lat)) < 0.01:
                indices.append([])
                values.append([])
            else:
                tmp_long -= min_long
                tmp_long /= (max_long - min_long)
                tmp_long *= 2
                tmp_long -= 1
                tmp_lat -= min_lat
                tmp_lat /= (max_lat - min_lat)
                tmp_lat *= 2
                tmp_lat -= 1
                indices.append(range(10))
                values.append(
                    [np.mean(tmp_long), np.std(tmp_long), np.max(tmp_long), np.min(tmp_long), np.median(tmp_long),
                     np.mean(tmp_lat), np.std(tmp_lat), np.max(tmp_lat), np.min(tmp_lat), np.median(tmp_lat)])
    indices = np.array(indices)
    values = np.array(values)
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
