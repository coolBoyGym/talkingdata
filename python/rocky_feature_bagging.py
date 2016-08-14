import cPickle as pkl
import time

import numpy as np
from random import random
import feature

# fea_phone_brand = feature.one_hot_feature(name='phone_brand', dtype='d')
# fea_device_model = feature.one_hot_feature(name='device_model', dtype='d')
fea_concat_1_gblinear = feature.multi_feature(name='concat_1_gblinear_1')
fea_concat_1_gbtree = feature.multi_feature(name='concat_1_gbtree_1')

#type 1:delete the whole line of one feature
#type 2:concat features first and then delete some dimensions randomly
#random ratio for deletion: 0.33
def feature_bagging(name,fea_list,type=1):
    extra = ','.join([fea.get_name() for fea in fea_list])
    print 'bagging feature list', extra
    print 'loading features...'

    spaces = []
    for fea in fea_list:
        fea.load()
        spaces.append(fea.get_space())

    print 'spaces', str(spaces)

    fea_bagging = feature.multi_feature(name=name, dtype='f')
    collect_indices = []
    collect_values = []

    for i in range(len(fea_list)):
        fea = fea_list[i]
        concat_indices, concat_values = fea.get_value()
        concat_indices += sum(spaces[:i])
        collect_indices.append(concat_indices)
        collect_values.append(concat_values)

    concat_indices = []
    concat_values = []

    if type == 1:
        for i in range(fea_list[0].get_size()):
            tmp_indices = []
            tmp_values = []
            for j in range(len(fea_list)):
                if random()<0.67:
                    tmp_indices.extend(feature.get_array(collect_indices[j][i]))
                    tmp_values.extend(feature.get_array(collect_values[j][i]))
                else:
                    tmp_indices.extend([])
                    tmp_values.extend([])
            concat_indices.append(np.array(tmp_indices))
            concat_values.append(np.array(tmp_values))
    elif type == 2:
        for i in range(fea_list[0].get_size()):
            tmp_indices = []
            tmp_values = []
            for j in range(len(fea_list)):
                tmp_indices.extend(feature.get_array(collect_indices[j][i]))
                tmp_values.extend(feature.get_array(collect_values[j][i]))

            random_indices = np.random.random([len(tmp_indices)])
            del_row = np.where(random_indices < 0.33)[0]
            tmp_indices = [tmp_indices[x] for x in range(len(tmp_indices)) if x not in del_row]
            tmp_values = [tmp_values[x] for x in range(len(tmp_values)) if x not in del_row]
            concat_indices.append(np.array(tmp_indices))
            concat_values.append(np.array(tmp_values))


    concat_indices = np.array(concat_indices)
    concat_values = np.array(concat_values)

    print concat_indices
    print concat_values

    fea_bagging.set_value(indices=concat_indices, values=concat_values)
    max_indices = map(feature.get_max, concat_indices)
    len_indices = map(lambda x: len(x), concat_values)
    fea_bagging.set_space(max(max_indices) + 1)
    fea_bagging.set_rank(max(len_indices))
    fea_bagging.set_size(len(concat_indices))

    fea_bagging.dump(extra=extra)




feature_bagging('bagging_test', [fea_concat_1_gblinear, fea_concat_1_gbtree], 1)
'''
    bagging_indices = []
    bagging_values = []

    for fea in fea_list:
        fea.load()
        indices, values = fea.get_value()
        random_indices = np.random.random([len(indices)])
        keep_row = np.where(random_indices < 0.67)[0]

        tmp_indices = []
        tmp_values = []
        for i in keep_row:
            tmp_indices.append(indices[i])
            tmp_values.append(values[i])
        bagging_indices.append(np.array(tmp_indices))
        bagging_values.append(np.array(tmp_values))

    bagging_indices = np.array(bagging_indices)
    bagging_values = np.array(bagging_values)

'''
