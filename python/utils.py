import cPickle as pkl

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

import feature


def make_submission(path_submission, test_pred):
    test_device_id = np.loadtxt('../data/raw/gender_age_test.csv', skiprows=1, dtype=np.int64)
    with open(path_submission, 'w') as fout:
        fout.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
        for i in range(len(test_device_id)):
            fout.write('%s,%s\n' % (test_device_id[i], ','.join(map(lambda d: str(d), test_pred[i]))))
    print path_submission


def make_feature_model_output(name, train_pred, valid_pred, test_pred, dump=True):
    values = np.vstack((valid_pred, train_pred, test_pred))
    indices = np.zeros_like(values, dtype=np.int64) + range(12)
    fea_pred = feature.multi_feature(name=name, dtype='f', space=12, rank=12, size=len(indices))
    fea_pred.set_value(indices, values)
    if dump:
        fea_pred.dump()
    return fea_pred


def split_feature(indices, values, spaces):
    index_list = map(lambda x: [], range(len(spaces)))
    value_list = map(lambda x: [], range(len(spaces)))
    offsets = [sum(spaces[:i]) for i in range(len(spaces))]
    offsets.append(sum(spaces))
    for i in range(len(indices)):
        num_group = 0
        for j in range(len(indices[i])):
            tmp_index = indices[i][j]
            tmp_value = values[i][j]
            while tmp_index >= offsets[num_group + 1]:
                num_group += 1
            while len(index_list[num_group]) < i + 1:
                index_list[num_group].append([])
                value_list[num_group].append([])
            index_list[num_group][i].append(tmp_index - offsets[num_group])
            value_list[num_group][i].append(tmp_value)
    for i in range(len(spaces)):
        while len(index_list[i]) < len(indices):
            index_list[i].append([])
            value_list[i].append([])
        index_list[i] = np.array(index_list[i])
        value_list[i] = np.array(value_list[i])
    return index_list, value_list


def check_type(data, dtype):
    dtype_name = type(data).__name__
    if dtype == 'int':
        return 'int' in dtype_name
    elif dtype == 'float':
        return 'float' in dtype_name
    elif dtype == 'list':
        return 'list' in dtype_name or 'array' in dtype_name
    elif dtype == 'set':
        return 'set' in dtype_name or 'tuple' in dtype_name
    elif dtype == 'agg':
        return 'set' in dtype_name or 'tuple' in dtype_name or \
               'list' in dtype_name or 'array' in dtype_name


def label_2_group_id(labels, num_class):
    tmp = np.arange(num_class)
    group_ids = labels.dot(tmp)
    return group_ids


def group_id_2_label(group_ids, num_class):
    labels = np.zeros([len(group_ids), num_class])
    for i in range(len(group_ids)):
        labels[i, group_ids[i]] = 1
    return labels


def libsvm_2_csr(indices, values, spaces):
    if check_type(spaces, 'int'):
        csr_indices = []
        csr_values = []
        for i in range(len(indices)):
            csr_indices.extend(map(lambda x: [i, x], indices[i]))
            csr_values.extend(values[i])
        csr_shape = [len(indices), spaces]
        return np.array(csr_indices), np.array(csr_values), csr_shape
    else:
        indices, values = split_feature(indices, values, spaces)
        csr_indices = []
        csr_values = []
        csr_shapes = []
        for i in range(len(indices)):
            indices_i, values_i, shape_i = libsvm_2_csr(indices[i], values[i], spaces[i])
            csr_indices.append(indices_i)
            csr_values.append(values_i)
            csr_shapes.append(shape_i)
        return csr_indices, csr_values, csr_shapes


def libsvm_2_csr_matrix(indices, values, spaces=None):
    csr_indices, csr_values, csr_shape = libsvm_2_csr(indices, values, spaces)
    return csr_matrix((csr_values, (csr_indices[:, 0], csr_indices[:, 1])), shape=csr_shape)


def csr_matrix_2_libsvm(csr_mat):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    values = csr_mat.data
    libsvm_indices = []
    libsvm_values = []
    for i in range(csr_mat.shape[0]):
        libsvm_indices.append(indices[indptr[i]:indptr[i + 1]])
        libsvm_values.append(values[indptr[i]:indptr[i + 1]])
    return np.array(libsvm_indices), np.array(libsvm_values)


def csr_2_libsvm(csr_indices, csr_values, csr_shape, reorder=False):
    data = np.hstack((csr_indices, np.reshape(csr_values, [-1, 1])))
    if reorder:
        data = sorted(data, key=lambda x: (x[0], x[1]))
    print data
    indices = []
    values = []
    for i in range(len(data)):
        r, c, v = data[i]
        if len(indices) <= r:
            while len(indices) <= r:
                indices.append([])
                values.append([])
            indices[r].append(c)
            values[r].append(v)
        elif len(indices) == r + 1:
            indices[r].append(c)
            values[r].append(v)
    while len(indices) < csr_shape[0]:
        indices.append([])
        values.append([])
    return np.array(indices), np.array(values)


def csr_matrix_2_csr(csr_mat):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    values = csr_mat.data
    csr_indices = []
    csr_values = []
    for i in range(csr_mat.shape[0]):
        for j in range(indptr[i], indptr[i + 1]):
            csr_indices.append([i, indices[j]])
            csr_values.append(values[j])
    csr_indices = np.array(csr_indices)
    csr_values = np.array(csr_values)
    return csr_indices, csr_values, csr_mat.shape


# def train_single_round(model, dtrain, batch_size):
#     indices, values, labels = dtrain
#     loss = []
#     y = []
#     y_prob = []
#     input_spaces = model.get_input_spaces()
#     drops = model.get_layer_drops()
#     if batch_size == -1:
#         indices, values, shape = libsvm_2_csr(indices, values, input_spaces)
#         loss, y, y_prob = model.train_batch(indices, values, shape, labels)
#     else:
#         for i in range(len(indices) / batch_size + 1):
#             batch_indices = indices[i * batch_size: (i + 1) * batch_size]
#             batch_values = values[i * batch_size: (i + 1) * batch_size]
#             batch_labels = labels[i * batch_size: (i + 1) * batch_size]
#             batch_indices, batch_values, batch_shape = libsvm_2_csr(batch_indices, batch_values, input_spaces)
#             batch_loss, batch_y, batch_y_prob = model.train_batch(batch_indices, batch_values, batch_shape,
#                                                                   batch_labels, drops)
#             loss.append(batch_loss)
#             y.extend(batch_y)
#             y_prob.extend(batch_y_prob)
#     return np.array(loss), np.array(y), np.array(y_prob)


# def predict(model, data, spaces, drops, batch_size):
#     indices, values = data[:2]
#     y = []
#     y_prob = []
#     if batch_size == -1:
#         indices, values, shape = libsvm_2_csr(indices, values, spaces)
#         y, y_prob = model.predict_batch(indices, values, shape)
#     else:
#         for i in range(len(indices) / batch_size + 1):
#             batch_indices = indices[i * batch_size: (i + 1) * batch_size]
#             batch_values = values[i * batch_size: (i + 1) * batch_size]
#             batch_indices, batch_values, batch_shape = libsvm_2_csr(batch_indices, batch_values, spaces)
#             batch_y, batch_y_prob = model.predict_batch(batch_indices, batch_values, batch_shape, drops)
#             y.extend(batch_y)
#             y_prob.extend(batch_y_prob)
#     return np.array(y), np.array(y_prob)


# def train(model, dtrain, dvalid, input_spaces, num_round, drops, batch_size, verbose, save_log, early_stop_round):
#     train_indices, train_values, train_labels = dtrain
#     valid_indices, valid_values, valid_labels = dvalid
#     train_scores = []
#     valid_scores = []
#     for i in range(num_round):
#         train_loss, train_y, train_y_prob = train_single_round(model, dtrain, input_spaces, drops, batch_size)
#         valid_y, valid_y_prob = predict(model, dvalid[:2], input_spaces, [1] * len(drops), batch_size)
#         train_score = log_loss(train_labels, train_y_prob)
#         valid_score = log_loss(valid_labels, valid_y_prob)
#         if verbose:
#             print '[%d]\tloss: %f \ttrain_score: %f\tvalid_score: %f' % \
#                   (i, train_loss.mean(), train_score, valid_score)
#         if save_log:
#             log_str = '%d\t%f\t%f\t%f\n' % (i, train_loss.mean(), train_score, valid_score)
#             model.write_log(log_str)
#         train_scores.append(train_score)
#         valid_scores.append(valid_score)
#         if check_early_stop(valid_scores, early_stop_round, 'no_decrease'):
#             if verbose:
#                 best_iteration = i + 1 - early_stop_round
#                 print 'best iteration:\n[%d]\ttrain_score: %f\tvalid_score: %f' % (
#                     best_iteration, train_scores[best_iteration], valid_scores[best_iteration])
#             break
#     return train_scores[-1], valid_scores[-1]


def check_early_stop(valid_scores, early_stop_round, mode, early_stop_precision=0.0001):
    if np.argmin(valid_scores) + early_stop_round > len(valid_scores):
        return False
    minimum = np.min(valid_scores)
    if mode == 'increase' and valid_scores[-1] - minimum > early_stop_precision:
        return True
    elif mode == 'no_decrease' and minimum - valid_scores[-1] < early_stop_precision:
        return True
    return False


def init_var_map(init_actions, init_path=None, stddev=0.01, minval=-0.01, maxval=0.01):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print 'load variable map from', init_path, load_var_map.keys()
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_actions:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=stddev, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=minval, maxval=maxval, dtype=dtype),
                                            dtype=dtype)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype) * init_method)
        elif init_method in load_var_map:
            var_map[var_name] = tf.Variable(load_var_map[init_method])
        elif 'res' in init_method:
            res_method = init_method.split(':')[1]
            if res_method in load_var_map:
                var_load = load_var_map[res_method]
                var_extend = np.zeros(var_shape, dtype=np.float32)
                if len(var_load.shape) == 2:
                    var_extend[:var_load.shape[0], :var_load.shape[1]] += var_load
                else:
                    var_extend[:var_load.shape[0]] += var_load
                print 'extend', var_load.shape, 'to', var_extend.shape
                # print var_load
                # print var_extend
                var_map[var_name] = tf.Variable(var_extend, dtype=dtype)
            elif res_method == 'pass':
                if var_shape[0] <= var_shape[1]:
                    var_diag = np.diag(np.ones(var_shape[0], dtype=np.float32), var_shape[0] - var_shape[1])
                    var_diag = var_diag[-1 * var_shape[0]:, :]
                else:
                    var_diag = np.diag(np.ones(var_shape[1], dtype=np.float32), var_shape[0] - var_shape[1])
                    var_diag = var_diag[:, -1 * var_shape[1]:]
                print 'by pass', var_diag.shape
                # print var_diag
                var_map[var_name] = tf.Variable(var_diag, dtype=dtype)
            else:
                print 'BadParam: init method', init_method
        else:
            print 'BadParam: init method', init_method
    return var_map


def get_loss(loss_func, y, y_true):
    if loss_func == 'sigmoid_log_loss':
        y_prob = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_true))
    elif loss_func == 'softmax_log_loss':
        y_prob = tf.nn.softmax(y)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_true))
    else:
        y_prob = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_true))
    return y_prob, loss


def get_optimizer(opt_algo, learning_rate, loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def get_l1_loss(weights):
    return tf.reduce_sum(tf.abs(weights))


def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights


def gather_2d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] + indices[:, 1]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def gather_3d(params, indices):
    shape = tf.shape(params)
    flat = tf.reshape(params, [-1])
    flat_idx = indices[:, 0] * shape[1] * shape[2] + indices[:, 1] * shape[2] + indices[:, 2]
    flat_idx = tf.reshape(flat_idx, [-1])
    return tf.gather(flat, flat_idx)


def max_pool_2d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r1 = tf.tile(r1, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    indices = tf.concat(1, [r1, tf.reshape(indices, [-1, 1])])
    return tf.reshape(gather_2d(params, indices), [-1, k])


def max_pool_3d(params, k):
    _, indices = tf.nn.top_k(params, k, sorted=False)
    shape = tf.shape(indices)
    r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
    r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
    r1 = tf.tile(r1, [1, k * shape[1]])
    r2 = tf.tile(r2, [1, k])
    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.tile(tf.reshape(r2, [-1, 1]), [shape[0], 1])
    indices = tf.concat(1, [r1, r2, tf.reshape(indices, [-1, 1])])
    return tf.reshape(gather_3d(params, indices), [-1, shape[1], k])


def read_buffer(fin, buf_size):
    line_buffer = []
    while True:
        try:
            line_buffer.append(next(fin))
        except StopIteration as e:
            print e
            break
        if len(line_buffer) == buf_size:
            break
    return line_buffer


def read_feature(fin, batch_size, num_class):
    line_buffer = read_buffer(fin, batch_size)
    indices = []
    values = []
    labels = []
    for line in line_buffer:
        fields = line.strip().split()
        tmp_y = [0] * num_class
        tmp_y[int(fields[0])] = 1
        labels.append(tmp_y)
        tmp_i = map(lambda x: int(x.split(':')[0]), fields[1:])
        tmp_v = map(lambda x: feature.str_2_value(x.split(':')[1]), fields[1:])
        indices.append(tmp_i)
        values.append(tmp_v)
    indices = np.array(indices)
    values = np.array(values)
    labels = np.array(labels)
    return indices, values, labels
