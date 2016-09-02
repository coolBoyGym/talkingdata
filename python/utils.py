import cPickle as pkl

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

import feature

DTYPE = tf.float32


def str_2_value(str_value):
    try:
        return int(str_value)
    except ValueError:
        return float(str_value)


def general_max(x):
    tx = type(x).__name__
    if 'set' in tx or 'list' in tx or 'array' in tx:
        if len(x) > 0:
            return max(x)
    else:
        return x


def wrap_array(x):
    tx = type(x).__name__
    if 'set' in tx or 'list' in tx or 'array' in tx:
        return x
    else:
        return [x]


def make_submission(path_submission, test_pred):
    test_device_id = np.loadtxt('../data/raw/gender_age_test.csv', skiprows=1, dtype=np.int64)
    with open(path_submission, 'w') as fout:
        fout.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
        for i in range(len(test_device_id)):
            fout.write('%s,%s\n' % (test_device_id[i], ','.join(map(lambda d: str(d), test_pred[i]))))
    print 'output submission file', path_submission


def make_feature_model_output(name, preds, num_class, dump=True):
    indices = np.zeros_like(preds, dtype=np.int64) + range(num_class)
    fea_pred = feature.MultiFeature(name=name, dtype='f', space=num_class, rank=num_class, size=len(indices))
    fea_pred.set_value(indices, preds)
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
    elif dtype == 'str':
        return 'str' in dtype_name
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


def csr_slice(csr_mats, begin, size):
    begin = min(begin, csr_mats.shape[0])
    if size == -1:
        return csr_mats[begin:]
    end = min(begin + size, csr_mats.shape[0])
    return csr_mats[begin:end]


def csr_2_coo(csr_mats):
    coo_mat = csr_mats.tocoo()
    indices = np.transpose(np.vstack((coo_mat.row, coo_mat.col)))
    values = coo_mat.data
    shape = csr_mats.shape
    return indices, values, shape


def coo_2_csr(coo_indices, coo_values, coo_shapes):
    coo_mat = coo_matrix((coo_values, (coo_indices[:, 0], coo_indices[:, 1])), shape=coo_shapes)
    return coo_mat.tocsr()


def feature_slice_inputs(input_types, csr_data, begin, size):
    if check_type(input_types, 'str'):
        if input_types == 'sparse':
            csr_slc = csr_slice(csr_data, begin, size)
            coo_slc = csr_slc.tocoo()
            indices = np.transpose(np.vstack((coo_slc.row, coo_slc.col)))
            values = coo_slc.data
            shape = csr_slc.shape
            return indices, values, shape
        else:
            if size == -1:
                return csr_data[begin:]
            else:
                return csr_data[begin:begin + size]
    else:
        input_data = []
        for i in range(len(input_types)):
            input_data_i = feature_slice_inputs(input_types[i], csr_data[i], begin, size)
            input_data.append(input_data_i)
        return input_data


def libsvm_2_coo(indices, values, space):
    coo_indices = []
    coo_values = []
    for i in range(len(indices)):
        coo_indices.extend(map(lambda x: [i, x], indices[i]))
        coo_values.extend(values[i])
    csr_shape = [len(indices), space]
    return np.array(coo_indices), np.array(coo_values), csr_shape


def libsvm_2_csr(indices, values, space):
    coo_indices, coo_values, coo_shape = libsvm_2_coo(indices, values, space)
    coo_mat = coo_matrix((coo_values, (coo_indices[:, 0], coo_indices[:, 1])), shape=coo_shape)
    return coo_mat.tocsr()


def libsvm_2_feature(indices, values, spaces, types):
    if check_type(spaces, 'int'):
        if types == 'sparse':
            return libsvm_2_csr(indices, values, spaces)
        elif len(values.shape) == 1:
            return libsvm_2_csr(indices, values, spaces).toarray()
        else:
            return values
    else:
        csr_data = []
        for i in range(len(spaces)):
            csr_data_i = libsvm_2_feature(indices[i], values[i], spaces[i], types[i])
            csr_data.append(csr_data_i)
        return csr_data


def csr_2_libsvm(csr_mat):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    values = csr_mat.data
    libsvm_indices = []
    libsvm_values = []
    for i in range(csr_mat.shape[0]):
        libsvm_indices.append(indices[indptr[i]:indptr[i + 1]])
        libsvm_values.append(values[indptr[i]:indptr[i + 1]])
    return np.array(libsvm_indices), np.array(libsvm_values)


def coo_2_libsvm(coo_indices, coo_values, coo_shape, reorder=False):
    data = np.hstack((coo_indices, np.reshape(coo_values, [-1, 1])))
    if reorder:
        data = sorted(data, key=lambda x: (x[0], x[1]))
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
    while len(indices) < coo_shape[0]:
        indices.append([])
        values.append([])
    return np.array(indices), np.array(values)


def csr_2_feature(name, csr_mat, reorder=False):
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    values = csr_mat.data
    libsvm_indices = []
    libsvm_values = []
    for i in range(csr_mat.shape[0]):
        libsvm_indices.append(indices[indptr[i]:indptr[i + 1]])
        libsvm_values.append(values[indptr[i]:indptr[i + 1]])
    libsvm_indices, libsvm_values = np.array(libsvm_indices), np.array(libsvm_values)
    fea_csr = feature.MultiFeature(name=name, dtype='f')
    fea_csr.set_value(indices=libsvm_indices, values=libsvm_values)
    max_indices = map(general_max, libsvm_indices)
    len_indices = map(lambda x: len(x), libsvm_values)
    fea_csr.set_space(max(max_indices) + 1)
    fea_csr.set_rank(max(len_indices))
    fea_csr.set_size(len(libsvm_indices))
    if reorder:
        fea_csr.reorder()
    fea_csr.dump()


def check_early_stop(valid_scores, early_stop_round, mode, early_stop_precision=0.0001):
    if np.argmin(valid_scores) + early_stop_round > len(valid_scores):
        return False
    minimum = np.min(valid_scores)
    if mode == 'increase' and valid_scores[-1] - minimum > early_stop_precision:
        return True
    elif mode == 'no_decrease' and minimum - valid_scores[-1] < early_stop_precision:
        return True
    return False


def extend_var(x, new_shape):
    if len(x.shape) == 2:
        x = np.vstack((x, np.zeros([new_shape[0] - x.shape[0], x.shape[1]], dtype=np.float32)))
        cols = np.arange(x.shape[1])
        cols += np.arange(x.shape[1]) * x.shape[1]
        indices = np.zeros((x.shape[1] * new_shape[1]), np.float32)
        indices[cols] = 1
        indices = np.reshape(indices, [new_shape[1], x.shape[1]])
        indices = indices.transpose()
        return x.dot(indices)
    else:
        x = np.reshape(x, [1, x.shape[0]])
        extend_x = extend_var(x, [1, new_shape[0]])
        return np.reshape(extend_x, [-1])


def random_repeat_var(x, new_shape, random_seed=None):
    if len(x.shape) == 2:
        if random_seed is not None:
            np.random.seed(random_seed)
        x = np.vstack((x, np.zeros([new_shape[0] - x.shape[0], x.shape[1]], dtype=np.float32)))
        repeat_cols = np.random.randint(0, x.shape[1], [new_shape[1] - x.shape[1]])
        repeat_cols = np.hstack((np.arange(x.shape[1]), repeat_cols))
        repeat_cols += np.arange(new_shape[1]) * x.shape[1]
        indices = np.zeros((x.shape[1] * new_shape[1]), np.float32)
        indices[repeat_cols] = 1
        indices = np.reshape(indices, [new_shape[1], x.shape[1]])
        indices = indices.transpose()
        indices = np.diag(np.power(np.sum(indices, axis=1), -1)).dot(indices)
        return x.dot(indices)
    else:
        x = np.reshape(x, [1, x.shape[0]])
        repeat_x = random_repeat_var(x, [1, new_shape[0]], random_seed)
        return np.reshape(repeat_x, [-1])


def init_var_map(init_actions, init_path=None, random_seed=None):
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
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=0.01, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape, mean=0.0, stddev=0.01, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=-0.01, maxval=0.01, dtype=dtype),
                                            dtype=dtype)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype) * init_method)
        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = tf.Variable(load_var_map[init_method])
            else:
                print 'BadParam: init method', init_method, 'shape', var_shape, load_var_map[init_method].shape
        elif 'res' in init_method:
            res_method = init_method.split(':')[1]
            if res_method != 'pass':
                if res_method in load_var_map:
                    load_var_name = res_method
                else:
                    load_var_name = var_name
                var_load = load_var_map[load_var_name]
                print 'extend', var_load.shape, 'to', var_shape
                # print var_load
                # print var_extend
                var_map[var_name] = tf.Variable(extend_var(var_load, var_shape), dtype=dtype)
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
        elif 'net2' in init_method:
            res_method = init_method.split(':')[1]
            if res_method in load_var_map:
                load_var_name = res_method
            else:
                load_var_name = var_name
            var_load = load_var_map[load_var_name]
            print 'random repeat', var_load.shape, 'to', var_shape
            var_map[var_name] = tf.Variable(random_repeat_var(var_load, var_shape, random_seed), dtype=dtype)
        else:
            print 'BadParam: init method', init_method
    return var_map


def init_input_units(input_spaces, input_types):
    if check_type(input_spaces, 'int'):
        if input_types == 'sparse':
            return tf.sparse_placeholder(DTYPE)
        else:
            return tf.placeholder(DTYPE)
    else:
        input_units = []
        for i in range(len(input_spaces)):
            input_unit_i = init_input_units(input_spaces[i], input_types[i])
            input_units.append(input_unit_i)
        return input_units


def init_feed_dict(input_types, input_data, input_units, feed_dict):
    if check_type(input_types, 'str'):
        if input_types == 'sparse':
            feed_dict[input_units] = tuple(input_data)
        else:
            feed_dict[input_units] = input_data
    else:
        for i in range(len(input_types)):
            init_feed_dict(input_types[i], input_data[i], input_units[i], feed_dict)


def embed_input_units(input_types, input_units, weights, biases):
    if check_type(input_types, 'str'):
        if input_types == 'sparse':
            return tf.sparse_tensor_dense_matmul(input_units, weights) + biases
        else:
            return tf.matmul(input_units, weights) + biases
    else:
        embeds = []
        for i in range(len(input_types)):
            embed_i = embed_input_units(input_types[i], input_units[i], weights[i], biases[i])
            embeds.append(embed_i)
        return embeds


def get_loss(loss_func, y, y_true):
    if loss_func == 'sigmoid_log_loss':
        y_prob = tf.nn.sigmoid(y)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_true))
    elif loss_func == 'softmax_log_loss':
        y_prob = tf.nn.softmax(y)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_true))
    elif loss_func == 'relu_mse':
        y_prob = tf.nn.relu(y)
        loss = tf.nn.l2_loss(y - y_true)
    elif loss_func == 'mse':
        y_prob = y
        loss = tf.nn.l2_loss(y - y_true)
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
        tmp_v = map(lambda x: str_2_value(x.split(':')[1]), fields[1:])
        indices.append(tmp_i)
        values.append(tmp_v)
    indices = np.array(indices)
    values = np.array(values)
    labels = np.array(labels)
    return indices, values, labels


def split_data_by_col(data, spaces, sub_spaces, split_cols):
    if check_type(spaces, 'int'):
        ids_1 = []
        ids_2 = []
        start_index = sum(sub_spaces[:split_cols])
        end_index = sum(sub_spaces[:split_cols + 1])
        for i in range(data.shape[0]):
            if len(data[i, start_index:end_index].data) == 0:
                ids_2.append(i)
            else:
                ids_1.append(i)
        return np.array(ids_1), np.array(ids_2)
    else:
        return split_data_by_col(data[split_cols], spaces[split_cols], [spaces[split_cols]], 0)
