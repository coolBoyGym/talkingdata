import numpy as np
from scipy.sparse import csr_matrix
import math
import train_impl as ti
def normalized_by_row_sum(mat):
    major_axis = 1 if len(mat.shape) == 2 else 0
    row_sum = np.array(mat.sum(axis=major_axis).ravel())[0]
    if isinstance(mat, np.ndarray):
        normalized_mat = mat.transpose()
        normalized_mat /= row_sum
        return normalized_mat
    if isinstance(mat, csr_matrix):
        normalized_mat = csr_matrix(mat.shape)
        row_sum = csr_matrix.sum(mat, axis=1).ravel()
        mat.data = np.array(mat.data, dtype='float64')
        normalized_mat.data = mat.data / np.array(row_sum.repeat(np.diff(mat.indptr)))[0]
        normalized_mat.indptr = mat.indptr
        normalized_mat.indices = mat.indices
        return normalized_mat

def tf_idf(count_mat):
    return _tf(count_mat).dot(_idf(count_mat))

def _counting_occurrence(arr):
    arr.sort()
    features = np.unique(arr)
    num_features = len(features)
    diff = np.ones(arr.shape, arr.dtype)
    diff[1:] = np.diff(arr)
    idx = np.where(diff > 0)[0]

    occurrence = np.ones(num_features)
    occurrence[0:num_features - 1] = np.diff(idx)
    occurrence[-1] = arr.shape[0] - np.diff(idx).sum()
    return features, occurrence

def _tf(count_mat):
    return normalized_by_row_sum(count_mat)

def _idf(count_mat):
    total_doc_count = count_mat.shape[0]
    features = np.array(count_mat.indices)
    feature, occurrence = _counting_occurrence(features)
    init_element = [math.log(float(total_doc_count) / occ) for occ in occurrence]
    return csr_matrix((init_element, (feature, feature)), shape=(count_mat.shape[1], count_mat.shape[1]))



if __name__ == '__main__':
    fin = open('../feature/active_app_label_diff_hour_category_num', 'r')
    train_indices, train_values, train_shape, train_labels = ti.read_csr_feature(fin, -1)
    Xtrain = csr_matrix((train_values, (train_indices[:, 0], train_indices[:, 1])), shape=train_shape)



# element = np.array([1., 1., 1., 4., 1., 1.])
# row_index = np.array([0, 1, 1, 1, 2, 2])
# col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
# mat = csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))
# print mat
# tf_mat = tf_idf(mat)
# print
# print tf_mat


