import feature
import numpy as np
import matplotlib.pyplot as plt

fea_test = feature.multi_feature(name='installed_app_label_embedding_1',dtype='f')
fea_test.load()
fea_indices, fea_values = fea_test.get_value()
fea_sum = np.sum(fea_values, axis=1)
# fea_values = np.transpose(np.transpose(fea_values) / fea_sum)
# fea_sum2 = np.sum(fea_values, axis=1)
# print fea_sum2
#
# for i in range(fea_values.shape[1]):
#     plt.hist(fea_values[:, i], bins=1000)
#     plt.show()
plt.hist(fea_sum, bins=1000)
plt.show()