# import numpy as np
#
# import feature
#
# fea_installed_app_label = feature.multi_feature('installed_app_label')
# fea_installed_app_label.load()
# indices, values = fea_installed_app_label.get_value()
#
# with open('../data/' + fea_installed_app_label.get_name() + '.corpus', 'w') as fout:
#     for aids in indices:
#         if len(aids) > 0:
#             aids = map(lambda x: str(x), aids)
#             dup = int(np.log(len(aids)) * 8)
#             for i in range(dup):
#                 np.random.shuffle(aids)
#                 fout.write(' '.join(aids) + '\n')


import word2vec

corpus = 'installed_app_label'
for size in [256, 512]:
    word2vec.word2vec(train='../data/' + corpus + '.corpus',
                      output='../data/' + corpus + '.vec.%d' % size,
                      size=size,
                      min_count=0,
                      verbose=True)

# model = word2vec.load('../data/' + corpus + '.vec.8', kind='bin')
# print model.vectors.shape
