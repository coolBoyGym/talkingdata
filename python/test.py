from random import random
from sklearn.cross_validation import train_test_split
import xgboost as xgb

# def split_file():
#     with open('../data/train_brand_model_installed_active.csv') as fin:
#         with open('../data/train.csv', 'w') as fout_train:
#             with open('../data/valid.csv', 'w') as fout_valid:
#                 for line in fin:
#                     if random() < 0.7:
#                         fout_train.write(line)
#                     else:
#                         fout_valid.write(line)


X_train, X_valid = train_test_split()