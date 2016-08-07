# training the whole data set using libsvm tools.
from svmutil import *
import random
from sklearn.metrics import log_loss

train_type = 'libsvm'
version = 2

path_submission = '../output/submission.csv.%s.%d' % (train_type, version)

path_train = '../data/train_with_label_for_svm.csv'
path_test = '../input/test_with_label_for_svm.csv'
path_prediction = '../output/result.csv'

def test():
    labels, datas = sample1()
    m = svm_train(labels, datas, '-t 0 -h 0 -q')
    print "training finished!"
    testLabels, testDatas = sample2()
    p_label, p_acc, p_val = svm_predict(testLabels, testDatas, m)
    # print 'p_label = ', p_label
    print 'p_acc = ', p_acc
    print 'p_val = ', p_val
    with open(path_prediction, 'w') as fout:
        for i in range(len(p_label)):
            fout.write(str(p_label[i]) + '\n')


def sampletest():
    labels, datas = svm_read_problem(path_train)
    with open(path_prediction, 'w') as fout:
        for i in range(len(labels)):
            fout.write(str(datas[i]) + '\n')

def sample1():
    labels, datas = svm_read_problem(path_train)
    l = []
    d = []
    print 'len(labels) = ', len(labels)
    for i in range(20000):
        l.append(labels[i])
        d.append(datas[i])
    return l, d


def sample2():
    labels, datas = svm_read_problem(path_test)
    l = []
    d = []
    for i in range(40):
        l.append(labels[i])
        d.append(datas[i])
    print 'l = ', l
    print 'd = ', d
    return l, d


# feature's index used in libsvm should start with 1, not 0
def deletezerofeatureintrain(name, valid_rate=0.2):
    path_before = '../input/' + name + '.train'
    path_after = '../input/' + name + '.nozerofeature.train'
    with open(path_before) as fin:
        with open(path_after, 'w') as fout:
            for line in fin:
                l = line.split(' ')
                i = 0
                fout.write(l[i] + ' ')
                i += 1
                while i < len(l):
                    s = l[i].split(':')
                    key, value = s[0], s[1]
                    if i == (len(l) - 1):
                        fout.write(str(int(key) + 1) + ":" + value)
                    else:
                        fout.write(str(int(key) + 1) + ":" + value + ' ')
                    i += 1

    with open(path_after) as train_in:
        with open('../input/' + name + '.nozerofeature.train.train', 'w') as train_out:
            with open('../input/' + name + '.nozerofeature.train.valid', 'w') as valid_out:
                for line in train_in:
                    if random.random() > valid_rate:
                        train_out.write(line)
                    else:
                        valid_out.write(line)


def deletezerofeatureintest(name):
    path_before = '../input/' + name + '.test'
    path_after = '../input/' + name + '.nozerofeature.test'
    with open(path_before) as fin:
        with open(path_after, 'w') as fout:
            for line in fin:
                l = line.split(' ')
                i = 0
                fout.write(l[i] + ' ')
                i += 1
                while i < len(l):
                    s = l[i].split(':')
                    key, value = s[0], s[1]
                    if i == (len(l) - 1):
                        fout.write(str(int(key) + 1) + ":" + value)
                    else:
                        fout.write(str(int(key) + 1) + ":" + value + ' ')
                    i += 1


def getsubmission():
    with open('../output/result2') as fin:
        res = []
        next(fin)
        for line in fin:
            s = line.split(' ')
            i = 1
            l = []
            while i <= 12:
                l.append(float(s[i]))
                i += 1
            res.append(l)

    with open(path_submission, 'w') as fout:
        fout.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
        with open('../data/gender_age_test.csv') as fin:
            next(fin)
            cnt = 0
            for line in fin:
                did = line.strip().split(',')[0]
                fout.write(did + ',')
                fout.write(str(res[cnt][10]) + ',')
                fout.write(str(res[cnt][2]) + ',')
                fout.write(str(res[cnt][4]) + ',')
                fout.write(str(res[cnt][11]) + ',')
                fout.write(str(res[cnt][3]) + ',')
                fout.write(str(res[cnt][9]) + ',')
                fout.write(str(res[cnt][8]) + ',')
                fout.write(str(res[cnt][6]) + ',')
                fout.write(str(res[cnt][7]) + ',')
                fout.write(str(res[cnt][1]) + ',')
                fout.write(str(res[cnt][0]) + ',')
                fout.write(str(res[cnt][5]))
                fout.write('\n')
                cnt += 1


# getsubmission()
# test()
# sampletest()
deletezerofeatureintrain('concat_2')
deletezerofeatureintest('concat_2')
