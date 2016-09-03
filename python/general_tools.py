import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('darkgrid')
colors = sns.color_palette()


def wrap_path_log(path_log):
    if '../model/' not in path_log:
        path_log = '../model/' + path_log
    if '.log' not in path_log:
        path_log += '.log'
    return path_log


def plot_train_valid_score(path_log, loss_col=None, train_col=2, valid_col=3):
    score = np.loadtxt(wrap_path_log(path_log), delimiter='\t', usecols=[1, 2, 3])
    if loss_col is not None:
        plt.plot(range(len(score)), score[:, 0], color=colors[1], label='loss')
    if train_col is not None:
        plt.plot(range(len(score)), score[:, 1], color=colors[2], label='train_score')
    if valid_col is not None:
        plt.plot(range(len(score)), score[:, 2], color=colors[0], label='valid_score')
    plt.legend()
    plt.show()


def plot_multi_score(path_logs, plot_train=False, plot_valid=True):
    for pl in path_logs:
        score = np.loadtxt(pl, delimiter='\t', usecols=[2, 3])
        if plot_train:
            plt.plot(range(len(score)), score[:, 0], label=pl + '.train')
        if plot_valid:
            plt.plot(range(len(score)), score[:, 1], label=pl + '.valid')
    plt.legend()
    plt.show()


def plot_concat_score(path_logs, plot_train=True, plot_valid=True):
    start_point = 0
    for pl in path_logs:
        score = np.loadtxt(wrap_path_log(pl), delimiter='\t', usecols=[2, 3])
        if plot_train:
            plt.plot(np.arange(len(score)) + start_point, score[:, 0], label=pl + '.train')
        if plot_valid:
            plt.plot(np.arange(len(score)) + start_point, score[:, 1], label=pl + '.valid')
        start_point += len(score)
    plt.legend()
    plt.show()


def plot_xgb_train_valid_score(path_log):
    data = np.loadtxt(wrap_path_log(path_log), delimiter='\t', dtype=str, usecols=[1, 2])
    score = np.array(map(lambda x: [float(x[0].split(':')[1]), float(x[1].split(':')[1])], data))
    plt.plot(range(len(score)), score[:, 0], color=colors[2])
    plt.plot(range(len(score)), score[:, 1], color=colors[0])
    plt.show()


def manage_feature_file():
    with open('../feature/device_day_event_num', 'r') as fin:
        with open('../feature/device_day_event_num_new', 'wb') as fout:
            next(fin)
            for line in fin:
                l = line.strip().split(' ')
                flag = (l[0] == '')
                if flag is True:
                    fout.write(line)
                else:
                    for i in range(len(l)):
                        s = l[i].split(':')
                        if int(s[0]) <= 7:
                            fout.write(l[i] + ' ')
                    fout.write('\n')


if __name__ == '__main__':
    # path_log = '../model/concat_7_norm_mlp_34.log'
    # plot_train_valid_score(path_log, train_col=1, valid_col=2)
    # plot_xgb_train_valid_score(path_log)
    path_logs = ['../model/concat_100_net2net_mlp_%d.log' % i for i in [5]]
    plot_concat_score(path_logs)
    # plot_multi_score(path_logs)
    # manage_feature_file()
