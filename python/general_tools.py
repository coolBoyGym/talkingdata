# this module file is used to store some useful tools to find best arguments

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def lines3d_demo():
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()
    plt.show()


def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin


def scatter3d_demo():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zl, zh)
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def scatter3d_use_file():
    path_input = '../output/argument.gblinear.out'
    with open(path_input) as fin:
        next(fin)
        x = []
        y = []
        z = []
        for line in fin:
            l = line.split('\t')
            x.append(float(l[0]))
            y.append(float(l[1]))
            z.append(float(l[2]))
    # print z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('alpha Label')
    ax.set_ylabel('lambda Label')
    ax.set_zlabel('value Label')
    plt.show()


def norm_argument_file_get_result(name):
    path_input = '../output/argument.' + name + '.gblinear'
    path_output = '../output/argument.' + name + '.gblinear.out'
    with open(path_input) as fin:
        with open(path_output, 'w') as fout:
            arr = []
            fout.write('alpha\tlambda\tvalue\n')
            for line in fin:
                l = line.split(' ')
                if l[0] == "alpha":
                    l1 = float(l[1])
                elif l[0] == "lambda":
                    l2 = float(l[1])
                    res = float(l[3])
                    # fout.write(str(l1) + '\t' + str(l2) + '\t' + str(res) + '\n')
                    arr.append((l1, l2, res))

            z = []
            for i in range(len(arr)):
                z.append(arr[i][2])
            z.sort()
            for j in z:
                for k in arr:
                    if k[2] == j:
                        fout.write(str(k) + '\n')


def find_best_argument(name):
    path_input = '../output/argument.' + name + '.gblinear.out'
    with open(path_input) as fin:
        next(fin)
        z = []
        s = []
        for line in fin:
            l = line.split('\t')
            z.append(float(l[2]))
            s.append((float(l[0]), float(l[1]), float(l[2])))
    z.sort()
    for j in z:
        for k in s:
            if k[2] == j:
                print k


def wire3d_demo():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()


# scatter3d_use_file()
# wire3d_demo()
norm_argument_file_get_result('concat_3')
# find_best_argument('concat_4')

