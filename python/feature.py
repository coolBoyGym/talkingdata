import numpy as np

import feature_impl


def get_map(dtype):
    if 'd' in dtype:
        return np.int64
    elif 'f' in dtype:
        return np.float64


def get_max(x):
    tx = type(x).__name__
    if 'set' in tx or 'list' in tx or 'array' in tx:
        if len(x) > 0:
            return max(x)
    else:
        return x


def get_array(x):
    tx = type(x).__name__
    if 'set' in tx or 'list' in tx or 'array' in tx:
        return x
    else:
        return [x]


class feature:
    def __init__(self, name=None, ftype=None, dtype=None, space=None, rank=None, size=None):
        """
        :param name: identifier, string
        :param ftype: feature type, string, could be num | one_hot | multi | seq
        :param dtype: data type, a format string
        :param proc: feature map, func in module feature_impl
        :param space: dimension of feature space, int | None
        :param rank: num of non-zero values, int | None
        """
        self.__name = name
        self.__ftype = ftype
        self.__dtype = dtype
        if name is not None and hasattr(feature_impl, name + '_proc'):
            self.__proc = getattr(feature_impl, name + '_proc')
        else:
            self.__proc = None
        self.__space = space
        self.__rank = rank
        self.__size = size
        self.__indices = None
        self.__values = None

    def get_name(self):
        return self.__name

    def get_feature_type(self):
        return self.__ftype

    def get_data_type(self):
        return self.__dtype

    def set_space(self, space):
        self.__space = space

    def get_space(self):
        return self.__space

    def set_rank(self, rank):
        self.__rank = rank

    def get_rank(self):
        return self.__rank

    def set_size(self, size):
        self.__size = size

    def get_size(self):
        return self.__size

    def process(self, **argv):
        self.__indices, self.__values = self.__proc(**argv)

    def set_value(self, indices=None, values=None):
        if indices is not None:
            self.__indices = indices
        if values is not None:
            self.__values = values

    def get_value(self):
        return self.__indices, self.__values

    def dump(self, extra=None):
        assert self.__indices is not None and self.__values is not None
        print 'feature dumped at: %s' % ('../feature/' + self.__name)
        with open('../feature/' + self.__name, 'w') as fout:
            header = '%s %s %s %s %s' % (
                self.__ftype, self.__dtype, str(self.__space), str(self.__rank), str(self.__size))
            if extra is not None:
                header += ' %s' % extra
            print 'header: %s' % header
            fout.write(header + '\n')

    def load(self):
        with open('../feature/' + self.__name, 'r') as fin:
            ftype, dtype, space, rank, size = fin.readline().strip().split()[:5]
            if space == 'None':
                space = None
            else:
                space = int(space)
            if rank == 'None':
                rank = None
            else:
                rank = int(rank)
            if size == 'None':
                size = None
            else:
                size = int(size)

            self.__init__(self.__name, ftype, dtype, space, rank, size)


class num_feature(feature):
    def __init__(self, name=None, ftype='num', dtype=None, space=1, rank=1, size=None):
        feature.__init__(self, name, ftype, dtype, space, rank, size)

    def load(self):
        feature.load(self)
        with open('../feature/' + self.get_name(), 'r') as fin:
            next(fin)
            data = [line.strip() for line in fin]
            values = map(get_map(self.get_data_type()), data)

            values = np.array(values)
            indices = np.zeros_like(values, dtype=np.int64)

            self.set_value(indices=indices, values=values)

    def dump(self, extra=None):
        feature.dump(self, extra)
        with open('../feature/' + self.get_name(), 'a') as fout:
            _, values = self.get_value()
            fmt = '%' + self.get_data_type() + '\n'
            for i in range(len(values)):
                fout.write(fmt % values[i])

    def process(self, **argv):
        feature.process(self, **argv)
        self.set_space(1)
        self.set_rank(1)
        self.set_size(len(self.get_value()[0]))


class one_hot_feature(feature):
    def __init__(self, name=None, ftype='one_hot', dtype=None, space=None, rank=1, size=None):
        feature.__init__(self, name, ftype, dtype, space, rank, size)

    def load(self):
        feature.load(self)
        with open('../feature/' + self.get_name(), 'r') as fin:
            next(fin)
            data = [line.strip().split(':') for line in fin]
            indices = map(get_map('d'), map(lambda x: x[0], data))
            values = map(get_map(self.get_data_type()), map(lambda x: x[1], data))
            indices = np.array(indices)
            values = np.array(values)

            self.set_value(indices=indices, values=values)

    def dump(self, extra=None):
        feature.dump(self, extra)
        with open('../feature/' + self.get_name(), 'a') as fout:
            indices, values = self.get_value()
            fmt = '%d:%' + self.get_data_type() + '\n'
            for i in range(len(indices)):
                fout.write(fmt % (indices[i], values[i]))

    def process(self, **argv):
        feature.process(self, **argv)
        indices, _ = self.get_value()
        self.set_space(np.max(indices) + 1)
        self.set_rank(1)
        self.set_size(len(indices))


class multi_feature(feature):
    def __init__(self, name=None, ftype='multi', dtype=None, space=None, rank=None, size=None):
        feature.__init__(self, name, ftype, dtype, space, rank, size)

    def load(self):
        feature.load(self)
        with open('../feature/' + self.get_name(), 'r') as fin:
            next(fin)
            indices = []
            values = []
            for line in fin:
                entry = map(lambda x: x.split(':'), line.strip().split())
                indices.append(map(lambda x: x[0], entry))
                values.append(map(lambda x: x[1], entry))
            indices = map(get_map('d'), indices)
            values = map(get_map(self.get_data_type()), values)
            indices = np.array(indices)
            values = np.array(values)

            self.set_value(indices=indices, values=values)

    def dump(self, extra=None):
        feature.dump(self, extra)
        with open('../feature/' + self.get_name(), 'a') as fout:
            indices, values = self.get_value()
            fmt = '%d:%' + self.get_data_type()
            for i in range(len(indices)):
                line = ' '.join([fmt % (indices[i][j], values[i][j]) for j in range(len(indices[i]))])
                fout.write(line + '\n')

    def process(self, **argv):
        feature.process(self, **argv)
        indices, _ = self.get_value()
        max_indices = map(get_max, indices)
        len_indices = map(lambda x: len(x), indices)
        self.set_space(max(max_indices) + 1)
        self.set_rank(max(len_indices))
        self.set_size(len(indices))


class seq_feature(feature):
    def __init__(self, name=None, ftype='seq', dtype=None, space=None, rank=None, size=None):
        feature.__init__(self, name, ftype, dtype, space, rank, size)

    def load(self):
        feature.load(self)

    def dump(self, extra=None):
        feature.dump(self, extra)
