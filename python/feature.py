import numpy as np

import feature_impl


def get_map(dtype):
    if 'd' in dtype:
        return np.int64
    elif 'f' in dtype:
        return np.float64


class feature:
    def __init__(self, name=None, ftype=None, dtype=None, space=None, rank=None):
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
        if name is not None:
            self.__proc = getattr(feature_impl, name + '_proc')
        self.__space = space
        self.__rank = rank
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

    def process(self, **argv):
        self.__indices, self.__values = self.__proc(**argv)

    def set_value(self, indices=None, values=None):
        if indices is not None:
            self.__indices = indices
        if values is not None:
            self.__values = values

    def get_value(self):
        return self.__indices, self.__values

    def dump(self):
        assert self.__indices is not None and self.__values is not None
        with open('../feature/' + self.__name, 'w') as fout:
            fout.write('%s %s %s %s\n' % (
                self.__ftype, self.__dtype, str(self.__space), str(self.__rank)))

    def load(self):
        with open('../feature/' + self.__name, 'r') as fin:
            ftype, dtype, space, rank = fin.readline().strip().split()
            if space == 'None':
                space = None
            if rank == 'None':
                rank = None

            self.__init__(self.__name, ftype, dtype, space, rank)


class num_feature(feature):
    def __init__(self, name=None, ftype='num', dtype=None, space=1, rank=1):
        feature.__init__(self, name, ftype, dtype, space, rank)

    def load(self):
        feature.load(self)
        with open('../feature/' + self.get_name(), 'r') as fin:
            next(fin)
            data = [line.strip() for line in fin]
            values = map(get_map(self.get_data_type()), data)

            values = np.array(values)
            indices = np.zeros_like(values, dtype=np.int64)

            self.set_value(indices=indices, values=values)

    def dump(self):
        feature.dump(self)
        with open('../feature/' + self.get_name(), 'a') as fout:
            _, values = self.get_value()
            fmt = '%' + self.get_data_type() + '\n'
            for i in range(len(values)):
                fout.write(fmt % values[i])


class one_hot_feature(feature):
    def __init__(self, name=None, ftype='one_hot', dtype=None, space=None, rank=1):
        feature.__init__(self, name, ftype, dtype, space, rank)

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

    def dump(self):
        feature.dump(self)
        with open('../feature/' + self.get_name(), 'a') as fout:
            indices, values = self.get_value()
            fmt = '%d:%' + self.get_data_type() + '\n'
            for i in range(len(indices)):
                fout.write(fmt % (indices[i], values[i]))


class multi_feature(feature):
    def __init__(self, name=None, ftype='multi', dtype=None, space=None, rank=None):
        feature.__init__(self, name, ftype, dtype, space, rank)

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

    def dump(self):
        feature.dump(self)
        with open('../feature/' + self.get_name(), 'a') as fout:
            indices, values = self.get_value()
            fmt = '%d:%' + self.get_data_type()
            for i in range(len(indices)):
                line = ' '.join([fmt % (indices[i][j], values[i][j]) for j in range(len(indices[i]))])
                fout.write(line + '\n')


class seq_feature(feature):
    def __init__(self, name=None, ftype='seq', dtype=None, space=None, rank=None):
        feature.__init__(self, name, ftype, dtype, space, rank)

    def load(self):
        feature.load(self)

    def dump(self):
        feature.dump(self)
