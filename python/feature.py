import numpy as np

import feature_impl
import utils


class Feature:
    def __init__(self, name=None, ftype=None, dtype=None, space=None, rank=None, size=None):
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
        self.__sub_features = None
        self.__sub_spaces = None
        self.__sub_ranks = None

    def get_name(self):
        return self.__name

    def get_feature_type(self):
        return self.__ftype

    def set_feature_type(self, ftype):
        self.__ftype = ftype

    def get_data_type(self):
        return self.__dtype

    def set_data_type(self, dtype):
        self.__dtype = dtype

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

    def set_sub_features(self, sub_features):
        self.__sub_features = sub_features

    def get_sub_features(self):
        return self.__sub_features

    def set_sub_spaces(self, sub_spaces):
        self.__sub_spaces = sub_spaces

    def get_sub_spaces(self):
        return self.__sub_spaces

    def set_sub_ranks(self, sub_ranks):
        self.__sub_ranks = sub_ranks

    def get_sub_ranks(self):
        return self.__sub_ranks

    def process(self, **argv):
        self.__indices, self.__values = self.__proc(**argv)

    def set_value(self, indices=None, values=None):
        if indices is not None:
            self.__indices = indices
        if values is not None:
            self.__values = values
        max_indices = map(utils.general_max, indices)
        len_indices = map(utils.general_len, indices)
        self.set_space(max(max_indices) + 1)
        self.set_rank(max(len_indices))
        self.set_size(len(indices))

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

    def load_meta(self):
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

    def load_meta_extra(self):
        with open('../feature/' + self.__name, 'r') as fin:
            meta = fin.readline().strip().split()
            if len(meta) > 5:
                extra = meta[5]
                fea_names = extra.split(',')
                spaces = []
                ranks = []
                for fn in fea_names:
                    fea_tmp = Feature(name=fn)
                    fea_tmp.load_meta()
                    spaces.append(fea_tmp.get_space())
                    ranks.append(fea_tmp.get_rank())
                self.set_sub_features(fea_names)
                self.set_sub_spaces(spaces)
                self.set_sub_ranks(ranks)
                return True
        return False


class NumFeature(Feature):
    def __init__(self, name=None, ftype='num', dtype=None, space=1, rank=1, size=None):
        Feature.__init__(self, name, ftype, dtype, space, rank, size)

    def load(self):
        Feature.load_meta(self)
        with open('../feature/' + self.get_name(), 'r') as fin:
            next(fin)
            data = [line.strip() for line in fin]
            values = map(utils.str_2_value, data)
            values = np.array(values)
            indices = np.zeros_like(values, dtype=np.int64)
            self.set_value(indices=indices, values=values)

    def dump(self, extra=None):
        Feature.dump(self, extra)
        with open('../feature/' + self.get_name(), 'a') as fout:
            _, values = self.get_value()
            for i in range(len(values)):
                fout.write(str(values[i]))

    def process(self, **argv):
        Feature.process(self, **argv)
        self.set_space(1)
        self.set_rank(1)
        self.set_size(len(self.get_value()[0]))


class OneHotFeature(Feature):
    def __init__(self, name=None, ftype='one_hot', dtype=None, space=None, rank=1, size=None):
        Feature.__init__(self, name, ftype, dtype, space, rank, size)

    def load(self):
        Feature.load_meta(self)
        with open('../feature/' + self.get_name(), 'r') as fin:
            next(fin)
            data = [line.strip().split(':') for line in fin]
            indices = map(utils.str_2_value, map(lambda x: x[0], data))
            values = map(utils.str_2_value, map(lambda x: x[1], data))
            indices = np.array(indices)
            values = np.array(values)
            self.set_value(indices=indices, values=values)

    def dump(self, extra=None):
        Feature.dump(self, extra)
        with open('../feature/' + self.get_name(), 'a') as fout:
            indices, values = self.get_value()
            for i in range(len(indices)):
                fout.write(str(indices[i]) + ':' + str(values[i]) + '\n')

    def process(self, **argv):
        Feature.process(self, **argv)
        indices, _ = self.get_value()
        self.set_space(np.max(indices) + 1)
        self.set_rank(1)
        self.set_size(len(indices))


class MultiFeature(Feature):
    def __init__(self, name=None, ftype='multi', dtype=None, space=None, rank=None, size=None):
        Feature.__init__(self, name, ftype, dtype, space, rank, size)

    def load(self):
        Feature.load_meta(self)
        with open('../feature/' + self.get_name(), 'r') as fin:
            next(fin)
            indices = []
            values = []
            for line in fin:
                entry = map(lambda x: x.split(':'), line.strip().split())
                indices.append(np.array(map(utils.str_2_value, map(lambda x: x[0], entry))))
                values.append(map(utils.str_2_value, map(lambda x: x[1], entry)))
        indices = np.array(indices)
        values = np.array(values)
        self.set_value(indices=indices, values=values)

    def dump(self, extra=None):
        Feature.dump(self, extra)
        with open('../feature/' + self.get_name(), 'a') as fout:
            indices, values = self.get_value()
            for i in range(len(indices)):
                line_arr = []
                for j in range(len(indices[i])):
                    line_arr.append(str(indices[i][j]) + ':' + str(values[i][j]))
                line = ' '.join(line_arr)
                fout.write(line + '\n')

    def process(self, **argv):
        Feature.process(self, **argv)
        indices, _ = self.get_value()
        max_indices = map(utils.general_max, indices)
        len_indices = map(lambda x: len(x), indices)
        self.set_space(max(max_indices) + 1)
        self.set_rank(max(len_indices))
        self.set_size(len(indices))

    def reorder(self):
        indices, values = self.get_value()
        for i in range(len(indices)):
            row_indices = np.array(indices[i])
            row_values = np.array(values[i])
            sorted_indices = sorted(range(len(row_indices)), key=lambda x: row_indices[x])
            indices[i] = row_indices[sorted_indices]
            values[i] = row_values[sorted_indices]
        self.set_value(indices=indices, values=values)


class SeqFeature(Feature):
    def __init__(self, name=None, ftype='seq', dtype=None, space=None, rank=None, size=None):
        Feature.__init__(self, name, ftype, dtype, space, rank, size)

    def load(self):
        Feature.load_meta(self)

    def dump(self, extra=None):
        Feature.dump(self, extra)
