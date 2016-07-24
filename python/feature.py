class value_function:
    def __init__(self, input_type, output_type):
        """
        :param input_type:
        :param output_type: 'num', 'pair', 'list'
        """
        self.__input_type = input_type
        self.__output_type = output_type

    def apply(self, **argv):
        pass

    def get_input_type(self):
        return self.__input_type

    def get_output_type(self):
        return self.__output_type


class feature:
    def __init__(self, name, ftype, dtype, value_func):
        """
        :param name: identifier | string
        :param ftype: feature type | string, could be num, one_hot, multi, seq
        :param dtype: data type
        """
        self.__name = name
        self.__ftype = ftype
        self.__dtype = dtype
        self.__value_func = value_func

    def get_name(self):
        return self.__name

    def get_feature_type(self):
        return self.__ftype

    def get_data_type(self):
        return self.__dtype

    def get_value_func(self):
        return self.__value_func

    def get_value(self, **argv):
        return self.__value_func.apply(**argv)


class num_feature(feature):
    def __init__(self, name, dtype, value_func):
        assert value_func.get_output_type() == 'num'
        feature.__init__(self, name, 'num', dtype, value_func)


class one_hot_feature(feature):
    def __init__(self, name, dtype, value_func, space=None):
        assert value_func.get_output_type() == 'pair'
        feature.__init__(self, name, 'one_hot', dtype, value_func)
        self.__space = space

    def get_space(self):
        return self.__space

    def set_space(self, space):
        self.__space = space


class multi_feature(feature):
    def __init__(self, name, dtype, value_func, space=None, size=None):
        assert value_func.get_output_type() == 'list'
        feature.__init__(self, name, 'multi', dtype, value_func)
        self.__space = space
        self.__size = size

    def get_space(self):
        return self.__space

    def set_space(self, space):
        self.__space = space

    def get_size(self):
        return self.__size

    def set_size(self, size):
        self.__size = size


class seq_feature(feature):
    def __init__(self, name, dtype, value_func):
        assert value_func.get_output_type() == 'list'
        feature.__init__(self, name, 'seq', dtype, value_func)
