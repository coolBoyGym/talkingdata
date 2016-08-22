class model:
    def __init__(self, name, mtype, eval_metric):
        self.__name = name
        self.__mtype = mtype
        self.__eval_metric = eval_metric
        self.__log_path = '../model/%s.log' % name
        self.__bin_path = '../model/%s.bin' % name
        self.__file_path = '../model/%s.dump' % name

    def get_name(self):
        return self.__name

    def get_model_type(self):
        return self.__mtype

    def get_eval_metric(self):
        return self.__eval_metric

    def get_log_path(self):
        return self.__log_path

    def get_bin_path(self):
        return self.__bin_path

    def get_file_path(self):
        return self.__file_path

    def write_log_header(self):
        with open(self.__log_path, 'w') as fout:
            fout.write('%s %s\n' % (self.__mtype, self.__eval_metric))

    def write_log(self, log_str):
        with open(self.__log_path, 'a') as fout:
            fout.write(log_str)

    def init(self, **argv):
        pass

    def train(self, **argv):
        pass

    def predict(self, **argv):
        pass

    def dump(self, **argv):
        pass


class Classifier(model):
    def __init__(self, name, eval_metric, input_spaces, input_types, num_class):
        model.__init__(self, name, 'classifier', eval_metric)
        print 'constructing classifier', name, eval_metric, input_spaces, input_types, num_class
        self.__input_spaces = input_spaces
        self.__input_types = input_types
        self.__num_class = num_class

    def get_num_class(self):
        return self.__num_class

    def get_input_spaces(self):
        return self.__input_spaces

    def get_input_types(self):
        return self.__input_types

    def write_log_header(self):
        with open(self.__log_path, 'w') as fout:
            fout.write('%s %s %s\n' % (self.__mtype, self.__eval_metric, self.__num_class))
