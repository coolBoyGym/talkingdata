class model:
    def __init__(self, name, mtype, eval_metric):
        self.__name = name
        self.__mtype = mtype
        self.__eval_metric = eval_metric
        self.__log_path = '../model/%s.log' % name
        self.__bin_path = '../model/%s.bin' % name
        self.__dump_path = '../model/%s.dump' % name

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

    def get_dump_path(self):
        return self.__dump_path

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


class classifier(model):
    def __init__(self, name, eval_metric, num_class):
        model.__init__(self, name, 'classifier', eval_metric)
        self.__num_class = num_class

    def get_num_class(self):
        return self.__num_class

    def write_log_header(self):
        with open(self.__log_path, 'w') as fout:
            fout.write('%s %s %s\n' % (self.__mtype, self.__eval_metric, self.__num_class))
