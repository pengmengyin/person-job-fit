import json
import logging
import time


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.loss_type = config["loss_type"]
        self.dataset = config["dataset"]
        self.conv_hid_size = config["conv_hid_size"]
        self.bert_hid_size = config["bert_hid_size"]
        self.dilation = config["dilation"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.bert_learning_rate = config["bert_learning_rate"]
        self.weight_decay = config["weight_decay"]

        for k, v in args.__dict__.items():  #__dict__:对象内部所有属性名和属性值组成的字典
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())
def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    #日志格式器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger()   #创建一个日志对象
    logger.setLevel(logging.INFO)


    #定义日志处理器，决定把日志发到哪里，输出到文件
    file_handler = logging.FileHandler(pathname)
    #定义日志级别和输出格式
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    #输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    #把handler添加到对应的logger中去
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
