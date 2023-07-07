import argparse
import utils
import data_loader
from torch.utils.data import DataLoader
from main import Trainer
import models

parser = argparse.ArgumentParser()  # 创建解析器，添加参数
parser.add_argument('--config', type=str, default='./config/chinese_news.json')
parser.add_argument('--save_path', type=str, default='./outputs')
parser.add_argument('--bert_name', type=str, default=r"../../RoBERTa_zh")
# parser.add_argument('--bert_name', type=str, default=r"../../bert-base-chinese")
parser.add_argument('--device', type=str, default="cuda")
args = parser.parse_args()  # 解析添加的参数
config = utils.Config(args)  # 字典

logger = utils.get_logger(config.dataset)  # 日志
logger.info(config)
config.logger = logger

# from data_loader.py import load_data_bert
datasets, ori_data = data_loader.load_data_bert(config)  # datasets:DataSets, ori_data:原始数据
# datasets = train_dataset, dev_dataset
train_loader, dev_loader = (
    DataLoader(dataset=dataset,  # DataLoader构建可迭代的数据装载器，每次从DataSets中获取一个batch_size大小的数据
               batch_size=config.batch_size,  # 批大小
               collate_fn=data_loader.collate_fn,  # 将数据进行进一步处理
               shuffle=i == 0,  # 每个epoch是否乱序
               # num_workers=4,
               drop_last=i == 0)  # 当样本数量不能被batch_size整除时是否舍去最后一批数据
    for i, dataset in enumerate(datasets)
)
updates_total = len(datasets[0]) // config.batch_size * config.epochs
logger.info("Building Model")
model = models.bertBiLSTM_CRF(config).to(config.device)
trainer = Trainer(model)

best_f1 = 0
best_test_f1 = 0
# 训练config.epochs次
for i in range(config.epochs):
    logger.info("Epoch: {}".format(i))
    # 训练模型
    trainer.train(i, train_loader)
    # 训练结束验证训练效果  f1值
    f1 = trainer.eval(i, dev_loader)
    if f1 > best_test_f1:
        best_f1 = f1
        trainer.save()
logger.info("Best DEV F1: {:3.4f}".format(best_f1))