from transformers import AdamW
import torch
from tqdm import tqdm
from seqeval.metrics import accuracy_score, recall_score, f1_score
import argparse
import prettytable as pt
import utils
import numpy as np
import data_loader
from torch.utils.data import DataLoader
import models
import os


class Trainer(object):
    def __init__(self, model):
        self.model = model
        criterion = {
            "ce": torch.nn.CrossEntropyLoss(),
        }
        self.criterion = criterion[config.loss_type]
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]
        self.optimizer = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = list()
        origin_labels = list()
        pred_labels = list()

        # 拿batch数据
        for i, data_batch in tqdm(enumerate(data_loader)):  # tqdm：进度条

            # 把数据拷贝到device上
            data_batch = [data.to(config.device) for data in data_batch]
            bert_inputs, bert_labels, sent_length = data_batch
            # 输入模型
            outputs = self.model(bert_inputs)
            # 计算损失函数
            ###########################################################################################
            loss = self.criterion(
                outputs.view(-1, config.label_num)[(bert_inputs.ne(0).view(-1)) == 1],
                bert_labels.view(-1)[(bert_inputs.ne(0).view(-1)) == 1]
            )
            ###########################################################################################

            #CRF的损失函数
            #######################################################
            # loss = self.model.compute_loss(bert_inputs, bert_labels)
            ##############################################

            # 梯度下降反向传播
            loss.backward()  # 反向传播得到每个参数的梯度值
            self.optimizer.step()  # 通过梯度下降执行一步参数更新
            self.optimizer.zero_grad()  # 将梯度归零
            loss_list.append(loss.cpu().item())

            for origin_label, pred_label, bert_input in zip(bert_labels, outputs, bert_inputs):
                origin_label = origin_label[bert_input.ne(0).byte()].cpu().numpy()
                pred_label = torch.argmax(pred_label, -1)[bert_input.ne(0).byte()].cpu().numpy()
                origin_label = [config.vocab.id_to_label(i) for i in origin_label]
                pred_label = [config.vocab.id_to_label(i) for i in pred_label]
                origin_labels.append(origin_label)
                pred_labels.append(pred_label)
        p = accuracy_score(origin_labels, pred_labels)  # 计算准确率
        r = recall_score(origin_labels, pred_labels)  # 计算召回率
        f1 = f1_score(origin_labels, pred_labels)  # 计算f1
        #f1 = 2*p*r/(p+r)

        # 表格
        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))

    def eval(self, epoch, data_loader):
        self.model.eval()
        loss_list = list()
        origin_labels = list()
        pred_labels = list()
        with torch.no_grad():  # 不会对Tensor自动求导
            for i, data_batch in tqdm(enumerate(data_loader)):
                data_batch = [data.to(config.device) for data in data_batch]
                bert_inputs, bert_labels, sent_length = data_batch
                outputs = self.model(bert_inputs)
                for origin_label, pred_label, bert_input in zip(bert_labels, outputs, bert_inputs):
                    origin_label = origin_label[bert_input.ne(0).byte()].cpu().numpy()
                    pred_label = torch.argmax(pred_label, -1)[bert_input.ne(0).byte()].cpu().numpy()
                    origin_label = [config.vocab.id_to_label(i) for i in origin_label]
                    pred_label = [config.vocab.id_to_label(i) for i in pred_label]
                    origin_labels.append(origin_label)
                    pred_labels.append(pred_label)
        p = accuracy_score(origin_labels, pred_labels)
        r = recall_score(origin_labels, pred_labels)
        f1 = f1_score(origin_labels, pred_labels)
        table = pt.PrettyTable(["Dev {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))
        return f1

    def save(self):
        torch.save(
            self.model.state_dict(), os.path.join(config.save_path, config.dataset, self.model.model_name + ".pt")
        )

    def load(self, path=None):
        if path:
            self.model.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(config.save_path, config.dataset, self.model.model_name + ".pt")
                )
            )


if __name__ == '__main__':
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
            best_test_f1 = f1
            trainer.save()
    logger.info("Best DEV F1: {:3.4f}".format(best_test_f1))
