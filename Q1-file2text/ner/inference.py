# -*- coding: utf-8 -*-
import re
import re

import numpy as np
import spacy
from transformers import AutoTokenizer

from ner import data_loader
from tqdm import tqdm
import torch
import json
import argparse
from torch.utils.data import DataLoader
from ner import models
import os


class Predictor(object):
    def __init__(self, model,args):
        self.model = model
        self.args = args

    def load(self, path):
        self.model.load_state_dict(torch.load(path),strict=False)

    def sequence_tag2tag(self, text, label):
        # 统计序列标注中的实体及其类型
        # 存放实体相关信息，以字典结构保存，其中包括entity、type以及index
        item = dict()
        # 保存当前正在读取的实体，实体结束后会存入item["entity"]中
        _entity = str()
        # ner中存放当前语料包含的所有实体
        ner = list()
        index = list()
        # 遍历序列标注形式的标签，如果当前标签中包含“B-”则表明“上一个实体已经读取完毕，现在开始要开始读取一个新的实体”
        # 如果当前标签中包含“I-”，说明正在读取的实体还未结束，将当前标签所对应的字添加进_entity中，继续遍历
        # 循环结束后，如果item中不为空，说明存在有未保存的实体，将相关实体信息添加到字典中，最后添加到数据集中。
        for i, (t, l) in enumerate(zip(text, label)):
            if "B-" in l:
                if item:
                    item["entity"] = _entity
                    item["index"] = index
                    ner.append(item)
                    _entity = str()
                    item = dict()
                    index = list()
                item["type"] = l.split("-")[1]
                _entity = t
                index.append(i)
            if "I-" in l and item is not None:
                _entity += t
                index.append(i)
        if item:
            item["entity"] = _entity
            item["index"] = index
            ner.append(item)
            _entity = str()
            item = dict()
            index = list()
        return ner

    def predcit(self, data_loader, origin_data):
        result = list()
        batch = 0
        with torch.no_grad():
            for data_batch in tqdm(data_loader):
                sentence_batch = origin_data[batch: batch + self.args.batch_size]
                data_batch = [data.to(self.args.device) for data in data_batch]
                bert_inputs, sent_length = data_batch
                print(bert_inputs)
                outputs = self.model(bert_inputs)
                for sentence, pred_label, bert_input in zip(sentence_batch, outputs, bert_inputs):
                    sentence = sentence["sentence"]
                    pred_label = torch.argmax(pred_label, -1)[bert_input.ne(0).byte()].cpu().numpy()
                    pred_label = [self.args.vocab.id_to_label(str(i)) for i in pred_label]
                    result.append({"sentence": sentence, "label": self.sequence_tag2tag(sentence, pred_label)})
                batch += self.args.batch_size
        with open(os.path.join(self.args.save_path, self.args.task, "model_predicted.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def predcit_one(self, bert_inputs):
        with torch.no_grad():
            outputs = self.model(bert_inputs)
            for pred_label, bert_input in zip( outputs, bert_inputs):
                pred_label = torch.argmax(pred_label, -1)[bert_input.ne(0).byte()].cpu().numpy()
                pred_label = [self.args.vocab.id_to_label(str(i)) for i in pred_label]
            return pred_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ruanjianbei2')
    parser.add_argument('--save_path', type=str, default='./outputs')
    parser.add_argument('--predict_path', type=str, default='./outputs')
    # parser.add_argument('--bert_name', type=str, default=r"../../bert-base-chinese")
    parser.add_argument('--bert_name', type=str, default=r"./RoBERTa_zh")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--bert_hid_size', type=int, default=768)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args(args=[])

    datasets, ori_data = data_loader.load_real_bert(args)
    real_loader = DataLoader(dataset=datasets,
                             batch_size=args.batch_size,
                             collate_fn=data_loader.pred_collate_fn,
                             shuffle=False,
                             # num_workers=4,
                             drop_last=False)

    model = models.bertBiLSTM_CRF(args).to(args.device)
    predictor = Predictor(model)

    predictor.load(os.path.join("./outputs", args.task, model.model_name + ".pt"))
    # predictor.predcit(real_loader, ori_data)
    text = "HELLOI’MLiulixia产品经理&产品运营刘力霞-关于我自己性别男年龄28生日1996.06.22星座巨蟹籍贯北京民族汉具有丰富的分销、供应链方面的项目管理经验；有较强的数据分析和网页设计能力，熟悉产品开发全部流程；熟悉UML语言，有系统设计，数据库设计经验。热爱互联网产品，具备产品线整体规划和把控能力，善于沟通陈述产品设计理念；能思考，能坚持，爱学习，爱合作。教育水平首都经济贸易大学软件工程交互艺术硕士2018.9-2021.6GPA3.7/4.0专业排名3/50首都经济贸易大学软件工程本科2014.18-2018.6GPA3.5/4.0专业排名6/57英语CET6国家计算机三级（数据库）工作经历微软在线OfficePLUS项目组产品经理2022.7至今•负责PC端及移动端的网站前后台产品的业务需求及规划；•撰写产品V1.0-V2.0版本需求文档，推动产品研发整个过程；•进行数据统计分析以及竞争对手情况分析跟踪；并通过数据和用户反馈，分析用户需求及行为，产品发布三个月用户量达10"
    text = ''.join(text)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, cache_dir="./cache/")
    vocab = data_loader.Vocabulary()
    vocab.load_Vocabulary(os.path.join("./outputs", args.task))
    args.vocab = vocab
    pieces = tokenizer.tokenize(text)
    _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
    _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
    length = len(_bert_inputs)
    # print(torch.tensor(_bert_inputs))
    # print(pieces)
    results = predictor.predcit_one(torch.tensor(_bert_inputs).unsqueeze(0).to('cuda'))
    # print(results)

    entities = []
    entity = ''
    labels = results[1:]+[results[0]]
    label_type = ''
    words = pieces
    for i in range(len(pieces)):

        if labels[i] != 'O':
            print(labels[i])
            if labels[i][0] == 'B':
                if entity != '':
                    entities.append((entity, label_type))
                entity = words[i]
                label_type = labels[i][2:]
            elif labels[i][0] == 'M':
                entity += words[i]
            elif labels[i][0] == 'E':
                entity += words[i]
                entities.append((entity, label_type))
                entity = ''
                label_type = ''
        else:
            if entity != '':
                entities.append((entity, label_type))
                entity = ''
                label_type = ''

    if entity != '':
        entities.append((entity, label_type))

    print(text)
    print(entities)
    # nlp = spacy.load('D:\work\study\pytorch2\\nlp\zh_core_web_sm-3.5.0')  # 加载中文包
    # doc = nlp(text)
    # entity = ['PERSON', 'DATE']
    # dic = {}
    # for ent in doc.ents:
    #
    #     if ent.label_ in entity:
    #         dic[ent.label_] = ent.text
    #
    # # 定义正则表达式
    # pattern = re.compile(r'([\u4e00-\u9fa5]{2})奖')
    #
    # # 抽取等奖级别
    # match = pattern.search(''.join(text))
    # if match:
    #     prize_level = match.group(1)
    #     dic['AWA'] = prize_level + '奖'
    # print(dic)
