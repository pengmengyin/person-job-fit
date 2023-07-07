# -*- coding: utf-8 -*-
import argparse

from pdf2text import pdf2text
from utils.file2text import request_webimage_file, docx2text, allowed_file, cleaner, ner
from utils.postprocessing import post_processing
import PyPDF2
import numpy as np
import pythoncom
import torch
from docx2pdf import convert
import logging
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ecloud import CMSSEcloudOcrClient
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from ner import data_loader, models
from ner.inference import Predictor
app = Flask(__name__)
CORS(app, supports_credentials=True)
class Flask_Engine(object):
    def __init__(self, http_cfg, log_cfg):
        self.port = http_cfg['port']
        self.host = http_cfg['host']

        self.logDir = log_cfg['dir']
        self.logFileName = log_cfg['filename']
        self.debug = False

    def run(self):
        app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=False)

    def set_log(self):
        # 创建app实例前先配置好日志文件
        fmt = '%(asctime)s [%(levelname)s] [%(message)s]'
        logging.basicConfig(level=logging.INFO,
                            format=fmt,  # 定义输出log的格式
                            datefmt='[%Y-%m-%d %H:%M:%S]')
        if not os.path.exists(self.logDir):  # 创建日志目录
            os.makedirs(self.logDir)

        # 实例对象从配置文件中加载配置
        app.config.from_object(logging.INFO)
        return app






@app.route('/file2text', methods=['POST'])
def main():
    REMOVE_FILETYPES = ('.png', '.jpg', '.jpeg', '.pdf', '.docx', '.txt')
    # 检查是否有文件上传
    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']
    # 检查文件是否符合要求
    if file.filename == '':
        return 'No file selected'
    if not allowed_file(file.filename):
        return 'Invalid file type'
    # 保存上传的文件
    file_path = r'D:\work\competition\软件杯\dataset\images\\' + file.filename
    file.save(file_path)
    file.close()
    data_dict = {}
    data_dict['msg'] = "处理失败"
    data_dict['result'] = ''
    data_dict['text'] = ''
    data_dict['status'] = 0
    if file_path.endswith('.jpg') or file_path.endswith('.png') :
        results = request_webimage_file(file_path)
    elif file_path.endswith('.pdf'):
        results = pdf2text(file_path)
    elif file_path.endswith('.docx'):

        results = docx2text(file_path)
    elif file_path.endswith('.txt'):
        results = ''
        with open(file_path, 'r',encoding='utf-8') as f:
            # 读取文件内容
            results = results + str(f.read())
    else:
        print('输入文件错误')
    text = results.replace(" ", "").replace("\n", "").replace("\r", "")
    print(text)
    data_list = []
    sen = ''
    count = 0
    for token in text:
        count += 1
        sen = sen + token
        if count > 300:
            data_list.append(sen)
            sen = ''
            count = 0
    data_list.append(sen)
    results_list = []
    for text in data_list:
        results_list.append(ner(text,predictor,args))
    print(results_list)
    results = post_processing(results_list, ''.join(data_list))

    if results != '':
        data_dict['result'] = results
        data_dict['msg'] = "处理成功"
        data_dict['status'] = 1
        data_dict['text'] = ''.join(data_list)
    print(data_dict)
    res = jsonify(data_dict)
    cleaner(file_path,REMOVE_FILETYPES)
    return res
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ruanjianbei2')
    parser.add_argument('--save_path', type=str, default='./ner/outputs')
    parser.add_argument('--predict_path', type=str, default='./ner/outputs')
    # parser.add_argument('--bert_name', type=str, default=r"../../bert-base-chinese")
    parser.add_argument('--bert_name', type=str, default=r"./ner/RoBERTa_zh")
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
    predictor = Predictor(model,args)
    predictor.load(os.path.join("./ner/outputs", args.task, model.model_name + ".pt"))
    conf = json.load(open('./config/config.json', 'r'))
    eng = Flask_Engine
    eng = eng(conf['httpserver'], conf['log'])
    eng.set_log()
    eng.run()

