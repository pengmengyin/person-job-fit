# -*- coding: utf-8 -*-
import argparse

import matplotlib
from matplotlib import pyplot as plt

from matching_algorithm.compute_score import compute_match_score
from matching_algorithm.text_summarization import text_summarization
from recommendation_algorithm.recommendation import recommendation_score
from utils.neo4j import CreatUserNode, UserNum, JobNum, user_lenth, inquiryUserNodes, inquiryJobNodes
from utils.file2text import request_webimage_file, docx2text, allowed_file, cleaner, ner, pdf2text, doc2text
from utils.postprocessing import post_processing
import os
from torch.utils.data import DataLoader
import json
from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from ner import data_loader, models
from ner.inference import Predictor
import seaborn as sns
matplotlib.use('Agg')
app = Flask(__name__)
CORS(app, supports_credentials=True)


class Flask_Engine(object):
    def __init__(self, http_cfg):
        self.port = http_cfg['port']
        self.host = http_cfg['host']
        self.debug = False

    def run(self):
        app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=False, threaded=True)


@app.route('/inquiryUser', methods=['POST'])
def inquiry_user():
    data_dict = {'msg': "处理失败", 'result': '', 'status': 0}
    user_nodes = inquiryUserNodes()
    if user_nodes != None:
        data_dict['result'] = user_nodes
        data_dict['msg'] = "处理成功"
        data_dict['status'] = 1
    res = jsonify(data_dict)
    return res


@app.route('/inquiryJob', methods=['POST'])
def inquiry_job():
    data_dict = {'msg': "处理失败", 'result': '', 'status': 0}
    job_nodes = inquiryJobNodes()
    if job_nodes is not None:
        data_dict['result'] = job_nodes
        data_dict['msg'] = "处理成功"
        data_dict['status'] = 1
    res = jsonify(data_dict)
    return res


@app.route('/recommendation', methods=['POST'])
def recommendation():
    data_dict = {'msg': "处理失败", 'result': '', 'status': 0}
    # 从 HTTP 请求中获取数据
    string1 = request.form.get("string1")
    input_user_text = text_summarization(string1)
    results = recommendation_score(input_user_text)
    if results != None:
        data_dict['result'] = results
        data_dict['msg'] = "处理成功"
        data_dict['status'] = 1
    res = jsonify(data_dict)
    return res


@app.route('/personJobMatching', methods=['POST'])
def person_job_matching():
    data_dict = {'msg': "处理失败", 'result': '', 'status': 0}
    # 从 HTTP 请求中获取数据
    string1 = request.form.get("string1")
    string2 = request.form.get("string2")
    # 进行处理
    text1 = text_summarization(string1)
    text2 = text_summarization(string2)
    match_score = compute_match_score(text1, text2)
    if match_score != None:
        data_dict['result'] = match_score
        data_dict['msg'] = "处理成功"
        data_dict['status'] = 1
    res = jsonify(data_dict)
    return res


@app.route('/statistics', methods=['POST'])
def statistics():
    data_dict = {'msg': "处理失败", 'result': '', 'status': 0}
    result = {}
    result['user_num'] = UserNum()
    result['job_num'] = JobNum()
    image_url = url_for('serve_image', filename='resume_length.png')
    result['image_url'] = f'<img src="{image_url}" alt="My Image">'
    if result != None:
        data_dict['result'] = result
        data_dict['msg'] = "处理成功"
        data_dict['status'] = 1
    df = user_lenth()
    df["resume_length"] = list(map(lambda x: len(str(x)), df["p.text"]))
    sns.distplot(df["resume_length"])
    plt.title('resume_length')
    plt.yticks([])
    plt.savefig('D:\work\competition\软件杯\dataset\images\\resume_length.png')
    res = jsonify(data_dict)

    return res

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/file2text', methods=['POST'])
def file2text():
    data_dict = {'msg': "处理失败", 'result': '', 'text': '', 'status': 0}
    REMOVE_FILETYPES = ('.png', '.jpg', '.jpeg', '.pdf', '.doc', '.docx', '.txt')
    # 检查是否有文件上传

    if 'file' not in request.files:
        return 'No file uploaded'
    file = request.files['file']

    # 检查文件是否符合要求
    if file.filename == '':
        return 'No file selected'
    if not allowed_file(file.filename):
        data_dict['result'] = '无效文件'
        return data_dict

    # 保存上传的文件
    file_path = r'D:\work\competition\软件杯\dataset\images\\' + file.filename
    file.save(file_path)
    file.close()

    if file_path.endswith('.jpg') or file_path.endswith('.png'):
        results = request_webimage_file(file_path)
    elif file_path.endswith('.pdf'):
        results = pdf2text(file_path)
    elif file_path.endswith('.docx'):
        results = docx2text(file_path)
    elif file_path.endswith('.doc'):
        results = doc2text(file_path)
    elif file_path.endswith('.txt'):
        results = ''
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取文件内容
            results = results + str(f.read())
    else:
        print('输入文件错误')
    text = results.replace(" ", "").replace("\n", "").replace("\r", "")

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
        results_list.append(ner(text, predictor, args))
    results = post_processing(results_list, ''.join(data_list))
    results['text'] = ''.join(data_list)
    if results != '':
        data_dict['result'] = results
        data_dict['msg'] = "处理成功"
        data_dict['status'] = 1
    CreatUserNode(results, ''.join(data_list))
    res = jsonify(data_dict)
    cleaner(file_path, REMOVE_FILETYPES)
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
    predictor = Predictor(model, args)
    predictor.load(os.path.join("./ner/outputs", args.task, model.model_name + ".pt"))
    conf = json.load(open('./config/config.json', 'r'))
    eng = Flask_Engine
    eng = eng(conf['httpserver'])
    eng.run()
