import pythoncom
import argparse
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
accesskey = 'c9f9e00293c247649c92e7b00be8fa47'
secretkey = '9ae7550615734b2c80b16434a24ced84'
url = 'https://api-wuxi-1.cmecloud.cn:8443'
def request_webimage_file(imagepath):
    print("请求File参数")
    requesturl = '/api/ocr/v1/webimage'
    string=""

    try:
        ocr_client = CMSSEcloudOcrClient(accesskey, secretkey, url)
        response = ocr_client.request_ocr_service_file(requestpath=requesturl, imagepath=imagepath)

        data = json.loads(response.text)
        if 'body' in data:
            prism_wordsInfo = data['body']['content']['prism_wordsInfo']
            for i in prism_wordsInfo:
                string = string + str(i['word'])

    except ValueError as e:
        print(e)
    return string
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'pdf', 'docx', 'txt'}
def ner(text,predictor,args):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, cache_dir="./ner/cache/")
    vocab = data_loader.Vocabulary()
    vocab.load_Vocabulary(os.path.join("./ner/outputs", args.task))
    args.vocab = vocab
    pieces = tokenizer.tokenize(text)
    _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
    _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
    results = predictor.predcit_one(torch.tensor(_bert_inputs).unsqueeze(0).to('cuda'))
    entities = []
    entity = ''
    labels = results[1:] + [results[0]]
    label_type = ''
    words = pieces
    for i in range(len(pieces)):
        if labels[i] != 'O':
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
    return entities
def cleaner(currentPath,REMOVE_FILETYPES):
  if not os.path.isdir(currentPath):
    if currentPath.endswith(REMOVE_FILETYPES) or os.path.basename(currentPath).startswith('.'):
      try:
        os.remove(currentPath)
        print('REMOVED: \"{removed}\"'.format(removed = currentPath))
      except BaseException as e:
        print('ERROR: Could not remove: \"{failed}\"'.format(failed = str(e)))
      finally:
        return True
    return False

  if all([cleaner(os.path.join(currentPath, file)) for file in os.listdir(currentPath)]):
    try:
      os.rmdir(currentPath)
    except:
      print('ERROR: Could not remove: \"{failed}\"'.format(failed = currentPath))
    finally:
      return True
  return False
def pdf2text(path):
    str = ''
    # Open the PDF file in read-binary mode
    with open(path, 'rb') as pdf_file:

        # Create a PdfReader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Get the number of pages in the PDF file
        num_pages = len(pdf_reader.pages)

        # Loop through each page and extract the text
        for page_num in range(num_pages):
            # Get the page object
            page_obj = pdf_reader.pages[page_num]

            # Extract the text from the page object
            page_text = page_obj.extract_text()
            #字符串拼接
            str = str + page_text
            # Do something with the text
            # ...

    pdf_file.close()
    return str.replace('\n', '')

def docx2text(folder_path):
    pythoncom.CoInitialize()
    str = ''
    try:
        new_file_name = folder_path.replace("docx", "pdf")
        # 将 DOCX 文件转换为 PDF 文件，并保存到与 DOCX 文件相同的路径下
        convert(folder_path, './images')
        str = str + pdf2text(new_file_name)
        str = str + os.linesep+ os.linesep
    except Exception as e:
        print(e)
    return str