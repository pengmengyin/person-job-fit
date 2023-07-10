# -*- coding: utf-8 -*-
import numpy as np
from py2neo import Graph
from matching_algorithm.compute_score import compute_match_score
from matching_algorithm.text_summarization import text_summarization
from transformers import BertTokenizer, BertModel
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./ner/RoBERTa_zh"
tokenizer = BertTokenizer.from_pretrained(model_path)
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "12345678"))
bert = BertModel.from_pretrained(model_path).to(device)
def text2vec(text):
    cutlen = len(str(text))
    if cutlen > 510:
        cutlen = 510
    encoders = tokenizer.encode_plus(str(text), max_length=cutlen, truncation=True,
                                     padding='max_length')

    input_ids1 = [encoders['input_ids']]

    attention_mask1 = [encoders['attention_mask']]

    token_type_ids1 = [encoders['token_type_ids']]

    input_ids1 = torch.tensor(input_ids1).int().to(device)
    token_type_ids1 = torch.tensor(token_type_ids1).int().to(device)
    attention_mask1 = torch.tensor(attention_mask1).int().to(device)
    bert_outputs = bert(input_ids=input_ids1, token_type_ids=token_type_ids1, attention_mask=attention_mask1)[
        0]
    tensor = torch.mean(bert_outputs, dim=1).squeeze()
    return tensor.cpu().detach().numpy()


def node2text(node):
    temp_list = []
    for i in node:
        if str(node[i]) is not 'nan':
            temp_list.append(str(node[i]))
    text = ' '.join(temp_list)
    return text


def node2text_job(node):
    columns = ['job_type', 'responsibility', 'requirements']
    temp_list = []
    text = ''
    if node != None:
        for i in columns:

            temp_list.append(str(node[i]))
        text = ' '.join(temp_list)
    return text


def get_node_vector(node):
    # 根据节点的属性获取特征向量
    text = node2text(node)
    return text2vec(text)


def get_job_node_vectors():
    # 查询所有节点
    query = "MATCH (n:JOB) RETURN n"
    result = graph.run(query)
    vectors = []
    nodes = []
    count = 0
    for record in result:
        count += 1
        node = record['n']
        vector = get_node_vector(node)
        nodes.append(node)
        vectors.append(vector)
    return np.array(vectors), nodes


def reversed_sorted_list(original_list):
    sorted_list = sorted(enumerate(original_list), key=lambda x: x[1])
    reversed_sorted_list = sorted_list[::-1]
    result = [x[0] for x in reversed_sorted_list]
    return result


def find_job_nodes(text, input):
    node_vectors, nodes = get_job_node_vectors()
    job_list = []
    result_list = []
    similarity_list = []
    for vec1, node in zip(node_vectors, nodes):
        job_list.append(node)
        vec2 = list(input)
        vec1 = list(vec1)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        match_score = compute_match_score(text, node2text_job(node))
        recommendation_score = (match_score + similarity) / 2
        similarity_list.append(recommendation_score)
    sorted_1 = reversed_sorted_list(similarity_list)
    for i in sorted_1:
        temp_list = [node2text_job(job_list[i]), similarity_list[i]]
        result_list.append(temp_list)
    return result_list


def recommendation_score(input_user_text):
    # input_user_text = r'14bc0a88099f6b0b8047bfd279b34cc8,639,"639,-,-",房地产/建筑/建材/工程,房地产销售/置业顾问,1500125000,房地产/建筑/建材/工程,房地产开发/经纪/中介,1000115000,大专,30,2009'
    input_user_text = text_summarization(input_user_text)
    input = text2vec(input_user_text)
    print('为以下用户进行推荐：', input_user_text)
    print('推荐企业：')
    result_list = find_job_nodes(input_user_text, input)
    print(result_list)
    unique_list = []
    for x in result_list:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
