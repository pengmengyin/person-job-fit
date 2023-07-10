import pandas as pd
from py2neo import Graph, NodeMatcher, Node
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "12345678"))


def CreatUserNode(data_dic, text):
    user_matcher = NodeMatcher(graph)
    if data_dic['Name'] != '':
        user_node = user_matcher.match('USER', user_name=data_dic['Name']).first()
        if not user_node:
            user = Node("USER", user_name=data_dic['Name'], age=data_dic['Age'], education=data_dic['Education'],
                        school=data_dic['SCHOOL'], yearsofwork=data_dic['YEARSOFWORK'], text=text)
            graph.create(user)
def inquiryUserNodes():
    # 定义 Cypher 查询语句，统计所有类型为 "Person" 的节点数量
    query = "MATCH (n:USER) RETURN n"
    # 执行查询语句，获取节点数量
    result = graph.run(query).data()
    result_list = []

    for node in result:
        for key in node['n'].keys():
            if node['n'][key] == '':
                node['n'][key] = '无'
        result_list.append(node['n'])
    return result_list

def inquiryJobNodes():
    # 定义 Cypher 查询语句，统计所有类型为 "Person" 的节点数量
    query = "MATCH (n:JOB) RETURN n"
    # 执行查询语句，获取节点数量
    result = graph.run(query).data()
    result_list = []

    for node in result:
        for key in node['n'].keys():
            if node['n'][key] == '':
                node['n'][key] = '无'
        result_list.append(node['n'])
    return result_list


def UserNum():
    # 定义 Cypher 查询语句，统计所有类型为 "Person" 的节点数量
    query = "MATCH (p:USER) RETURN COUNT(p)"
    # 执行查询语句，获取节点数量
    result = graph.evaluate(query)
    return result

def JobNum():
    # 定义 Cypher 查询语句，统计所有类型为 "Person" 的节点数量
    query = "MATCH (p:JOB) RETURN COUNT(p)"
    # 执行查询语句，获取节点数量
    result = graph.evaluate(query)
    return result

def user_lenth():
    # 定义 Cypher 查询语句，查询所有类型为 "Person" 的节点的 "name" 字段长度分布
    query = """
    MATCH (p:USER)
    RETURN p.text
    """

    # 执行查询语句，获取查询结果
    result = graph.run(query).data()
    # 将查询结果转换为 Pandas 数据框
    df = pd.DataFrame(result)
    # df["resume_length"] = list(map(lambda x: len(str(x)), df["p.text"]))
    # sns.distplot(df["resume_length"])
    # plt.title('resume_length')
    # plt.yticks([])
    # plt.savefig('D:\work\competition\软件杯\dataset\images\\resume_length.png')

    # 打印数据框
    return df
if __name__ == '__main__':
    print(user_lenth())