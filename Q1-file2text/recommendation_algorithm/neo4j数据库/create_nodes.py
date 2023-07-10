import pandas as pd
from py2neo import Graph, NodeMatcher,  Node

graph = Graph("neo4j://localhost:7687", auth=("neo4j", "12345678"))
# 定义 Cypher 查询语句，删除所有类型为 "Person" 的节点
query = "MATCH (p:JOB) DELETE p"

# 执行查询语句
graph.run(query)
job_df = pd.read_csv('./import/job.csv')[['岗位类型','岗位职责','任职要求']]

job_matcher = NodeMatcher(graph)
# 创建关系匹配器
flag = 0
for job_data  in job_df.values:
    job_type = job_data[0]
    job_node = job_matcher.match('JOB', job_type=job_type).first()
    if not job_node:
        job = Node("JOB", job_type=job_type, responsibility=job_data[1], requirements=job_data[2])
        graph.create(job)
