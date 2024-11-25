from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient, SecurityClient

# es = Elasticsearch(["http://220.248.90.51:14920"],http_auth=("researcher", "researcher_readonly_password!"),timeout=180)
es = Elasticsearch(["http://123.56.26.96:9200"],http_auth=("researcher", "researcher_readonly_password!"),timeout=180)

from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient, SecurityClient
from tqdm import tqdm

print("es connected")

# 初始化和查询设置
term_list=[] # 用于存储查询的关键词
subjects=['Trump',"川建国","选举"] # 查询的关键词
scroll_id="" # 用于存储scroll_id
for t in subjects: # 遍历关键词，添加到term_list中
    term_list.append({"term":{"text":t}})

print("query initialized")

# 执行查询操作
query={
  "query": { # 查询条件
    "query_string": { # 使用查询字符串查询
      "query": "tweet_id:1746144577935769928"
    }
  },
  "size": 10, # 每次查询返回的数据量
  "from": 0, # 从第0条数据开始查询
  "sort": [] # 排序条件
}

result = es.search(index='tweet_timing_sequence', scroll='1m', body=query)

print("query executed")

# 有一些其他的index，里面会存储用户数据
# 处理第一批数据
text_list=[]
source_list=[]
harm_list=[]
logits=[]
# 提取每个文档的_source字段，打印text字段
for hit in result['hits']['hits']:
    source=hit['_source']
    text=source['text']
    print(text)

# 处理剩余数据
text_list=[]
source_list=[]
scroll_id=result['_scroll_id']
# 使用es.scroll方法获取下一批数据，并提取_source字段
while len(result['hits']['hits']) > 0:
    result = es.scroll(scroll_id=result['_scroll_id'], scroll='1m')
    for hit in result['hits']['hits']:
        source=hit['_source']
            
es.clear_scroll(scroll_id=scroll_id)
