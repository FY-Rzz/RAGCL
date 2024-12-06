# -*- coding: utf-8 -*-
# @Time    : 20241206
# @Author  : xiezw
# @Email   :
# @File    : commentcnt.py
# @Software: vscode
# @Note    : 计算评论数
import pandas as pd
import json
import os
import os.path as osp

# 数据路径
dataset = 'DRWV3'
dirpath = osp.dirname(osp.abspath(__file__))
csvpath = osp.join(dirpath, '..', 'TestLog', f'incorrect_samples_{dataset}_new.csv')
jsonpath = osp.join(dirpath, '..', 'Data', f'{dataset}', 'source') #f

# 读取 CSV 文件
df = pd.read_csv(csvpath)

# 添加新列 'comment_cnt' 用于存储评论数量
df['comment_cnt'] = 0
df['pagerank_gt_0.1'] = 0
df['pagerank_gt_1'] = 0

# 遍历 DataFrame 中的每一行
for index, row in df.iterrows():
    raw_id = row['raw_id']
    json_file_path = osp.join(jsonpath, f'{raw_id}.json')
    
    # 检查 JSON 文件是否存在
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            comment_length = len(data.get('comment', []))
            df.at[index, 'comment_cnt'] = comment_length  # 更新 DataFrame 中的 'comment_cnt' 列

            # 计算 Pagerank 数列中大于 0.1 和大于 1 的属性值数量
            pagerank_values = data.get('centrality', {}).get('Pagerank', [])
            pagerank_gt_01 = sum(1 for value in pagerank_values if value > 0.1)
            pagerank_gt_1 = sum(1 for value in pagerank_values if value > 1)
            df.at[index, 'pagerank_gt_0.1'] = pagerank_gt_01
            df.at[index, 'pagerank_gt_1'] = pagerank_gt_1

            # 获取数据'theme'属性值
            theme = data.get('source', {}).get('theme', '')
            df.at[index, 'theme'] = theme
    else:
        print(f'JSON file {json_file_path} does not exist')


# 将更新后的 DataFrame 写回到新的 CSV 文件
new_csvpath = osp.join(dirpath, '..', 'TestLog', f'incorrect_samples_{dataset}_new_with_cnt.csv')
df.to_csv(new_csvpath, index=False, encoding='utf-8-sig')

print(f'Updated CSV file saved to {new_csvpath}')