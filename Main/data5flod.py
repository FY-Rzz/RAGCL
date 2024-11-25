# -*- coding: utf-8 -*-
# @Time    : 2024/11/25
# @Author  : xiezw
# @Email   :
# @File    : data5flod.py
# @Software: vscode
# @Note    : 对数据集进行5折交叉验证

import os
import json
import random
from sklearn.model_selection import KFold

# 获取文件夹中所有JSON文件的文件名
data_dir = './data/json_files'
file_list = [f for f in os.listdir(data_dir) if f.endswith('.json')]

# 打乱文件列表
random.shuffle(file_list)

# 定义5折交叉验证
kf = KFold(n_splits=5)

# 准备存储每折的数据
folds = []

# 划分数据集
for train_index, test_index in kf.split(file_list):
    train_files = [file_list[i] for i in train_index]
    test_files = [file_list[i] for i in test_index]
    folds.append((train_files, test_files))

# 示例：加载第一个折的训练集和测试集数据
def load_data(file_list, data_dir):
    data = []
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            data.append(json_data)
    return data

# 加载第一个折的数据
train_files, test_files = folds[0]
train_data = load_data(train_files, data_dir)
test_data = load_data(test_files, data_dir)

# 现在可以使用 train_data 和 test_data 进行训练和测试

