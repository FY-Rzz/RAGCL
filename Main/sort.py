# -*- coding: utf-8 -*-
# @Time    : 20241125
# @Author  : xiezw
# @Email   :
# @File    : sort.py
# @Software: vscode
# @Note    : 两种数据集划分函数
import os
import json
import random
import os.path as osp
from Main.utils import write_post, dataset_makedirs
from sklearn.model_selection import KFold

# 数据集划分
def sort_dataset(label_source_path, label_dataset_path, k_shot=10000, split='622'):
    if split == '622':
        train_split = 0.6
        test_split = 0.8
    elif split == '802':
        train_split = 0.8
        test_split = 0.8

    train_path, val_path, test_path = dataset_makedirs(label_dataset_path)

    label_file_paths = []
    for filename in os.listdir(label_source_path):
        label_file_paths.append(os.path.join(label_source_path, filename))

    all_post = []
    for filepath in label_file_paths:
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        all_post.append((post['source']['tweet id'], post))

    random.seed(1234)
    random.shuffle(all_post)
    train_post = []

    multi_class = False
    for post in all_post:
        if post[1]['source']['label'] == 2 or post[1]['source']['label'] == 3:
            multi_class = True

    num0 = 0
    num1 = 0
    num2 = 0
    num3 = 0
    for post in all_post[:int(len(all_post) * train_split)]:
        if post[1]['source']['label'] == 0 and num0 != k_shot:
            train_post.append(post)
            num0 += 1
        if post[1]['source']['label'] == 1 and num1 != k_shot:
            train_post.append(post)
            num1 += 1
        if post[1]['source']['label'] == 2 and num2 != k_shot:
            train_post.append(post)
            num2 += 1
        if post[1]['source']['label'] == 3 and num3 != k_shot:
            train_post.append(post)
            num3 += 1
        if multi_class:
            if num0 == k_shot and num1 == k_shot and num2 == k_shot and num3 == k_shot:
                break
        else:
            if num0 == k_shot and num1 == k_shot:
                break
    if split == '622':
        val_post = all_post[int(len(all_post) * train_split):int(len(all_post) * test_split)]
        test_post = all_post[int(len(all_post) * test_split):]
    elif split == '802':
        val_post = all_post[-1:]
        test_post = all_post[int(len(all_post) * test_split):]
    write_post(train_post, train_path)
    write_post(val_post, val_path)
    write_post(test_post, test_path)


# 5折划分
def sort_5fold_dataset(label_source_path, label_dataset_path, k_shot=4000, n_splits=5):
    # 创建存储5折数据集的文件夹
    fold_paths = [osp.join(label_dataset_path, f'fold_{i}') for i in range(n_splits)]
    for path in fold_paths:
        os.makedirs(path, exist_ok=True)

    label_file_paths = []
    for filename in os.listdir(label_source_path):
        label_file_paths.append(os.path.join(label_source_path, filename))

    all_post = []
    for filepath in label_file_paths:
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        all_post.append((post['source']['tweet id'], post))

    random.seed(1234)
    random.shuffle(all_post)

    # 定义5折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, test_index) in enumerate(kf.split(all_post)):
        train_post = [all_post[i] for i in train_index]
        test_post = [all_post[i] for i in test_index]

        # 控制每个类别的样本数量
        train_post_limited = []
        num0, num1, num2, num3 = 0, 0, 0, 0
        for post in train_post:
            label = post[1]['source']['label']
            if label == 0 and num0 < k_shot:
                train_post_limited.append(post)
                num0 += 1
            elif label == 1 and num1 < k_shot:
                train_post_limited.append(post)
                num1 += 1
            elif label == 2 and num2 < k_shot:
                train_post_limited.append(post)
                num2 += 1
            elif label == 3 and num3 < k_shot:
                train_post_limited.append(post)
                num3 += 1
            if num0 == k_shot and num1 == k_shot and num2 == k_shot and num3 == k_shot:
                break

        # 写入当前折的数据
        fold_train_path = osp.join(fold_paths[fold], 'train')
        fold_test_path = osp.join(fold_paths[fold], 'test')

        os.makedirs(fold_train_path, exist_ok=True)
        os.makedirs(fold_test_path, exist_ok=True)
        os.makedirs(osp.join(fold_train_path,'raw'), exist_ok=True)
        os.makedirs(osp.join(fold_train_path,'processed'), exist_ok=True)
        os.makedirs(osp.join(fold_test_path,'raw'), exist_ok=True)
        os.makedirs(osp.join(fold_test_path,'processed'), exist_ok=True)


        write_post(train_post_limited, osp.join(fold_train_path,'raw'))
        write_post(test_post, osp.join(fold_test_path,'raw'))