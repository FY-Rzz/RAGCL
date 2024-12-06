# -*- coding: utf-8 -*-
# @Time    : 20241205
# @Author  : xiezw
# @Email   :
# @File    : nodep.py
# @Software: vscode
# @Note    : 数据分析可视化

import pandas as pd
import json
import os
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns


def plot_theme():
    dataset = 'DRWV3'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 数据路径
    dirpath = osp.dirname(osp.abspath(__file__))
    csvpath = osp.join(dirpath, '..', 'TestLog', f'correct_samples_{dataset}_new_with_cnt.csv')

    # 读取 CSV 文件
    df = pd.read_csv(csvpath)

    # 统计 'theme' 属性值的分布
    theme_counts = df['theme'].value_counts()

    # 打印 'theme' 属性值的分布
    print(theme_counts)

    # 可视化 'theme' 属性值的分布
    plt.figure(figsize=(10, 6))
    #sns.barplot(x=theme_counts.index, y=theme_counts.values, palette='viridis')
    sns.countplot(data=df, x='theme', hue='true_label', palette='viridis')
    plt.xlabel('Theme')
    plt.ylabel('Count')
    plt.title('主题分布')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图表
    plot_path = osp.join(dirpath, '..', 'TestLog', 'theme_distribution.png')
    plt.savefig(plot_path)

    # 显示图表
    plt.show()

def cnt_theme():
    dataset = 'DRWV3'
    dirpath = osp.dirname(osp.abspath(__file__))
    csvpath1 = osp.join(dirpath, '..', 'TestLog', f'correct_samples_{dataset}_new_with_cnt.csv')
    csvpath2 = osp.join(dirpath, '..', 'TestLog', f'incorrect_samples_{dataset}_new_with_cnt.csv')
    
    # 读取两个 CSV 文件
    df1 = pd.read_csv(csvpath1)
    df2 = pd.read_csv(csvpath2)    
    # 统计每个 CSV 文件中 'theme' 属性值的数量
    theme_counts1 = df1['theme'].value_counts()
    theme_counts2 = df2['theme'].value_counts()

    # 计算相同 'theme' 属性值的数据条目数量比值
    theme_ratios = {}
    for theme in theme_counts1.index:
        if theme in theme_counts2:
            ratio = theme_counts1[theme] / (theme_counts2[theme] + theme_counts1[theme])
            theme_ratios[theme] = ratio

    # 打印比值
    for theme, ratio in theme_ratios.items():
        print(f"Theme: {theme}, Ratio: {ratio:.2f}")

if __name__ == '__main__':
    #plot_theme()
    cnt_theme()

