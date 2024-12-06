# -*- coding: utf-8 -*-
# @Time    : 20241125
# @Author  : xiezw
# @Email   :
# @File    : means.py
# @Software: vscode
# @Note    : 聚类分析和可视化
import sys
import os
import os.path as osp
import warnings
import csv
import numpy as np  # 导入 numpy 库

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))
sys.path.append(osp.join(dirname, '..', 'Main'))

import torch
from torch_geometric.loader import DataLoader
from Main.dataset import TreeDataset
from Main.model import ResGCN_graphcl, BiGCN_graphcl
from Main.word2vec import Embedding
from Main.pargs import pargs

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.close()  # 关闭窗口

def plot_tsne(features, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette=sns.color_palette("hsv", len(set(labels))))
    plt.title(title)
    plt.show()
    plt.close()  # 关闭窗口

def plot_kmeans(features, labels, title):
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(features)
    clusters = kmeans.labels_
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=clusters, palette=sns.color_palette("hsv", len(set(clusters))))
    plt.title(title)
    plt.show()
    plt.close()  # 关闭窗口

def test_method_1():
    print("测试方法1启动！-----------")
    # 设置路径和参数
    Dataset = 'Weibo' # 'DRWeiboV3'  # 
    dirname = osp.dirname(osp.abspath(__file__))
    model_state_dict_path = osp.join(dirname, '..', 'Model', '2024-11-08 02-57-02_run_0_model_state_dict.pth')  # 修改为实际的模型路径
    label_dataset_path = osp.join(dirname, '..', 'Data', Dataset, 'dataset')  # 修改为实际的数据集路径
    #label_dataset_path = osp.join(dirname, '..', 'Data', 'Weibo', 'dataset')
    test_path = osp.join(label_dataset_path, 'test')
    lang = 'ch' # 'en'

    args = pargs()
    tokenize_mode = args.tokenize_mode  
    vector_size = args.vector_size
    centrality = args.centrality
    undirected = args.undirected
    batch_size = args.batch_size
    word_embedding = 'word2vec'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("path set done")

    # 加载词嵌入模型
    model_path = osp.join(dirname, '..', 'Model', f'w2v_Weibo_{tokenize_mode}_20000_{vector_size}.model')  # 修改为实际的词嵌入模型路径
    word2vec = Embedding(model_path, lang, tokenize_mode) 
    print("word2vec load done")

    # 加载测试数据集
    test_dataset = TreeDataset(test_path, word_embedding, word2vec, centrality, undirected)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("dataset load done")

    # 初始化模型
    num_classes = test_dataset.num_classes
    model = ResGCN_graphcl(dataset=test_dataset, num_classes=num_classes, hidden=args.hidden,  
                        num_feat_layers=args.n_layers_feat, num_conv_layers=args.n_layers_conv, num_fc_layers=args.n_layers_fc, gfn=False, collapse=False,
                        residual=args.skip_connection, res_branch=args.res_branch, global_pool=args.global_pool, dropout=args.dropout,
                        edge_norm=args.edge_norm).to(device)
    # model = BiGCN_graphcl(test_dataset.num_features, args.hidden, args.hidden, num_classes).to(device)

    # 加载模型状态字典
    model.load_state_dict(torch.load(model_state_dict_path))
    model.eval()  # 将模型设置为评估模式
    print("model load done")

    with torch.no_grad():
        y_true = []
        y_pred = []
        sum_num = 0
        incorrect_samples = []
        correct_samples = []
        all_features = []  # 初始化 all_features 为空列表
        all_labels = []
        for data in test_loader:
            data = data.to(device)
            pred = model(data)
            
            y_true += data.y.tolist()
            y_pred += pred.argmax(dim=1).tolist()
            sum_num += data.num_graphs

            # 记录所有样本的特征和标签
            all_features.append(data.x.cpu().numpy())
            all_labels += data.y.tolist()

            # 记录预测错误和正确的样本
            for i in range(data.num_graphs):
                if pred.argmax(dim=1)[i].item() != data.y[i].item():
                    incorrect_samples.append({
                        'true_label': data.y[i].item(),
                        'predicted_label': pred.argmax(dim=1)[i].item(),
                        'raw_id': str(data.raw_id[i]) if hasattr(data, 'raw_id') else None,
                        'raw_data' : data.raw_data[i] if hasattr(data, 'raw_data') else None
                    })
                else:
                    correct_samples.append({
                        'true_label': data.y[i].item(),
                        'predicted_label': pred.argmax(dim=1)[i].item(),
                        'raw_id': str(data.raw_id[i]) if hasattr(data, 'raw_id') else None,
                        'raw_data' : data.raw_data[i] if hasattr(data, 'raw_data') else None
                    })

        print("条目数：" + f"{sum_num}" + "\n")
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)

        print(f"错误样本数量: {len(incorrect_samples)}")

        # 计算准确率
        accuracy = (y_true == y_pred).sum().item() / y_true.size(0)
        print(f"准确率: {accuracy:.4f}")

        # 数据写入
        csv_file_path = osp.join(dirname, '..', 'TestLog', f"incorrect_samples_{Dataset}.csv")
        with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=['true_label', 'predicted_label', 'raw_id', 'raw_data'])
            writer.writeheader()
            for sample in incorrect_samples:
                writer.writerow(sample)
        print("incorrect_samples write done")

        # 绘制混淆矩阵
        classes = [str(i) for i in range(num_classes)]
        plot_confusion_matrix(y_true, y_pred, classes)
        print("draw done")

        # 可视化错误样本与正确样本的特征区别
        all_features = np.concatenate(all_features, axis=0)
        labels = ['correct' if y_true[i] == y_pred[i] else 'incorrect' for i in range(len(y_true))]
        assert len(all_features) == len(labels), "Features and labels must have the same length"
        
        # 使用 t-SNE 降维并可视化
        plot_tsne(all_features, labels, "t-SNE Visualization of Correct and Incorrect Samples")

        # 使用 K-means 聚类并可视化
        plot_kmeans(all_features, labels, "K-means Clustering of Features")

    print("test1 done")

if __name__ == '__main__':
    test_method_1()