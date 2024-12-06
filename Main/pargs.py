import argparse

# 解析命令行参数
def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='DRWeiboV3')
    parser.add_argument('--unsup_dataset', type=str, default='UWeiboV1')
    parser.add_argument('--tokenize_mode', type=str, default='naive')

    parser.add_argument('--vector_size', type=int, help='word embedding size', default=200)
    parser.add_argument('--unsup_train_size', type=int, help='word embedding unlabel data train size', default=20000)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--ft_runs', type=int, default=2)

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    # 622 or 802
    parser.add_argument('--split', type=str, default='802')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--unsup_bs_ratio', type=int, default=1)
    parser.add_argument('--undirected', type=str2bool, default=True)

    # ResGCN or BiGCN
    parser.add_argument('--model', type=str, default='ResGCN') # BiGCN , ResGCN
    parser.add_argument('--n_layers_feat', type=int, default=1)
    parser.add_argument('--n_layers_conv', type=int, default=3)
    parser.add_argument('--n_layers_fc', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--global_pool', type=str, default="sum")
    parser.add_argument('--skip_connection', type=str2bool, default=True)
    parser.add_argument('--res_branch', type=str, default="BNConvReLU")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--edge_norm', type=str2bool, default=True)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ft_lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--ft_epochs', type=int, default=25)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lamda', dest='lamda', type=float, default=0.001)

    # Node centrality metric can be chosen from "Degree", "PageRank", "Eigenvector", "Betweenness".
    parser.add_argument('--centrality', type=str, default="PageRank")
    # Augmentation can be chosen from "DropEdge,mean,0.3,0.7", "NodeDrop,0.3,0.7", "AttrMask,0.3,0.7",
    # or augmentation combination "DropEdge,mean,0.3,0.7||NodeDrop,0.3,0.7", "DropEdge,mean,0.3,0.7||AttrMask,0.3,0.7".
    # Str like "DropEdge,mean,0.3,0.7" means "AugName,[aggr,]p,threshold".
    parser.add_argument('--aug1', type=str, default="DropEdge,mean,0.2,0.7")
    parser.add_argument('--aug2', type=str, default="NodeDrop,0.2,0.7")

    parser.add_argument('--use_unlabel', type=str2bool, default=False)
    parser.add_argument('--use_unsup_loss', type=str2bool, default=True)

    parser.add_argument('--k', type=int, default=10000)

    args = parser.parse_args()
    return args

'''
数据集参数：
--dataset：数据集名称，默认为 Weibo。
--unsup_dataset：无监督数据集名称，默认为 UWeiboV1。
--tokenize_mode：分词模式，默认为 naive。

词嵌入参数：
--vector_size：词嵌入的维度，默认为 200。
--unsup_train_size：无标签数据的训练大小，默认为 20000。

运行参数：
--runs：运行次数，默认为 10。
--ft_runs：微调运行次数，默认为 10。

设备参数：
--cuda：是否使用 CUDA，默认为 True。
--gpu：使用的 GPU 设备编号，默认为 0。

数据集划分参数：
--split：数据集划分比例，默认为 802（80% 训练，20% 测试）。
--batch_size：批量大小，默认为 32。
--unsup_bs_ratio：无监督批量大小比例，默认为 1。
--undirected：是否使用无向图，默认为 True。

模型参数：
--model：模型名称，默认为 ResGCN。
--n_layers_feat：特征层的数量，默认为 1。
--n_layers_conv：卷积层的数量，默认为 3。
--n_layers_fc：全连接层的数量，默认为 2。
--hidden：隐藏层的维度，默认为 128。
--global_pool：全局池化方法，默认为 sum。
--skip_connection：是否使用跳跃连接，默认为 True。
--res_branch：残差分支类型，默认为 BNConvReLU。
--dropout：dropout 概率，默认为 0.3。
--edge_norm：是否归一化邻接矩阵，默认为 True。

优化参数：
--lr：学习率，默认为 0.001。
--ft_lr：微调学习率，默认为 0.001。
--epochs：训练轮数，默认为 100。
--ft_epochs：微调训练轮数，默认为 100。
--weight_decay：权重衰减，默认为 0。
--lamda：无监督损失的权重，默认为 0.001。

中心性度量参数：
--centrality：节点中心性度量方法，默认为 PageRank。

数据增强参数：
--aug1：第一种数据增强方法，默认为 DropEdge,mean,0.2,0.7。
--aug2：第二种数据增强方法，默认为 NodeDrop,0.2,0.7。

其他参数：
--use_unlabel：是否使用无标签数据，默认为 False。
--use_unsup_loss：是否使用无监督损失，默认为 True。
--k：每类样本数量，默认为 10000。
'''