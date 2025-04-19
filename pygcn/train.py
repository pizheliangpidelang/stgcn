from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, normalize
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0015,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1], #输入的特征数量为1，即1个点的流量
            nhid=args.hidden,
            nclass=1,#改成输出层数量，30个节点
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.mse_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.mse_loss(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    # 假设你的测试数据文件名为 'test_data.xlsx'
    test_data = pd.read_excel(r"D:\个人资料\课题组\海工\pygcn\test1.xlsx")

    # 将需要的列提取为特征
    feature1 = test_data[['distance','volume']].values[:2784]  # 替换为你的列名
    label1 = test_data[['label']].values[:2784]
    # 如果需要转换为 Tensor
    feature1 = sp.csr_matrix(feature1, dtype=np.float32)  # 转换为稀疏矩阵

    # 进行与之前类似的标准化
    feature_data = normalize(feature1)

    # 转换为PyTorch的张量格式
    feature1 = torch.FloatTensor(feature_data.todense())
    label1 = torch.FloatTensor(label1)  # 同样将标签转换为2D tensor (n, 1)

    model.eval()
    output = model(feature1, adj)
    print(output)
    loss_test = F.mse_loss(output, label1)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
