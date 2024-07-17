import warnings

warnings.simplefilter("ignore", UserWarning)

import logging
import torch
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import LADataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss

from config import opts
from models.ddl import DDL


def main():
    # 解析参数
    args = opts.parse_opt()
    logging.info(str(args))

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # 创建分词器
    tokenizer = Tokenizer(args)

    # 创建数据加载器
    train_dataloader = LADataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = LADataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = LADataLoader(args, tokenizer, split='test', shuffle=False)

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # 从文件中读取邻接矩阵，并将其转换为PyTorch张量。然后，创建了一个单位矩阵，并与邻接矩阵相加，用于构建前向和后向邻接矩阵。
    with open('data/14_nodes.txt', 'r') as matrix_file:
        adjacency_matrix = [[int(num) for num in line.split(', ')] for line in matrix_file]
    num_classes = 14
    fw_adj = torch.tensor(adjacency_matrix, dtype=torch.float, device=device)
    bw_adj = fw_adj.t()
    identity_matrix = torch.eye(num_classes + 1, device=device)
    fw_adj = fw_adj.add(identity_matrix)
    bw_adj = bw_adj.add(identity_matrix)

    # 构建模型架构
    model = DDL(args, tokenizer, num_classes, fw_adj, bw_adj).to(device)

    # 获取损失函数和评估指标的函数句柄
    criterion = compute_loss
    metrics = compute_scores

    # 构建优化器和学习率调度器
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # 构建训练器并开始训练
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    trainer.train()
    logging.info(str(args))


if __name__ == '__main__':
    main()
