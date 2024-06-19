# -*- coding: utf-8 -*-
# @Time    : 2021/3/22
# @Author  : Aspen Stars
# @Contact : aspenstars@qq.com
# @FileName: text_encoder.py
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np

from modules.Transformer import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Encoder, \
    EncoderLayer, Embeddings, SublayerConnection, clones


class TextEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_layers, tgt_vocab, num_labels=14, h=3, dropout=0.1):
        super(TextEncoder, self).__init__()
        # TODO:
        #  将eos,pad的index改为参数输入
        # eos索引，表示句子结束
        # pad索引，用于填充句子
        self.eos_idx = 0
        self.pad_idx = 0

        # 多头注意力机制
        attn = MultiHeadedAttention(h, d_model)
        # 前馈神经网络
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 位置编码
        position = PositionalEncoding(d_model, dropout)
        # 线性分类器，用于预测标签
        self.classifier = nn.Linear(d_model, num_labels)
        # 编码器
        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), num_layers)
        # 输入嵌入层
        self.src_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), position)

    def prepare_mask(self, seq):
        # 根据eos和pad索引创建掩码
        seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
        # bos索引，表示句子开始
        seq_mask[:, 0] = 1
        # 在倒数第二个维度上增加一个维度
        seq_mask = seq_mask.unsqueeze(-2)
        return seq_mask

    def forward(self, src):
        # 创建输入序列的掩码
        src_mask = self.prepare_mask(src)
        # 对输入序列进行编码
        feats = self.encoder(self.src_embed(src), src_mask)
        # 提取特征的第一个位置的输出
        pooled_output = feats[:, 0, :]
        # 使用线性分类器对特征进行标签预测
        labels = self.classifier(pooled_output)
        return feats, pooled_output, labels


class MHA_FF(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super(MHA_FF, self).__init__()
        # 多头注意力机制
        self.self_attn = MultiHeadedAttention(h, d_model)
        # 子层连接
        self.sublayer = SublayerConnection(d_model, dropout)

    def forward(self, x, feats, mask=None):
        # 在子层连接中应用注意力机制
        x = self.sublayer(x, lambda x: self.self_attn(x, feats, feats))
        return x
