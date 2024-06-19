import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder, MHA_FF

from torch.nn.parameter import Parameter

import torch.nn.functional as F
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(10, in_features, out_features))  # 权重矩阵

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters_xavier()

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)  # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):
        x = x.permute(0, 2, 1)
        self.weight = Parameter(torch.FloatTensor(x.size()[0], self.in_features, self.out_features).to('cuda'))
        self.reset_parameters_xavier()
        support = torch.bmm(x, self.weight)
        output = torch.bmm(adj, support)

        if self.bias is not None:
            return (output + self.bias).permute(0, 2, 1)
        else:
            return output.permute(0, 2, 1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_size, state_size):
        super(GraphConvolutionLayer, self).__init__()
        self.in_size = in_size
        self.state_size = state_size

        self.condense = nn.Conv1d(in_size, state_size, 1)
        self.condense_norm = nn.BatchNorm1d(state_size)

        self.gcn_forward = GraphConvolution(in_size, state_size)
        self.gcn_backward = GraphConvolution(in_size, state_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.conv1d = nn.Conv1d(3 * state_size, in_size, 1, bias=False)
        self.norm = nn.BatchNorm1d(in_size)

        self.test_conv = nn.Conv1d(state_size, in_size, 1, bias=False)

    def forward(self, x, fw_A, bw_A):
        states = x
        condensed_message = self.relu(self.condense_norm(self.condense(x)))
        fw_message = self.relu(self.gcn_forward(x, fw_A))
        bw_message = self.relu(self.gcn_backward(x, bw_A))
        update = torch.cat((condensed_message, fw_message, bw_message), dim=1)
        x = self.norm(self.conv1d(update))
        x = self.relu(x + states)

        return x


class GCN(nn.Module):
    def __init__(self, in_size, state_size):
        # super(GCN_new, self).__init__()
        super(GCN, self).__init__()
        # in_size:1280, state_size:256
        self.gcn1 = GraphConvolutionLayer(in_size, state_size)
        self.gcn2 = GraphConvolutionLayer(in_size, state_size)
        self.gcn3 = GraphConvolutionLayer(in_size, state_size)

    def forward(self, states, fw_A, bw_A):
        # states: batch_size * feature_size(in_size) * number_classes
        states = states.permute(0, 2, 1)
        # states: batch_size * number_classes * feature_size(in_size)
        states = self.gcn1(states, fw_A, bw_A)
        states = self.gcn2(states, fw_A, bw_A)
        states = self.gcn3(states, fw_A, bw_A)

        return states.permute(0, 2, 1)


class DDL(nn.Module):
    def __init__(self, args, tokenizer, num_classes, fw_adj, bw_adj):
        super(DDL, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = TransformerModel(args, tokenizer)
        self.proj = nn.Linear(args.num_labels, args.d_vf)

        self._init_weight(self.proj)

        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        self.txt_encoder = TextEncoder(self.d_model, self.d_ff, self.num_layers, self.tgt_vocab, self.num_labels,
                                       self.h, self.dropout)
        self.update_mem = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.f_fusion = MHA_FF(self.d_model, self.d_ff, args.num_memory_heads, self.dropout)
        self.memory, self.mask = self.init_memory()

        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        self.classifier = nn.Linear(self.d_model, self.num_labels)
        self.embed_labels = nn.Linear(1, self.d_model)

        self.init_weight()

        # 设置模型的类别数
        self.num_classes = num_classes
        feat_size = 1280
        # 使用 ClsAttention 类定义一个分类器注意力机制
        self.cls_atten = ClsAttention(feat_size, num_classes)
        # 使用 GCN 类定义一个图卷积网络，输入特征大小为 feat_size，隐藏层大小为 256
        self.gcn = GCN(feat_size, 256)
        # 定义一个全连接层，输入大小为 feat_size，输出大小为 num_classes
        self.fc2 = nn.Linear(feat_size, num_classes)
        # 计算前向邻接矩阵 fw_adj 每行的和，并将其转换为对角矩阵 fw_D
        fw_D = torch.diag_embed(fw_adj.sum(dim=1))
        # 计算反向邻接矩阵 bw_adj 每行的和，并将其转换为对角矩阵 bw_D
        bw_D = torch.diag_embed(bw_adj.sum(dim=1))
        # 计算 fw_D 的逆平方根，并处理无穷大的情况
        inv_sqrt_fw_D = fw_D.pow(-0.5)
        inv_sqrt_fw_D[torch.isinf(inv_sqrt_fw_D)] = 0
        # 计算 bw_D 的逆平方根，并处理无穷大的情况
        inv_sqrt_bw_D = bw_D.pow(-0.5)
        inv_sqrt_bw_D[torch.isinf(inv_sqrt_bw_D)] = 0
        # 计算标准化的前向邻接矩阵 fw_A
        self.fw_A = inv_sqrt_fw_D.mm(fw_adj).mm(inv_sqrt_fw_D)
        # 计算标准化的反向邻接矩阵 bw_A
        self.bw_A = inv_sqrt_bw_D.mm(bw_adj).mm(inv_sqrt_bw_D)

    def init_weight(self):
        self._init_weight(self.linear_z)
        self._init_weight(self.get_mem)
        self._init_weight(self.linear_feat)
        self._init_weight(self.classifier)
        self._init_weight(self.embed_labels)

    def init_memory(self):
        memory = nn.Parameter(torch.eye(self.num_slots).unsqueeze(0))
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]
        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0
        return memory, mask

    def forward_iu_xray(self, images):
        patch_feats_0, global_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        patch_feats_1, global_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        global_feats = torch.mean(torch.stack([global_feats_0, global_feats_1]), dim=0)
        patch_feats = torch.cat((patch_feats_0, patch_feats_1), dim=1)

        out_labels = torch.mean(torch.stack([labels_0, labels_1]), dim=0)
        return patch_feats, global_feats, global_feats_0, global_feats_1, out_labels

    def forward_mimic_cxr(self, images):
        patch_feats, global_feats, out_labels = self.visual_extractor(images)
        return patch_feats, global_feats, out_labels


    def forward(self, images, targets=None, labels=None, mode='train'):
        bsz = images.shape[0]
        # 根据数据集类型选择不同的前向传播函数
        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        # 对数据集进行特征提取，得到注意力特征和池化特征
        # 使用图神经网络对图像进行分类，得到视觉标签
        if self.args.dataset_name == 'iu_xray':
            patch_feats, global_feats, global_feats_0, global_feats_1, _ = ve(images)
            node_states, vis_labels = self.cls_forward(images, global_feats_0, global_feats_1)
        else:
            patch_feats, global_feats, _ = ve(images)
            node_states, vis_labels = self.cls_forward_mimic(images, global_feats)

        # 在维度 1 上应用 softmax
        node_states = F.softmax(node_states, dim=1)
        # 图像特征通过线性层进行变换
        l_img = self.linear_z(node_states)

        # 获取记忆和掩码
        memory = self.get_mem(self.memory.to(images)).expand(bsz, -1, -1)
        mask = self.mask.to(images).expand(bsz, -1, -1)
        if mode == 'train':
            # 如果是训练模式，则对目标序列进行编码，得到文本特征 txt_feats 和文本标签 txt_labels，并更新记忆
            txt_feats, l_txt, _ = self.txt_encoder(targets)
            memory = self.update_mem(memory, txt_feats, mask)
        # 将视觉标签转换为嵌入向量
        embed_labels = self.embed_labels(vis_labels.unsqueeze(-1))
        # 跨模态融合特征
        cross_f = self.f_fusion(embed_labels, memory)
        full_feats = torch.cat((patch_feats, self.linear_feat(cross_f)), dim=1)

        if mode == 'train':
            # 如果是训练模式，则调用编码器-解码器模块进行前向传播，并返回输出、视觉标签、文本标签、图像特征和文本特征
            output = self.encoder_decoder(_, full_feats, targets, mode='forward')
            return output, vis_labels, l_img, l_txt
        elif mode == 'sample':
            # 如果是采样模式，则调用编码器-解码器模块进行采样，并返回输出和视觉标签
            output, _ = self.encoder_decoder(_, full_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError

    def cls_forward(self, images, g_feats_0, g_feats_1):
        batch_size = images[:, 0].size(0)
        # 将标准化的邻接矩阵复制扩展为与输入图像批量大小相同的维度
        fw_A = self.fw_A.repeat(batch_size, 1, 1)
        bw_A = self.bw_A.repeat(batch_size, 1, 1)

        # 计算图像特征的全局平均值，得到全局特征
        global_feats0 = g_feats_0.mean(dim=(2, 3))
        global_feats1 = g_feats_1.mean(dim=(2, 3))

        # 使用分类器注意力机制对图像特征进行加权融合，得到分类特征
        cls_feats0 = self.cls_atten(g_feats_0)
        cls_feats1 = self.cls_atten(g_feats_1)

        # 将全局特征和分类特征在特征维度上进行拼接，得到节点特征
        node_feats0 = torch.cat((global_feats0.unsqueeze(1), cls_feats0), dim=1)
        node_feats1 = torch.cat((global_feats1.unsqueeze(1), cls_feats1), dim=1)
        # 将节点特征进行连续化处理，以便进行图卷积操作
        node_feats0 = node_feats0.contiguous()
        node_feats1 = node_feats1.contiguous()
        # 使用图卷积网络对节点特征进行图卷积操作，得到节点状态
        node_states0 = self.gcn(node_feats0, fw_A, bw_A)
        node_states1 = self.gcn(node_feats1, fw_A, bw_A)
        # 计算节点状态的平均值，得到全局状态
        node_states = node_states0.mean(dim=1) + node_states1.mean(dim=1)
        # 将全局状态通过全连接层进行分类，得到分类结果
        vis_labels = self.fc2(node_states)

        return node_states, vis_labels

    def cls_forward_mimic(self, images, g_feats):
        # 前向传播函数，定义模型的计算逻辑

        batch_size = images.size(0)
        # 将标准化的邻接矩阵复制扩展为与输入图像批量大小相同的维度
        fw_A = self.fw_A.repeat(batch_size, 1, 1)
        bw_A = self.bw_A.repeat(batch_size, 1, 1)

        # 计算图像特征的全局平均值，得到全局特征
        global_feats = g_feats.mean(dim=(2, 3))
        # global_feats0 = cnn_feats0.mean(dim=(2, 3))
        # 使用分类器注意力机制对图像特征进行加权融合，得到分类特征
        cls_feats = self.cls_atten(g_feats)

        # 将全局特征和分类特征在特征维度上进行拼接，得到节点特征
        node_feats = torch.cat((global_feats.unsqueeze(1), cls_feats), dim=1)
        # 将节点特征进行连续化处理，以便进行图卷积操作
        node_feats = node_feats.contiguous()
        # 使用图卷积网络对节点特征进行图卷积操作，得到节点状态
        node_states = self.gcn(node_feats, fw_A, bw_A)
        # 计算节点状态的平均值，得到全局状态
        node_states = node_states.mean(dim=1) + node_states.mean(dim=1)
        # 将全局状态通过全连接层进行分类，得到分类结果
        vis_labels = self.fc2(node_states)

        return node_states, vis_labels

    @staticmethod
    def _init_weight(f):
        nn.init.kaiming_normal_(f.weight)
        f.bias.data.fill_(0)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class ClsAttention(nn.Module):
    def __init__(self, feat_size, num_classes):
        super().__init__()
        self.feat_size = feat_size
        self.num_classes = num_classes
        self.channel_w = nn.Conv2d(feat_size, num_classes, 1, bias=False)

    def forward(self, feats):
        # feats: batch size x feat size x H x W
        batch_size, feat_size, H, W = feats.size()
        att_maps = self.channel_w(feats)
        att_maps = torch.softmax(att_maps.view(batch_size, self.num_classes, -1), dim=2)
        feats_t = feats.view(batch_size, feat_size, H * W).permute(0, 2, 1)
        cls_feats = torch.bmm(att_maps, feats_t)
        return cls_feats
