# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 11:07
# @Author  : Fisher
# @File    : GlobalPointer.py
# @Software: PyCharm
# @Desc    : GlobalPointer

"""
GlobalPointer:
    1、类似一种span矩阵标注的方式，矩阵的行是句子i-n，列为句子i-n
    2、矩阵i、j处为1表示以列j为开始，以行i为结束的片段是要抽取的目标
"""

import torch
import torch.nn as nn


# 增加相对位置编码
class SinusoidalPositionEmbedding(nn.Module):
    """ 定义Sin-Cos位置Embedding """

    def __init__(
            self, output_dim, merge_mode = 'add', custom_position_ids = False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        input_shape = inputs.shape
        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim = -1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


class GlobalPointer(nn.Module):
    """ 全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """
    pass
