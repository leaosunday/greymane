#!/usr/bin/env python3
# coding: utf-8

"""HotChoc --> leaosunday25@gmail.com"""

import numpy as np


class StandardScaler():
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        """根据训练数据集 X 获得数据的均值和标准差"""
        assert X.ndim == 2, 'The dimension of X must be 2'

        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """将 X 根据均值和标准差进行标准归一化处理"""
        assert X.ndim == 2, 'The dimension of X must be 2'
        assert self.mean_ is not None and self.scale_ is not None, \
            'must fit before transform'
        assert X.shape[1] == self.mean_.shape[0], \
            'the feature number of X must be equal to mean_ and std_'

        return (X - self.mean_) / self.scale_
