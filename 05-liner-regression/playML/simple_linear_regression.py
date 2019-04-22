#!/usr/bin/env python3
# coding: utf-8

"""HotChoc --> leaosunday25@gmail.com"""

import numpy as np
from .metrics import r2_score


class SimpleLinearRegression1:
    def __init__(self):
        """初始化 SimpleLinearRegression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """根据训练数据集 x_train, y_train 训练 SimpleLinearRegression 模型"""
        assert x_train.ndim == 1, \
            'Simple Linear Regression can only solve single feature training data'
        assert len(x_train) == len(y_train), \
            'the size of x_train must be equal to the size of y_train'

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict: np.ndarray) -> np.ndarray:
        """给定待预测数据集 x_predict, 返回表示 x_predict 的结果向量"""
        assert x_predict.ndim == 1, \
            'Simple Linear Regression can only solve single feature training data'
        assert self.a_ is not None and self.b_ is not None, \
            'must fit before predict'

        return self.a_ * x_predict + self.b_

    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """根据测试数据集 x_test 和 y_test 确定当前模型的准确度"""

        return r2_score(y_test, self.predict(x_test))

    def __repr__(self):
        return f'SimpleLinearRegression1(a_={self.a_}, b_={self.b_})'


class SimpleLinearRegression2:
    def __init__(self):
        """初始化 SimpleLinearRegression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """根据训练数据集 x_train, y_train 训练 SimpleLinearRegression 模型"""
        assert x_train.ndim == 1, \
            'Simple Linear Regression can only solve single feature training data'
        assert len(x_train) == len(y_train), \
            'the size of x_train must be equal to the size of y_train'

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict: np.ndarray) -> np.ndarray:
        """给定待预测数据集 x_predict, 返回表示 x_predict 的结果向量"""
        assert x_predict.ndim == 1, \
            'Simple Linear Regression can only solve single feature training data'
        assert self.a_ is not None and self.b_ is not None, \
            'must fit before predict'

        return self.a_ * x_predict + self.b_

    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """根据测试数据集 x_test 和 y_test 确定当前模型的准确度"""

        return r2_score(y_test, self.predict(x_test))

    def __repr__(self):
        return f'SimpleLinearRegression1(a_={self.a_}, b_={self.b_})'

