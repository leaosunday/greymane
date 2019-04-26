#!/usr/bin/env python3
# coding: utf-8

"""HotChoc --> leaosunday25@gmail.com"""

import numpy as np
from .metrics import r2_score


class LinearRegression:
    def __init__(self):
        """初始化 Linear Regression 模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train: np.ndarray, y_train: np.ndarray):
        """根据训练数据集 X_train 和 y_train 训练 Linear Regression 模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            'the size of X_train must be equal to the size of y_train'

        X_b = np.hstack([np.ones(shape=(len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train: np.ndarray, y_train: np.ndarray, eta: float = 0.01, n_iters: int = 1e4):
        """根据训练数据集 X_train 和 y_train, 使用梯度下降法训练 Linear Regression 模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            'the size of X_train must be equal to the size of y_train'

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b) # vectorize

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

        return self

    def predict(self, X_predict: np.ndarray) -> np.ndarray:
        """给定待预测数据集 X_predict, 返回表示 X_predict 的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            'must fit before predict'
        assert X_predict.shape[1] == len(self.coef_), \
            'the feature number of X_predict must be equal to X_train'

        X_b = np.hstack([np.ones(shape=(len(X_predict), 1)), X_predict])

        return X_b.dot(self._theta)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        return r2_score(y_test, self.predict(X_test))

    def __repr__(self):
        return f'LinearRegression(coef_={self.coef_}, intercept_={self.intercept_})'
