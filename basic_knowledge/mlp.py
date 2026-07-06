import numpy as np


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.lr = lr
        # 权重初始化, Xavier 简单初始化.
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros(1, hidden_dim)
        self.W2 = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b2 = np.zeros(1, output_dim)

    def forward(self, X):
        """保存中间值用于反向传播."""
        self.Z1 = np.dot(X, self.W1) + self.b1

    def backward():
        pass

    def train_step():
        pass
