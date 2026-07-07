import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mse_deriv(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.lr = lr
        # 权重初始化, Xavier 简单初始化.
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        """保存中间值用于反向传播."""
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y_true):
        """反向传播，计算所有梯度并更新权重"""
        batch_size = X.shape[0]
        # 输出层梯度 dL/dZ2
        dL_dA2 = mse_deriv(self.A2, y_true)
        dZ2 = dL_dA2 * sigmoid_deriv(self.A2)

        # W2, b2 梯度
        dW2 = np.dot(self.A1.T, dZ2) / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size

        # 隐藏层梯度 dL/dZ1
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_deriv(self.A1)

        # W1, b1 梯度
        dW1 = np.dot(X.T, dZ1) / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size

        # 梯度下降更新参数
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_step(self, X, y):
        pred = self.forward(X)
        loss = mse_loss(pred, y)
        self.backward(X, y)
        return loss


# ---------------------- 测试：拟合XOR异或（经典非线性任务） ----------------------
if __name__ == "__main__":
    # XOR数据集
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # 构建网络：输入2维，隐藏4神经元，输出1维
    model = MLP(input_dim=2, hidden_dim=4, output_dim=1, lr=0.8)

    # 训练循环
    epochs = 10000
    for i in range(epochs):
        loss = model.train_step(X, y)
        if i % 1000 == 0:
            print(f"Epoch {i:5d} | Loss: {loss:.4f}")

    # 推理预测
    pred = model.forward(X)
    print("\n预测结果: ")
    print(pred)
    print("真实标签: ")
    print(y)
