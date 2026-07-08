import numpy as np


class SGDRegressor:
    def __init__(self, lr=0.01, epochs=1000, random_state=42):
        self.lr = lr  # 学习率
        self.epochs = epochs  # 迭代轮数
        self.random_state = random_state
        self.w = None  # 权重
        self.b = None  # 偏置

    def fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        np.random.seed(self.random_state)
        n_samples, n_feats = X.shape

        # 初始化参数
        self.w = np.zeros(n_feats)
        self.b = 0.0

        for epoch in range(self.epochs):
            # 1. 打乱数据集(SGD随机采样)
            shuffle_idx = np.random.permutation(n_samples)
            X_shuffle = X[shuffle_idx]
            y_shuffle = y[shuffle_idx]

            # 2. 逐样本更新参数(标准SGD，每次1个样本)
            for xi, yi in zip(X_shuffle, y_shuffle):
                # 预测
                y_pred = np.dot(xi, self.w) + self.b
                # 梯度
                dw = 2 * (y_pred - yi) * xi
                db = 2 * (y_pred - yi)
                # 参数更新
                self.w -= self.lr * dw
                self.b -= self.lr * db

            # 每100轮打印损失
            if epoch % 100 == 0:
                total_pred = np.dot(X, self.w) + self.b
                loss = np.mean((total_pred - y) ** 2)
                print(f"Epoch {epoch:4d} | MSE Loss: {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# ---------------------- 测试 ----------------------
if __name__ == "__main__":
    # 生成模拟数据集 y = 3x1 + 2x2 + 4 + noise
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + 4 + np.random.randn(1000) * 0.1

    # 训练SGD
    model = SGDRegressor(lr=0.02, epochs=800)
    model.fit(X, y)

    print("\nLearned weight w:", model.w)
    print("Learned bias b:", model.b)

    # 预测测试样本
    test_x = np.array([[1, 1], [2, 3]])
    print("Predictions:", model.predict(test_x))
