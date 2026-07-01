import numpy as np


class LayerNorm:
    def __init__(self, hidden_dim, eps=1e-5):
        self.eps = eps
        # 可学习参数 gamma, beta
        self.gamma = np.ones(hidden_dim)
        self.beta = np.zeros(hidden_dim)

    def forward(self, x):
        # x shape: [B, L, D] 或者 [N, D]
        # 1. 计算均值, 沿最后一维求平均, 保持维度方便广播
        mean = np.mean(x, axis=-1, keepdims=True)
        # 2. 方差
        var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
        # 3. 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        # 4. 缩放 + 偏移
        out = x_norm * self.gamma + self.beta
        return out


if __name__ == "__main__":
    B, L, D = 2, 3, 4
    x = np.random.randn(B, L, D)
    ln = LayerNorm(hidden_dim=D)
    res = ln.forward(x)
    print("output shape:", res.shape)  # (2,3,4)
