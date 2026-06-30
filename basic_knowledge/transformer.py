import torch
import torch.nn as nn
import math

# Mask 工具 -> PositionalEncoding -> MultiHeadAttention -> FFN -> EncoderLayer -> DecoderLayer -> Encoder/Decoder 堆叠 -> 总 Transformer.


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        pos = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # [d_modle // 2]
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # 注册一个不需要梯度更新、但随模型移动(CPU/GPU)、会保存进 state_dict 的张量.

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


def generate_subsequent_mask(seq_len):
    # 解码器上三角mask，掩盖未来token
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 1  # True代表mask掉


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    # 缩放点积除以√dk: 防止维度大时内积爆炸, softmax 梯度消失.
    def scaled_dot_product(self, q, k, v, mask=None):
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_score.masked_fill_(mask, -1e9)
        attn_weight = torch.softmax(attn_score, dim=-1)
        out = torch.matmul(attn_weight, v)
        return out, attn_weight

    def split_head(self, x):
        # [B, L, d_model] -> [B, n_head, L, d_k]
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def concat_head(self, x):
        # [B, n_head, L, d_k] -> [B, L, d_model]
        B, h, L, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h*dk)

    # 多头注意力: 多组独立 QKV, 捕捉不同语义依赖.
    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.split_head(self.w_q(q))
        k = self.split_head(self.w_k(k))
        v = self.split_head(self.w_v(v))
        out, attn = self.scaled_dot_product(q, k, v, mask)
        out = self.concat_head(out)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        # LayerNorm 是对某个样本进行 Normalization.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    # Pre-LN vs Post-LN: 代码用 Pre-LN, 收敛更快, 现在大模型通用.
    def forward(self, x, src_mask=None):
        # 多头自注意力, 残差
        attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)
        x = x + attn_out
        # FFN残差
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    # 解码器两个注意力: 自注意力屏蔽未来 Token, 交叉注意力读取源序列信息.
    def forward(self, x, enc_out, tgt_mask=None, src_tgt_mask=None):
        # 1. 解码器自注意力
        x1 = self.masked_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        x = x + x1
        # 2. 交叉注意力
        x2 = self.cross_attn(self.norm2(x), enc_out, enc_out, src_tgt_mask)
        x = x + x2
        # 3. FFN
        x3 = self.ffn(self.norm3(x))
        x = x + x3
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model) # 每层残差直接叠加无归一化, 多层堆叠后向量幅值持续累积变大, 所以需要最后加一个 norm.

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, cross_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_heads=8, d_ff=2048, n_layers=6):
        super().__init__()
        self.d_model = d_model
        # 嵌入层 + 位置编码
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        # 位置编码: Transformer 无时序感知, 正弦 PE 提供相对位置信息.
        self.pe = PositionalEncoding(d_model)
        # 编解码器
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers)
        # 输出映射
        self.proj = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        # 编码流程
        src_emb = self.pe(self.src_emb(src) * math.sqrt(self.d_model))
        enc_out = self.encoder(src_emb, src_mask)
        # 解码流程
        tgt_emb = self.pe(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask, cross_mask)
        # 预测token分布
        logits = self.proj(dec_out)
        return logits


# 测试超参
src_vocab = 1000
tgt_vocab = 1000
batch_size = 2
src_len = 10
tgt_len = 8

model = Transformer(src_vocab, tgt_vocab)
src = torch.randint(0, src_vocab, (batch_size, src_len))
tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))
tgt_mask = generate_subsequent_mask(tgt_len)

logits = model(src, tgt, tgt_mask=tgt_mask)
print(logits.shape)  # [2,8,1000] 输出正确
