import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, : x.size(1)]


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
        return x.transpose(1, 2).contiguous().view(B, L, h * dk)

    def forward(self, q, k, v, mask=None):
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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = src
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)
        x = x + self.ffn(self.norm2(x))
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

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = tgt
        x = x + self.masked_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        x = x + self.cross_attn(self.norm2(x), memory, memory, memory_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self, src_vocab, tgt_vocab, d_model=512, n_heads=8, d_ff=2048, n_layers=6
    ):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pe = PositionalEncoding(d_model)
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers)
        self.proj = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src_emb = self.pe(self.src_emb(src) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_mask)
        tgt_emb = self.pe(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        output = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)
        logits = self.proj(output)
        return logits


def generate_subsequent_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 1


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
print(logits.shape)
