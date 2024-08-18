import math

import numpy as np

import agrad.functional as F
import agrad.nn as nn
from agrad import Module, Tensor, ModuleList
from agrad.nn.norm import rmsnorm
from agrad.utils import ones, random

params = {
    "dim": 256,
    "n_layers": 4,
    "n_heads": 2,
    "vocab_size": 32000,
    "eps": 1e-6,
    "ctx_len": 2048,
    "dropout": 0.0,
    "hidden_dim": 8640,
}


class RoPE(Module):
    def __init__(self, params):
        self.dim = params["dim"] // params["n_heads"]
        self.ctx_len = params["ctx_len"]

    @staticmethod
    def build_cs_cache(dim, ctx_len):
        theta = np.power(10000, -2 * (np.arange(dim // 2)) / dim)
        seq = np.arange(ctx_len)
        seq_theta = np.outer(seq, theta)
        ot = np.cos(seq_theta), np.sin(seq_theta)
        return Tensor(ot[0], req_grad=False), Tensor(ot[1], req_grad=False)

    def forward(self, x: "Tensor", c: "Tensor", s: "Tensor"):
        """
        Expects x to be of shape (B, T, n_heads, dim)
        """
        T = x.shape[1]
        xs = x.reshape((*x.shape[:-1], self.dim // 2, 2))
        c, s = c[:T].reshape((1, T, 1, self.dim // 2)), s[:T].reshape(
            (1, T, 1, self.dim // 2)
        )
        return F.stack(
            [xs[..., 0] * c - xs[..., 1] * s, xs[..., 1] * c + xs[..., 0] * s], axis=-1
        ).reshape(x.shape)


class Attention(Module):
    def __init__(self, params) -> None:
        self.dim = params["dim"]
        self.n_heads = params["n_heads"]
        self.eps = params["eps"]
        self.ctx_len = params["ctx_len"]
        self.query = nn.Linear(params["dim"], params["dim"], mean=0.0, std=0.02)
        self.key = nn.Linear(params["dim"], params["dim"], mean=0.0, std=0.02)
        self.value = nn.Linear(params["dim"], params["dim"], mean=0.0, std=0.02)
        self.o = nn.Linear(params["dim"], params["dim"], mean=0.0, std=0.02)
        self.rope = RoPE(params)

    def forward(self, x, mask, c, s):
        B, T, C = x.shape
        qkv = self.query(x), self.key(x), self.value(x)
        q, k, v = [i.reshape((B, T, self.n_heads, C // self.n_heads)) for i in qkv]
        q = self.rope.forward(q, c, s)
        k = self.rope.forward(k, c, s)
        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))
        scores = (q @ k.transpose((0, 1, 3, 2))) / math.sqrt(C // self.n_heads)
        scores = scores + mask[:, :, :T, :T]
        scores = F.softmax(scores)
        scores = (scores @ v).transpose((0, 2, 1, 3)).reshape((B, T, C))
        return self.o(scores)


class MLP(Module):
    def __init__(self, params):
        self.dim = params["dim"]
        self.hidden = params["hidden_dim"]
        self.w1 = nn.Linear(self.dim, self.hidden, mean=0.0, std=0.01)  # gate_proj
        self.w2 = nn.Linear(self.hidden, self.dim, mean=0.0, std=0.01)  # down_proj
        self.w3 = nn.Linear(self.dim, self.hidden, mean=0.0, std=0.01)  # up_proj

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(Module):
    def __init__(self, params, filename={}):
        self.post_attn_norm = rmsnorm(params["dim"])
        self.post_inpt = rmsnorm(params["dim"])
        self.attn = Attention(params)
        self.mlp = MLP(params)
        self.params = params

    def forward(self, x, mask, c, s):
        x = x + self.attn(self.post_inpt(x), mask, c, s)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class llama(Module):
    def __init__(self, params):

        self.w_embed = nn.Embedding(
            params["vocab_size"], params["dim"], mean=0.0, std=0.02
        )
        self.layers = ModuleList(
            [TransformerBlock(params) for _ in range(params["n_layers"])]
        )
        self.norm = rmsnorm(params["dim"])
        self.lm_head = nn.Linear(params["dim"], params["vocab_size"], bias=False)
        self.c, self.s = RoPE.build_cs_cache(
            params["dim"] // params["n_heads"], params["ctx_len"]
        )
        self.mask = Tensor(
            (-1 / np.tril(np.ones((params["ctx_len"], params["ctx_len"]))) + 1)[
                np.newaxis, np.newaxis
            ],
            req_grad=False,
        )
        self.params = params

    def forward(self, x):
        y = self.w_embed(x)
        for layer in self.layers:
            y = layer(y, self.mask, self.c, self.s)
        y = self.norm(y)
        return self.lm_head(y)

    def generate(self, x, max_new=10):
        for _ in range(max_new):
            if x.shape[1] < params["ctx_len"]:
                x_c = x
            else:
                x_c = x[:, -params["ctx_len"] :]
            p = self.forward(x_c)
            new_tok = p[:, -1, :]
            probs = F.softmax(new_tok)
            nxt = np.argmax(np.random.multinomial(1, probs[0]), keepdims=True)[
                np.newaxis
            ]
            x = np.concatenate((x, nxt), axis=-1)
        return x


model = llama(params)
yhat = model(ones((1, 1), dtype=int))
yhat.backward(np.ones_like(yhat, dtype=float))
