# ============================================================
#  model.py — 작은 GPT를 처음부터 구현
# ------------------------------------------------------------
#  PyTorch는 "행렬 계산 + 자동 미분" 도구로만 쓴다.
#  어텐션, 트랜스포머 블록, 문장 생성은 전부 여기서 직접 짠다.
#
#  흐름:  글자 번호 → 임베딩(+위치) → [블록 × N] → 다음 글자 확률
#  블록:  x = x + 어텐션(정규화(x))   ← 앞 글자들을 참고
#         x = x + MLP(정규화(x))       ← 혼자 생각
# ============================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=6, n_embd=192, dropout=0.1):
        self.vocab_size = vocab_size  # 아는 글자 수
        self.block_size = block_size  # 한 번에 볼 수 있는 최대 글자 수(기억 길이)
        self.n_layer = n_layer        # 블록 층 수
        self.n_head = n_head          # 어텐션 머리 수
        self.n_embd = n_embd          # 글자 하나를 표현하는 숫자 개수
        self.dropout = dropout

    def to_dict(self):
        return dict(self.__dict__)


class SelfAttention(nn.Module):
    """각 글자가 '앞에 나온 글자들 중 누구를 얼마나 참고할지' 정한다."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # 질문(Q), 열쇠(K), 내용(V)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        # 미래 글자는 못 보게 가리는 삼각형 마스크
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, 머리, T, 머리크기)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if x.is_cuda:
            # GPU에서는 아래와 똑같은 계산을 한 번에 빠르게 해 주는 함수를 쓴다
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.drop.p if self.training else 0.0)
        else:
            # 점수 = Q·K / √d   (얼마나 관련 있나)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)       # 점수 → 비율
            att = self.drop(att)
            y = att @ v                         # 비율만큼 내용을 섞는다
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.out = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.out(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # 글자 → 벡터
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # 위치 → 벡터
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # 벡터 → 다음 글자 점수
        self.head.weight = self.tok_emb.weight  # 입력/출력 사전 공유 (파라미터 절약)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            # 정답 다음 글자를 얼마나 잘 맞혔나 (낮을수록 좋음)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20, stop_id=None):
        """한 글자씩 뽑아서 이어 붙인다."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]          # 기억 길이만큼만 본다
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature             # 마지막 위치 = 다음 글자
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")     # 상위 k개만 후보
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)                   # 확률대로 하나 뽑기
            idx = torch.cat([idx, nxt], dim=1)
            if stop_id is not None and nxt.item() == stop_id:
                break
        return idx
