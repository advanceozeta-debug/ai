# ============================================================
#  big_prepare.py — 큰 오미크론용 데이터 준비 (Colab에서 실행)
# ------------------------------------------------------------
#  1) 한국어 위키백과를 원하는 글자 수만큼 받기 (기본 8200만 글자)
#  2) 기존 데이터(영화 리뷰 + 일상 대화 + 성격)와 합쳐 약 9000만 글자
#  3) "단어 조각" 사전(BPE) 만들기: 자주 붙어 나오는 글자들을 한 조각으로 묶음
#     예) "안녕하세요" → ["▁안녕", "하세요"]  글자 5개가 조각 2개가 됨
#  4) 모든 글을 조각 번호로 바꿔서 data/ 폴더에 저장
#
#  실행: python big_prepare.py --wiki-chars 82000000 --vocab 16000
#  빠른 시험: python big_prepare.py --wiki-chars 0 --vocab 4000
# ============================================================
import argparse
import os
import random
import re
import subprocess
import time

import numpy as np
from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers, trainers

ap = argparse.ArgumentParser()
ap.add_argument("--wiki-chars", type=int, default=82_000_000, help="위키백과에서 읽을 글자 수")
ap.add_argument("--vocab", type=int, default=16000, help="단어 조각 개수")
ap.add_argument("--out", default="data")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
random.seed(0)

# ---------- 1. 기존 데이터 (리뷰 + 일상 대화 + 성격) ----------
if not (os.path.exists("pretrain.txt") and os.path.exists("dialog.txt")):
    subprocess.run(["python", "prepare_korean.py"], check=True)
small_docs = open("pretrain.txt", encoding="utf-8").read().split("\n")
dialog_text = open("dialog.txt", encoding="utf-8").read()


# ---------- 2. 위키백과 ----------
def clean_wiki(text):
    out = []
    for line in text.split("\n"):
        line = re.sub(r"[^\S\n]+", " ", line).strip()
        if len(line) < 20 or not re.search("[가-힣]", line):
            continue                      # 제목, 목록 조각, 빈 줄은 버림
        out.append(line)
    return "\n".join(out)


wiki_docs = []
if args.wiki_chars > 0:
    from datasets import load_dataset
    print(f"위키백과 받는 중... 목표 {args.wiki_chars:,}글자", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
    total, t0 = 0, time.time()
    for row in ds:
        doc = clean_wiki(row["text"])
        if len(doc) < 100:
            continue
        wiki_docs.append(doc)
        total += len(doc)
        if len(wiki_docs) % 20000 == 0:
            print(f"  문서 {len(wiki_docs):,}개 | {total:,}글자 | {time.time()-t0:.0f}초", flush=True)
        if total >= args.wiki_chars:
            break
    print(f"위키백과 문서 {len(wiki_docs):,}개, {total:,}글자", flush=True)

docs = wiki_docs + small_docs
random.shuffle(docs)
n_chars = sum(map(len, docs)) + len(dialog_text)
print(f"전체 학습 글: {n_chars:,}글자", flush=True)

# ---------- 3. 단어 조각 사전 (BPE) ----------
# 띄어쓰기는 ▁ 로 바꾸고, 줄바꿈은 항상 혼자 한 조각이 되게 한다
# (줄바꿈 = 오미크론이 말을 끝내는 신호라서)
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.normalizer = normalizers.Replace(" ", "▁")
tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\n|▁?[^▁\n]+|▁+"), behavior="isolated")
trainer = trainers.BpeTrainer(vocab_size=args.vocab, min_frequency=2, special_tokens=["<unk>"],
                              limit_alphabet=4000, initial_alphabet=["\n", "▁"], show_progress=True)


def text_iter():
    for d in docs:
        yield d
    for d in dialog_text.split("\n\n"):
        yield d


print("단어 조각 사전 만드는 중...", flush=True)
tok.train_from_iterator(text_iter(), trainer=trainer, length=len(docs))
tok.save(os.path.join(args.out, "tokenizer.json"))
NL = tok.token_to_id("\n")
print(f"사전 크기 {tok.get_vocab_size():,}개 | 줄바꿈 번호 {NL}", flush=True)


# ---------- 4. 번호로 바꿔 저장 ----------
def encode_docs(texts, sep_newlines=1):
    parts, B, sep = [], 2000, [NL] * sep_newlines
    for i in range(0, len(texts), B):
        ids = []
        for e in tok.encode_batch(texts[i:i + B]):
            ids.extend(e.ids)
            ids.extend(sep)
        parts.append(np.array(ids, dtype=np.uint16))
    return np.concatenate(parts)


def save_split(name, arr, val_frac=0.01):
    n = int(len(arr) * (1 - val_frac))
    arr[:n].tofile(os.path.join(args.out, f"{name}_train.bin"))
    arr[n:].tofile(os.path.join(args.out, f"{name}_val.bin"))
    print(f"{name}: 조각 {len(arr):,}개 (학습 {n:,} / 검증 {len(arr)-n:,})", flush=True)


t0 = time.time()
pre = encode_docs(docs, sep_newlines=2)
save_split("pre", pre)
dlg_blocks = [b + "\n" if not b.endswith("\n") else b for b in dialog_text.split("\n\n") if b.strip()]
dlg = encode_docs(dlg_blocks, sep_newlines=1)
save_split("dlg", dlg, val_frac=0.03)
print(f"글자 {n_chars:,}개 → 조각 {len(pre)+len(dlg):,}개 (조각당 {n_chars/(len(pre)+len(dlg)):.2f}글자) | {time.time()-t0:.0f}초")
