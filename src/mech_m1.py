"""M1 / M1b: counterfactual teacher forcing.

The SAME story text is run under different secret contexts, so any difference in the residual
stream at story positions is caused by the secret in context and not by story content.

M1  : 40 secret-free stories (E1 `no_secret`) x 31 contexts (dont_reveal with each of the 15 curated
      words, each of 15 COCA words, and the no-secret prompt).
      Saves per (story, context): mean residual over story tokens at all 48 layers (fp16), position-
      binned residuals (32-token bins) at 12 layers, and final-layer log-probs of every word's tokens.
M1b : 40 E1 `external_plan` stories x 15 curated secrets x 3 contexts (with the outline it followed /
      plain no-plan prompt / with the length-matched irrelevant passage) -> same summaries.
Outputs: results/mech/m1_*.pt
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
import gemma as G
from e1_generate import IRREL_TASK, PLAN_TASK

OUT = os.path.join(RES, "mech")
os.makedirs(OUT, exist_ok=True)
BIN = 32
NBINS = 28  # up to 896 tokens
BIN_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47]
ALL_LAYERS = list(range(48))
N_STORIES = 40


def word_token_ids(tok, words):
    """First token of ' word' and ' Word' for each word (union used for log-prob tracking)."""
    ids = []
    for w in words:
        for v in (" " + w.lower(), " " + w.capitalize()):
            i = tok(v, add_special_tokens=False)["input_ids"][0]
            ids.append(i)
    return ids  # 2 per word, ordered as words


def reduce_fn(h):
    """h: [48, T, d] fp16 -> dict(mean=[48,d] fp16, bins=[12, NBINS, d] fp16, nb=int)."""
    hf = h.float()
    mean = hf.mean(1).bfloat16()
    T = h.shape[1]
    nb = min(NBINS, (T + BIN - 1) // BIN)
    bins = torch.zeros(len(BIN_LAYERS), NBINS, h.shape[-1])
    for b in range(nb):
        bins[:, b] = hf[BIN_LAYERS, b * BIN:(b + 1) * BIN].mean(1)
    return {"mean": mean, "bins": bins.bfloat16(), "nb": nb}


def run_set(prompts, texts, tok_ids, tag):
    res = G.residuals(prompts, texts, ALL_LAYERS, batch_size=6, reduce=reduce_fn, extra_logits_tokens=[tok_ids] * len(prompts))
    return res


def m1():
    path = os.path.join(OUT, "m1.pt")
    if os.path.exists(path):
        return
    tok, _ = G.load()
    st = jload(os.path.join(RES, "e1", "stories.json"))["no_secret"]
    st = [s for s in st if s["finished"]][:N_STORIES]
    ctx_words = WORDS + COCA
    contexts = [SYS["dont_reveal"].format(X=w) for w in ctx_words] + [SYS["no_secret"]]
    tok_ids = word_token_ids(tok, ctx_words)
    prompts, texts, meta = [], [], []
    for si, s in enumerate(st):
        for ci, c in enumerate(contexts):
            prompts.append(G.render([{"role": "system", "content": c}, {"role": "user", "content": STORY_TASK}]))
            texts.append(s["text"])
            meta.append((si, ci))
    res = run_set(prompts, texts, tok_ids, "m1")
    S, C = len(st), len(contexts)
    mean = torch.zeros(S, C, 48, 3840, dtype=torch.bfloat16)
    bins = torch.zeros(S, C, len(BIN_LAYERS), NBINS, 3840, dtype=torch.bfloat16)
    nb = torch.zeros(S, dtype=torch.long)
    lp = [[None] * C for _ in range(S)]
    for (si, ci), r in zip(meta, res):
        mean[si, ci] = r["h"]["mean"]
        bins[si, ci] = r["h"]["bins"]
        nb[si] = r["h"]["nb"]
        lp[si][ci] = r["lp"].bfloat16()  # [T, 2*len(ctx_words)]
    torch.save({"mean": mean, "bins": bins, "nb": nb, "lp": lp, "ctx_words": ctx_words, "bin_layers": BIN_LAYERS,
                "story_idx": [s.get("slot") for s in st], "tok_ids": tok_ids}, path)
    print("saved", path)


def m1b():
    path = os.path.join(OUT, "m1b.pt")
    if os.path.exists(path):
        return
    tok, _ = G.load()
    stim = jload(os.path.join(RES, "e1", "stimuli.json"))
    st = [s for s in jload(os.path.join(RES, "e1", "stories.json"))["external_plan"] if s["finished"]][:N_STORIES]
    ctx_kinds = ["plan", "plain", "irrelevant"]
    prompts, texts, meta = [], [], []
    for si, s in enumerate(st):
        o = stim[s["stim"]]
        for ki, kind in enumerate(ctx_kinds):
            user = {"plan": PLAN_TASK.format(outline=o["outline"]), "plain": STORY_TASK, "irrelevant": IRREL_TASK.format(passage=o["passage"])}[kind]
            for wi, w in enumerate(WORDS):
                prompts.append(G.render([{"role": "system", "content": SYS["dont_reveal"].format(X=w)}, {"role": "user", "content": user}]))
                texts.append(s["text"])
                meta.append((si, ki, wi))
    tok_ids = word_token_ids(tok, WORDS)
    res = run_set(prompts, texts, tok_ids, "m1b")
    S = len(st)
    mean = torch.zeros(S, 3, 15, 48, 3840, dtype=torch.bfloat16)
    bins = torch.zeros(S, 3, 15, len(BIN_LAYERS), NBINS, 3840, dtype=torch.bfloat16)
    nb = torch.zeros(S, dtype=torch.long)
    lp = {}
    for (si, ki, wi), r in zip(meta, res):
        mean[si, ki, wi] = r["h"]["mean"]
        bins[si, ki, wi] = r["h"]["bins"]
        nb[si] = r["h"]["nb"]
        lp[(si, ki, wi)] = r["lp"].bfloat16()
    torch.save({"mean": mean, "bins": bins, "nb": nb, "lp": lp, "ctx_kinds": ctx_kinds, "bin_layers": BIN_LAYERS, "tok_ids": tok_ids}, path)
    print("saved", path)


if __name__ == "__main__":
    which = sys.argv[1:] or ["m1", "m1b"]
    if "m1" in which:
        m1()
    if "m1b" in which:
        m1b()
