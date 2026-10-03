"""Generic leakage evaluation for a set of stories with known secrets.

2AFC (Holtzman & West): for every unordered pair of secret words (A, B) we draw one story
written with A and one with B (same condition), and ask the judge both "which text has
secret A?" and "which text has secret B?", each in both presentation orders -> 4 trials per
pair. Under the null (secret has no effect on the text) accuracy is exactly 50% because
secrets are randomly assigned to samples/stimuli.

Rank: the judge ranks all 15 candidate words for each story; we record the rank of the true word.
"""
import itertools
import os
import random
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa


def build_pairs(stories, seed, key="word", matched_by=None):
    """Return list of (i, j) story-index pairs with different secrets.
    Default: one random pair per unordered secret pair. matched_by: pair stories sharing this field
    (e.g. same outline) instead."""
    rng = random.Random(seed)
    if matched_by:
        groups = {}
        for i, s in enumerate(stories):
            groups.setdefault(s[matched_by], []).append(i)
        pairs = []
        for g in groups.values():
            for i, j in itertools.combinations(g, 2):
                if stories[i][key] != stories[j][key]:
                    pairs.append((i, j))
        return pairs
    by = {}
    for i, s in enumerate(stories):
        by.setdefault(s[key], []).append(i)
    keys = sorted(by)
    return [(rng.choice(by[a]), rng.choice(by[b])) for a, b in itertools.combinations(keys, 2)]


def run_2afc(stories, pairs, model=JUDGE_MODEL, key="word", judge_fn=None, one_target=False):
    """Returns list of trial dicts. judge_fn(target, t1, t2) -> 1/2/None (defaults to word judge).
    one_target=True: only the first story's secret is used as target (2 trials per pair, budget mode)."""
    judge_fn = judge_fn or (lambda X, t1, t2: judge_2afc(X, t1, t2, model=model))
    jobs = []
    for k, (i, j) in enumerate(pairs):
        for tgt_idx in (((i,) if k % 2 == 0 else (j,)) if one_target else (i, j)):
            for order in (0, 1):
                first, second = (i, j) if order == 0 else (j, i)
                jobs.append((i, j, tgt_idx, first, second))

    def f(job):
        i, j, t, first, second = job
        ans = judge_fn(stories[t][key], stories[first]["text"], stories[second]["text"])
        correct = None if ans is None else int((first if ans == 1 else second) == t)
        return {"i": i, "j": j, "target": t, "first": first, "answer": ans, "correct": correct}

    return pmap(f, jobs)


def run_rank(stories, model=JUDGE_MODEL, candidates=WORDS):
    def f(s):
        r = judge_rank(s["text"], candidates, model=model)
        return r.index(s["word"]) + 1 if r and s["word"] in r else None

    return pmap(f, stories)


# ----------------------------------------------------------------------------- statistics
def acc_stats(trials, n_boot=10000, seed=0):
    """Primary metric: accuracy with judge abstentions ("Neither text...") scored as 0.5 (half credit),
    so conditions with different abstention rates stay comparable. Also: parsed-only accuracy with a
    two-sided binomial test vs .5, cluster-bootstrap 95% CI (resampling pairs; each pair has 4 trials),
    abstention rate, and the rate of answering '1' (position bias)."""
    vals = [0.5 if x["correct"] is None else float(x["correct"]) for x in trials]
    t = [x for x in trials if x["correct"] is not None]
    k, n = sum(x["correct"] for x in t), len(t)
    p = stats.binomtest(k, n, 0.5).pvalue if n else float("nan")
    clusters = {}
    for x, v in zip(trials, vals):
        clusters.setdefault((x["i"], x["j"]), []).append(v)
    cl = [np.array(v) for v in clusters.values()]
    rng = np.random.default_rng(seed)
    sums = np.array([c.sum() for c in cl])
    cnts = np.array([len(c) for c in cl])
    idx = rng.integers(0, len(cl), size=(n_boot, len(cl)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    pos1 = np.mean([x["answer"] == 1 for x in t]) if t else float("nan")
    return {"acc": float(np.mean(vals)) if vals else float("nan"), "acc_parsed": k / n if n else float("nan"), "k": k, "n": len(trials),
            "n_parsed": n, "p_binom": p, "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "answer1_rate": float(pos1), "abstain_rate": 1 - n / len(trials) if trials else float("nan"), "_clusters": cl}


def diff_test(a, b, n_boot=10000, seed=0):
    """Cluster-bootstrap CI and two-sided p for acc(a) - acc(b) (independent conditions)."""
    rng = np.random.default_rng(seed)
    out = []
    for st in (a, b):
        cl = st["_clusters"]
        sums = np.array([c.sum() for c in cl])
        cnts = np.array([len(c) for c in cl])
        idx = rng.integers(0, len(cl), size=(n_boot, len(cl)))
        out.append(sums[idx].sum(1) / cnts[idx].sum(1))
    d = out[0] - out[1]
    obs = a["acc"] - b["acc"]
    # two-sided p: bootstrap distribution shifted to the null (centred at 0)
    p = np.mean(np.abs(d - obs) >= abs(obs))
    return {"diff": obs, "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)), "p_boot": float(p)}


def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for r, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * n / r)
        q[i] = prev
    return q
