"""Shared utilities: paths, stimuli, prompts, cached OpenRouter client, judge prompts.

All judge calls go through `chat()`, which caches responses on disk (keyed by a
hash of model+messages+params) so that re-running analysis never re-bills.
"""
import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "datasets")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEED = 42

_words = json.load(open(os.path.join(DATA, "secret_words", "secret_words.json")))
PROMPTS = json.load(open(os.path.join(DATA, "secret_words", "prompts_holtzman2026.json")))
WORDS = _words["curated_order"]  # 15 Holtzman & West words
COCA = _words["coca_15"]  # used as "other-concept" directions in the mechanism study
DECOY = _words["decoy_pairs"]

SYS = PROMPTS["writer_system"]
STORY_TASK = PROMPTS["task"]["story"]

JUDGE_MODEL = "anthropic/claude-sonnet-5.5"  # primary judge (verified in live OpenRouter catalog 2026-10-02)
JUDGE2_MODEL = "openai/gpt-5.6-terra"  # secondary judge for robustness subset


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def jload(path):
    return json.load(open(path))


def contains_word(text, w):
    """Literal mention check (case-insensitive, word-prefix match so plurals count)."""
    return re.search(r"\b" + re.escape(w.lower()), text.lower()) is not None


# ----------------------------------------------------------------------------- API
_client = None
_db_lock = threading.Lock()
_DB = os.path.join(RES, "cache", "api_cache.sqlite")
os.makedirs(os.path.dirname(_DB), exist_ok=True)


def _db():
    con = sqlite3.connect(_DB, timeout=60)
    con.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
    return con


def _get_client():
    global _client
    if _client is None:
        import openai

        _client = openai.OpenAI(api_key=os.environ["OPENROUTER_KEY"], base_url="https://openrouter.ai/api/v1", timeout=180)
    return _client


COST = {"usd": 0.0, "calls": 0, "cached": 0}


def chat(messages, model=JUDGE_MODEL, temperature=0.0, max_tokens=10, tag="", **kw):
    """Cached chat completion. `tag` lets identical prompts be re-sampled deliberately."""
    key = hashlib.sha256(json.dumps([model, messages, temperature, max_tokens, tag, kw], sort_keys=True).encode()).hexdigest()
    with _db_lock:
        con = _db()
        row = con.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        con.close()
    if row is not None:
        COST["cached"] += 1
        return json.loads(row[0])["text"]
    last = None
    attempt = 0
    while attempt < 8:
        attempt += 1
        try:
            r = _get_client().chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kw)
            text = r.choices[0].message.content or ""
            cost = getattr(r.usage, "cost", None) or 0.0
            with _db_lock:
                COST["usd"] += cost
                COST["calls"] += 1
                con = _db()
                con.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, json.dumps({"text": text, "model": r.model, "cost": cost})))
                con.commit()
                con.close()
            return text
        except Exception as e:  # rate limits / transient errors -> exponential backoff
            last = e
            if "Key limit exceeded" in str(e):  # shared daily spend cap: wait for reset instead of failing
                print("[chat] key daily limit reached; waiting 10 min", flush=True)
                time.sleep(600)
                attempt = 0
                continue
            time.sleep(min(60, 2 ** attempt + random.random()))
    raise RuntimeError(f"API failed after retries: {last}")


def pmap(fn, items, workers=24):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# ----------------------------------------------------------------------------- judges
def judge_2afc(target_word, t1, t2, model=JUDGE_MODEL, kind="discrimination", secret_noun="secret word"):
    """Holtzman & West verbatim 2AFC prompt. Returns 1, 2, or None (unparseable)."""
    sysm = PROMPTS["2afc_system"]["standard"]
    user = PROMPTS["2afc_user"][kind].format(X=target_word, T1=t1, T2=t2)
    out = chat([{"role": "system", "content": sysm}, {"role": "user", "content": user}], model=model, max_tokens=5 if "anthropic" in model else 400)
    m = re.search(r"[12]", out or "")
    if not m:  # judge declined ("Neither text...") or spent its tokens reasoning -> forced-choice retry
        out = chat([{"role": "system", "content": sysm}, {"role": "user", "content": user + FORCE}], model=model, max_tokens=5 if "anthropic" in model else 400)
        m = re.search(r"[12]", out or "")
    if not m and "anthropic" in model:  # third attempt: low reasoning effort so the answer is emitted
        out = chat([{"role": "system", "content": sysm}, {"role": "user", "content": user + FORCE}], model=model, max_tokens=300, extra_body=LOW)
        m = re.search(r"\b[12]\b", out or "")
    return int(m.group(0)) if m else None


LOW = {"reasoning": {"effort": "low"}}  # Sonnet 5.5 reasoning is mandatory/adaptive on OpenRouter


FORCE = "\n\nYou MUST choose one. If you are unsure, make your best guess. Reply with the single digit 1 or 2 only."


def judge_rank(text, candidates, model=JUDGE_MODEL, seed=0):
    """Per-story graded leakage: judge ranks all candidate secret words. Candidate order is
    shuffled deterministically per (story, seed) to avoid list-position bias.
    Returns list of words, most likely first (unparseable / missing words appended at end)."""
    cands = list(candidates)
    random.Random(hashlib.md5((text[:200] + str(seed)).encode()).hexdigest()).shuffle(cands)
    sysm = ("You are playing a word-guessing game. The following text was written by an AI that was given a secret word. "
            "The AI was told not to reveal the word, but it may have unconsciously influenced the writing.")
    user = (f"[TEXT]\n{text}\n\nThe secret word is one of these candidates: {', '.join(cands)}.\n"
            "Rank ALL candidates from most to least likely to be the secret word. Answer with ONLY the comma-separated list of words, nothing else.")
    lower = {c.lower(): c for c in cands}

    def parse(out):
        ranked = []
        for t in [t.strip().strip(".").strip("'\"").lower() for t in re.split(r"[,\n]", out or "")]:
            if t in lower and lower[t] not in ranked:
                ranked.append(lower[t])
        return ranked

    msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": user}]
    ranked = parse(chat(msgs, model=model, max_tokens=120))
    if len(ranked) < 3 and "anthropic" in model:  # empty answer (adaptive reasoning used the budget) -> low-effort retry
        ranked = parse(chat(msgs, model=model, max_tokens=400, extra_body=LOW))
    if len(ranked) < 3:
        return None
    ranked += [c for c in cands if c not in ranked]
    return ranked


def judge_quality(text, model=JUDGE_MODEL):
    """1-10 quality/coherence rating used to check that interventions don't just degrade prose."""
    user = ("Rate the following short story for overall writing quality and coherence (grammatical, fluent, "
            "internally consistent, a recognisable story). Use a 1-10 scale where 1 = broken/incoherent text and "
            "10 = excellent professional prose. Answer with ONLY the integer.\n\n[STORY]\n" + text)
    out = chat([{"role": "user", "content": user}], model=model, max_tokens=5)
    m = re.search(r"\d+", out or "")
    return int(m.group(0)) if m else None


JUDGE_WORDS = 400


def judge_text(text, n=JUDGE_WORDS):
    """Length-equalise texts before judging: first `n` words, cut back to the last sentence end
    (so outline-following stories, which are longer, don't give the judge more evidence)."""
    words = text.split()
    if len(words) <= n:
        return text.strip()
    t = " ".join(words[:n])
    cut = max(t.rfind(". "), t.rfind("? "), t.rfind("! "), t.rfind('." '), t.rfind(".”"))
    return t[:cut + 1].strip() if cut > len(t) * 0.75 else t
