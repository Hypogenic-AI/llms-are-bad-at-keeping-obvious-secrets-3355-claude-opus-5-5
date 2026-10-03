"""Sanity sweep for M3: which layer band can be mean-ablated without destroying coherence?
Generates 1 short story per word (first 5 words) for secret / other-concept / random directions
under several bands, and reports the judged-free proxies: fraction of distinct words and repetition."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
import gemma as G
import mech_m3 as M


def distinct(t):
    w = t.lower().split()
    return len(set(w)) / max(1, len(w))


r_cur, r_coca, mu = M.load_dirs()
res = {}
for band in [(24, 40), (32, 44), (20, 32), (36, 48)]:
    M.BAND = band
    for cond in ["abl_secret", "abl_other", "abl_random"]:
        items = [it for it in M.build_items(cond) if it["slot"] == 0]
        prompts = [G.render(it["messages"]) for it in items]
        outs = G.generate(prompts, max_new_tokens=300, batch_size=16, seed=5, per_item=M.make_per_item(cond, items, prompts, r_cur, r_coca, mu))
        d = sum(distinct(o["text"]) for o in outs) / len(outs)
        print(f"BAND {band} {cond:12s} distinct-word ratio {d:.2f} | {outs[1]['text'][:160]!r}", flush=True)
        res[f"{band}_{cond}"] = {"distinct": d, "samples": [o["text"][:400] for o in outs[:3]]}
jdump(res, os.path.join(RES, "mech", "m3_sanity.json"))
