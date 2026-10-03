"""Judge E1 (and any other word-secret story sets with the same schema).

Usage: python src/e1_judge.py [stories.json] [out.json]
For each condition: 2AFC (105 random cross-secret pairs x 2 targets x 2 orders = 420 trials),
per-story 15-way rank of the true word, and (for plan conditions) a content-matched 2AFC where
both stories followed the same outline/premise/passage. Results are cached (API) and saved.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
from judging import build_pairs, run_2afc, run_rank

SKIP_2AFC = {"no_secret"}  # no secret -> discrimination undefined
HALF = {"abl_secret_late", "abl_secret_early", "ko_sentence"}  # budget mode: 210 trials (one target per pair, both orders)


def judge_file(src, dst, conds=None, quality=False):
    stories = jload(src)
    for items_ in stories.values():  # judge length-equalised texts
        for s_ in items_:
            s_["text"] = judge_text(s_["text"])
    out = jload(dst) if os.path.exists(dst) else {}
    for cond, items in stories.items():
        if conds and cond not in conds:
            continue
        seed = sum(map(ord, cond))
        rec = out.get(cond, {})
        if "rank" not in rec:
            rec["rank"] = run_rank(items)
        if cond not in SKIP_2AFC and not cond.startswith("steer") and "afc" not in rec:
            pairs = build_pairs(items, seed)
            rec["afc_pairs"] = pairs
            rec["afc"] = run_2afc(items, pairs, one_target=cond in HALF)
        if items and items[0].get("stim") is not None and cond not in SKIP_2AFC and "afc_matched" not in rec:
            mp = build_pairs(items, seed, matched_by="stim")
            rec["afc_matched_pairs"] = mp
            rec["afc_matched"] = run_2afc(items, mp)
        # repair: re-judge trials that were unparsed under an older protocol (forced-choice retry now built in)
        for key_ in ("afc", "afc_matched"):
            if key_ in rec and any(t["answer"] is None for t in rec[key_]):
                pairs_ = rec[key_ + "_pairs"] if key_ == "afc_matched" else rec["afc_pairs"]
                fixed = run_2afc(items, pairs_, one_target=cond in HALF and key_ == "afc")
                rec[key_] = fixed
        if quality and "quality" not in rec and not cond.startswith("steer"):  # budget: steering gets rank only
            rec["quality"] = pmap(lambda s: judge_quality(s["text"]), items)
        out[cond] = rec
        jdump(out, dst)
        acc = [t["correct"] for t in rec.get("afc", []) if t["correct"] is not None]
        ranks = [r for r in rec["rank"] if r]
        print(f"{cond:28s} 2AFC={sum(acc) / max(1, len(acc)):.3f} (n={len(acc)})  mean rank={sum(ranks) / len(ranks):.2f}  "
              f"top1={sum(r == 1 for r in ranks) / len(ranks):.3f}   cost so far ${COST['usd']:.2f}", flush=True)
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "e1", "stories.json")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RES, "e1", "judged.json")
    q = "--quality" in sys.argv
    judge_file(src, dst, quality=q)
