"""E1-R: replication of the E1 plan conditions with a larger API writer (google/gemma-3-27b-it via
OpenRouter, T=1.0, top_p=0.95), using exactly the same stimuli (outlines/premises/passages) and design.
Outputs results/e1_27b/stories.json (same schema as E1)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
import e1_generate as E

WRITER = "google/gemma-3-27b-it"
OUT = os.path.join(RES, "e1_27b")
os.makedirs(OUT, exist_ok=True)
CONDS = ["no_plan", "decoy", "premise", "irrelevant", "external_plan", "self_plan"]


def write(msgs, tag):
    return chat(msgs, model=WRITER, temperature=1.0, max_tokens=1500, tag=tag, top_p=0.95).strip()


def main():
    stim = jload(os.path.join(RES, "e1", "stimuli.json"))
    path = os.path.join(OUT, "stories.json")
    stories = jload(path) if os.path.exists(path) else {}
    for cond in CONDS:
        if cond in stories:
            continue
        items = E.items_for(cond, stim)
        for k, it in enumerate(items):
            it["tag"] = f"{cond}-{k}"
        if cond == "self_plan":
            outl = pmap(lambda it: write(it["messages"], it["tag"] + "-outline"), items, workers=16)
            for it, o in zip(items, outl):
                it["self_outline"] = o
                it["messages"] = it["messages"] + [{"role": "assistant", "content": o}, {"role": "user", "content": E.SELF_PLAN_2ND}]
        texts = pmap(lambda it: write(it["messages"], it["tag"]), items, workers=16)
        for it, t in zip(items, texts):
            it.update({"text": t, "n_tokens": len(t.split()), "finished": True})
        stories[cond] = items
        jdump(stories, path)
        print(cond, "done; literal", sum(contains_word(i["text"], i["word"]) for i in items), f"${COST['usd']:.2f}", flush=True)
    if "self_plan_secret_removed" not in stories:
        items = []
        for k, it in enumerate(stories["self_plan"]):
            items.append({"word": it["word"], "stim": it["stim"], "self_outline": it["self_outline"],
                          "messages": [{"role": "system", "content": SYS["no_secret"]},
                                       {"role": "user", "content": E.PLAN_TASK.format(outline=it["self_outline"])}], "tag": f"spsr-{k}"})
        texts = pmap(lambda it: write(it["messages"], it["tag"]), items, workers=16)
        for it, t in zip(items, texts):
            it.update({"text": t, "n_tokens": len(t.split()), "finished": True})
        stories["self_plan_secret_removed"] = items
        jdump(stories, path)


if __name__ == "__main__":
    main()
