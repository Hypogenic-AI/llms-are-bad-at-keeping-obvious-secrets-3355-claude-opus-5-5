"""E2: plot-twist secrets. Gemma-3-12B writes chapter 1 of a two-chapter story while knowing
(or not) the twist revealed at the end. 30 premises x 4 twists, one opening per cell per condition.
Outputs results/e2/stimuli.json, results/e2/stories.json"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
import gemma as G
from e1_generate import TOPICS

OUT = os.path.join(RES, "e2")

BASE = "You are writing a short story in two chapters. Premise: {premise}"
TWIST = " The story's twist, which is revealed only at the very end of the final chapter, is: {twist}"
HIDE = (" This twist is a secret. In this chapter, do not reveal it, hint at it, or foreshadow it in any way; "
        "a reader of this chapter must not be able to guess it.")
CH1 = "Write the first chapter of the story (5-8 paragraphs). Just write the chapter, nothing else."
CH1_PLAN = ("Write the first chapter of the story (5-8 paragraphs), following this outline exactly.\n\n[OUTLINE]\n{outline}\n[END OUTLINE]\n\n"
            "Just write the chapter, nothing else.")
CH1_IRREL = ("Below is a reference text. It is unrelated to your task; do not use it.\n\n[TEXT]\n{passage}\n[END TEXT]\n\n" + CH1)
OUTLINE_REQ = ("Write a paragraph-by-paragraph outline for the first chapter (5-8 paragraphs) of this story. It should set up the "
               "characters and the situation, but must not resolve the central question. One numbered line per paragraph, with concrete "
               "details of what happens. Just write the outline, nothing else.")

CONDS = ["no_twist", "twist_unsuppressed", "twist_no_plan", "twist_irrelevant", "twist_external_plan"]


def first_numbered_list(t):
    """Keep only the first numbered list (Gemma sometimes repeats an expanded version)."""
    out, n = [], 0
    for ln in t.replace("**", "").split("\n"):
        m = re.match(r"^\s*(\d+)\.", ln)
        if m:
            if int(m.group(1)) == 1 and n > 0:
                break
            n += 1
            out.append(ln.strip())
        elif ln.strip() and n > 0 and not out[-1].endswith(":"):
            out[-1] += " " + ln.strip()  # wrapped continuation line
    return "\n".join(out)


def build_stimuli():
    path = os.path.join(OUT, "stimuli.json")
    if os.path.exists(path):
        return jload(path)
    tok, _ = G.load()
    prem = jload(os.path.join(OUT, "premises.json"))
    ps = [G.render([{"role": "system", "content": BASE.format(premise=p["premise"])}, {"role": "user", "content": OUTLINE_REQ}]) for p in prem]
    outl = G.generate(ps, max_new_tokens=1000, batch_size=30, seed=21, desc="ch1 outlines")
    req = "Write an encyclopedia-style article of about 400 words on {t}. Plain prose paragraphs, no headings. Just write the article."
    pas = G.generate([G.render([{"role": "user", "content": req.format(t=t)}]) for t in TOPICS[-30:] + TOPICS[:6]], max_new_tokens=800,
                     batch_size=36, seed=22, desc="passages")
    pas = [x["text"] for x in pas if x["finished"]]
    for k, p in enumerate(prem):
        p["outline"] = first_numbered_list(outl[k]["text"])
        n = len(tok(p["outline"], add_special_tokens=False)["input_ids"])
        src, j = pas[k], 1
        while len(tok(src, add_special_tokens=False)["input_ids"]) < n:  # too short: append another article
            src += "\n\n" + pas[(k + 7 * j) % len(pas)]
            j += 1
        ids = tok(src, add_special_tokens=False)["input_ids"]
        txt = tok.decode(ids[:n])
        cut = max(txt.rfind(". "), txt.rfind(".\n"))
        p["passage"] = txt[:cut + 1] if cut > len(txt) * 0.8 else txt
        p["outline_tokens"] = n
        p["passage_tokens"] = len(tok(p["passage"], add_special_tokens=False)["input_ids"])
    jdump(prem, path)
    return prem


def items_for(cond, prem):
    items = []
    for k, p in enumerate(prem):
        for t in range(4):
            sysm = BASE.format(premise=p["premise"])
            if cond != "no_twist":
                sysm += TWIST.format(twist=p["twists"][t])
            if cond not in ("no_twist", "twist_unsuppressed"):
                sysm += HIDE
            user = {"twist_irrelevant": CH1_IRREL.format(passage=p["passage"]),
                    "twist_external_plan": CH1_PLAN.format(outline=p["outline"])}.get(cond, CH1)
            # for no_twist, `twist` is only a nominal label (exchangeable null for the 4-way test)
            items.append({"premise_id": k, "twist_id": t, "word": f"{k}:{t}", "twist": p["twists"][t],
                          "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": user}]})
    return items


def run():
    set_seed()
    prem = build_stimuli()
    path = os.path.join(OUT, "stories.json")
    stories = jload(path) if os.path.exists(path) else {}
    for ci, cond in enumerate(CONDS):
        if cond in stories:
            continue
        items = items_for(cond, prem)
        outs = G.generate([G.render(it["messages"]) for it in items], max_new_tokens=900, batch_size=32, seed=500 + ci, desc=cond)
        for it, o in zip(items, outs):
            it.update(o)
        stories[cond] = items
        jdump(stories, path)
    for c, its in stories.items():
        print(c, len(its), "finished", sum(i["finished"] for i in its), "mean tokens", round(sum(i["n_tokens"] for i in its) / len(its)))


if __name__ == "__main__":
    run()
