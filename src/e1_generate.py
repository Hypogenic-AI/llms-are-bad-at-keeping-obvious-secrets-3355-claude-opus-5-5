"""E1: word secrets x plan conditions, writer = Gemma-3-12B-it.

Step 1 builds stimuli (secret-free outlines with a premise line; length-matched irrelevant
passages), step 2 generates 120 stories (8 per secret word) for each condition.
Outputs: results/e1/stimuli.json, results/e1/stories.json
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
import gemma as G

OUT = os.path.join(RES, "e1")
os.makedirs(OUT, exist_ok=True)
N_PER_WORD = 8
N_OUTLINES = 60

OUTLINE_REQ = ("Write an outline for a short story (5-8 paragraphs){setting}. Use exactly this format:\n"
               "Premise: <one sentence>\nCharacters: <names and one-line descriptions>\nSetting: <one line>\n"
               "Paragraph plan:\n1. <what happens in paragraph 1>\n2. ...\n"
               "Give one numbered line per paragraph of the story (5-8 lines), with concrete details of what happens, "
               "the objects and places involved, and how the story ends. Just write the outline, nothing else.")
PLAN_TASK = ("Write a short story (5-8 paragraphs) that follows this outline exactly.\n\n[OUTLINE]\n{outline}\n[END OUTLINE]\n\n"
             "Just write the story, nothing else.")
PREMISE_TASK = "Write a short story (5-8 paragraphs) based on this premise: {premise}\n\nJust write the story, nothing else."
IRREL_TASK = ("Below is a reference text. It is unrelated to your task; do not use it.\n\n[TEXT]\n{passage}\n[END TEXT]\n\n"
              "Write a short story (5-8 paragraphs). Just write the story, nothing else.")
SELF_PLAN_2ND = "Now write the story (5-8 paragraphs), following your outline exactly. Just write the story, nothing else."

# Varied settings used to diversify the secret-free outlines (unprompted, Gemma writes the same
# lighthouse-keeper story most of the time). None names a secret word.
SETTINGS = ["a night market in Taipei", "a Mars research base", "a small-town bakery", "a 1920s jazz club", "a hospital night shift",
            "a high-school chess tournament", "a fishing trawler in the Bering Sea", "a medieval monastery", "a family road trip across Australia",
            "a startup office in Berlin", "a desert caravan", "a suburban garden competition", "an Antarctic research station", "a Victorian circus",
            "a subway car stuck between stations", "a wedding in rural India", "a haunted boarding school", "a vineyard during harvest",
            "a refugee camp kitchen", "a space elevator under construction", "a jungle expedition in Borneo", "a retirement home talent show",
            "a ski resort during a blizzard", "a Roman legion's winter camp", "a cyberpunk megacity", "a dairy farm in Wisconsin",
            "a kabuki theatre in Edo-era Japan", "an underwater habitat", "a dog-sledding race", "a busy airport terminal", "a Venetian glassblowing workshop",
            "a beekeeping village", "a submarine on patrol", "a Moroccan spice market", "a forest fire lookout", "a museum after closing time",
            "a coal-mining town in Wales", "a generation ship", "a tailor's shop in Naples", "a summer camp by a lake", "a courtroom in 1950s Alabama",
            "a hot-air balloon festival", "a monastery in Tibet", "a pirate ship in the Caribbean", "a laundromat at midnight", "a Viking settlement",
            "a mountain rescue team", "a rooftop garden in New York", "a traveling puppet show", "a fortune teller's tent", "a crumbling castle in Scotland",
            "a 24-hour diner", "a bamboo forest in China", "a robotics competition", "a sheep station in New Zealand", "a library in Alexandria",
            "a carnival in Rio", "a remote weather station", "a chocolate factory", "a train crossing Siberia", "a bookshop in Paris",
            "a cattle drive in Texas", "a tea plantation in Sri Lanka", "an ice-fishing village", "a floating city", "a symphony orchestra's tour bus",
            "a mushroom farm", "a submarine cable repair ship", "a puppet theatre in Prague", "a firefighter station", "a zoo during a power outage",
            "a quilting circle", "a houseboat in Amsterdam", "an abandoned amusement park", "a Saharan oasis", "a moon colony school"]

# Mundane expository topics for the irrelevant-context control (screened by hand to avoid the 15 secrets).
TOPICS = ["how postal sorting machines read addresses", "the history of the standard shipping pallet", "how concrete cures",
          "the life cycle of the common housefly", "how barcodes encode numbers", "the rules of curling", "how dishwashers heat water",
          "the history of the paperclip", "how traffic lights are timed", "the biology of moss", "how yeast makes bread rise",
          "the invention of the zipper", "how elevators stay safe", "the migration of eels", "how crossword puzzles are constructed",
          "the history of soap", "how asphalt roads are laid", "the anatomy of a honeybee hive", "how refrigerators move heat",
          "the origins of chess", "how hearing aids work", "the history of the bicycle", "how potatoes are farmed", "the structure of a beaver dam",
          "how printers lay down ink", "the history of buttons", "how sourdough starters are maintained", "how tides are predicted",
          "the life of earthworms in soil", "how airports schedule runways", "the chemistry of rust prevention on cars", "how glass bottles are recycled",
          "the history of the fork", "how speed bumps are designed", "the domestication of chickens", "how tea is processed",
          "the workings of a mechanical lock", "the history of the sandwich", "how rice paddies are irrigated", "how lawn mowers cut grass",
          "the history of playing cards", "how cheese is aged", "how hot air balloons fly", "the biology of mushrooms", "how libraries catalog books",
          "the history of the toothbrush", "how wind turbines are maintained", "how carpets are woven by machine", "the training of sheepdogs",
          "how subway tunnels are ventilated", "the history of chewing gum", "how olive oil is pressed", "the design of fire hydrants",
          "how sunflowers track light", "the history of the stapler", "how bowling pins are reset", "how snails build shells",
          "the history of the shopping cart", "how ice rinks are resurfaced", "how pasta is extruded", "how bridges expand in heat",
          "the history of the pencil eraser", "how hedgehogs hibernate", "how fabric dye is fixed", "the origins of bowling"]


def clean_outline(t):
    """Keep the header and the FIRST numbered paragraph plan (Gemma often repeats an expanded plan)."""
    t = t.replace("**", "")
    lines, out, seen_plan, nums = t.split("\n"), [], False, 0
    for ln in lines:
        s_ = ln.strip()
        if "paragraph plan" in s_.lower():
            seen_plan = True
            out.append(ln)
            continue
        if seen_plan:
            m = re.match(r"^(\d+)\.", s_)
            if m:
                if int(m.group(1)) == 1 and nums > 0:
                    break
                nums += 1
                out.append(ln)
            elif s_ and nums > 0:
                break
        else:
            out.append(ln)
    return "\n".join(out).strip(), nums


def secret_free(text):
    return not any(contains_word(text, w) for w in WORDS)


def assignment(k):
    """Balanced design: stimulus k (0..59) is used with two different secrets; each word gets 8 stimuli."""
    return [WORDS[k % 15], WORDS[(k + 1 + k // 15) % 15]]


def build_stimuli():
    path = os.path.join(OUT, "stimuli.json")
    if os.path.exists(path):
        return jload(path)
    tok, _ = G.load()
    # Outlines from a secret-free call.
    ps = [G.render([{"role": "system", "content": SYS["no_secret"]}, {"role": "user", "content": OUTLINE_REQ.format(setting=" set in " + st_)}])
          for st_ in SETTINGS]
    raw_path = os.path.join(OUT, "raw_outlines.json")
    outs = jload(raw_path) if os.path.exists(raw_path) else []
    outlines, used = [], set()
    reasons = {"unfinished": 0, "format": 0, "secret_word": 0}
    rnd = 0
    while True:
        for o in outs:
            t, nums = clean_outline(o["text"])
            m = re.search(r"Premise:\s*(.+)", t)
            if not o["finished"] and nums < 5:
                reasons["unfinished"] += 1
            elif not (m and "paragraph plan" in t.lower() and 5 <= nums <= 8):
                reasons["format"] += 1
            elif not secret_free(t):
                reasons["secret_word"] += 1
            elif o["setting"] not in used:  # at most one outline per setting (diversity)
                used.add(o["setting"])
                outlines.append({"outline": t, "premise": m.group(1).strip(), "setting": o["setting"]})
        print("valid outlines", len(outlines), reasons, flush=True)
        if len(outlines) >= N_OUTLINES or rnd >= 4:
            break
        todo = [i for i, st_ in enumerate(SETTINGS) if st_ not in used]
        new = G.generate([ps[i] for i in todo], max_new_tokens=1100, batch_size=32, seed=11 + rnd, desc=f"outlines r{rnd}")
        for o_, i in zip(new, todo):
            o_["setting"] = SETTINGS[i]
        outs = new
        allraw = (jload(raw_path) if os.path.exists(raw_path) else []) + new
        jdump(allraw, raw_path)
        rnd += 1
    outlines = outlines[:N_OUTLINES]
    assert len(outlines) == N_OUTLINES
    # Irrelevant passages, length-matched (tokens) to each outline.
    req = "Write an encyclopedia-style article of about 400 words on {t}. Plain prose paragraphs, no headings. Just write the article."
    pp = [G.render([{"role": "user", "content": req.format(t=t)}]) for t in TOPICS]
    raw_p = os.path.join(OUT, "raw_passages.json")
    pas = jload(raw_p) if os.path.exists(raw_p) else G.generate(pp, max_new_tokens=800, batch_size=33, seed=12, desc="passages")
    jdump(pas, raw_p)
    pas = [p_["text"] for p_ in pas if secret_free(p_["text"]) and p_["finished"]]
    print("valid passages", len(pas))
    assert len(pas) >= N_OUTLINES
    for k, o in enumerate(outlines):
        n = len(tok(o["outline"], add_special_tokens=False)["input_ids"])
        src, j = pas[k], 1
        while len(tok(src, add_special_tokens=False)["input_ids"]) < n:  # too short: append another article
            src += "\n\n" + pas[(k + 20 * j) % len(pas)]
            j += 1
        ids = tok(src, add_special_tokens=False)["input_ids"]
        txt = tok.decode(ids[:n])
        # trim to last full sentence so the passage reads naturally (length within a few tokens)
        cut = max(txt.rfind(". "), txt.rfind(".\n"))
        o["passage"] = txt[:cut + 1] if cut > len(txt) * 0.8 else txt
        o["outline_tokens"] = n
        o["passage_tokens"] = len(tok(o["passage"], add_special_tokens=False)["input_ids"])
        o["words"] = assignment(k)
    jdump(outlines, path)
    return outlines


def items_for(cond, stim):
    """List of (word, slot, messages) for a condition. slot = sample index within word (0..7)."""
    items = []
    if cond in ("no_secret", "not_suppressed", "no_plan", "decoy"):
        for w in WORDS:
            for s in range(N_PER_WORD):
                sysm = {"no_secret": SYS["no_secret"], "not_suppressed": SYS["not_suppressed"].format(X=w),
                        "no_plan": SYS["dont_reveal"].format(X=w), "decoy": SYS["decoy"].format(X=w, Y=DECOY[w])}[cond]
                items.append({"word": w, "slot": s, "stim": None,
                              "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": STORY_TASK}]})
        return items
    for k, st in enumerate(stim):
        for w in st["words"]:
            if cond == "external_plan":
                user = PLAN_TASK.format(outline=st["outline"])
            elif cond == "premise":
                user = PREMISE_TASK.format(premise=st["premise"])
            elif cond == "irrelevant":
                user = IRREL_TASK.format(passage=st["passage"])
            elif cond == "self_plan":
                user = None
            else:
                raise ValueError(cond)
            msgs = [{"role": "system", "content": SYS["dont_reveal"].format(X=w)}]
            msgs.append({"role": "user", "content": user if user else OUTLINE_REQ.format(setting="")})
            items.append({"word": w, "stim": k, "messages": msgs})
    return items


def run():
    set_seed()
    stim = build_stimuli()
    path = os.path.join(OUT, "stories.json")
    stories = jload(path) if os.path.exists(path) else {}
    conds = ["no_plan", "no_secret", "not_suppressed", "decoy", "premise", "irrelevant", "external_plan", "self_plan"]
    for ci, cond in enumerate(conds):
        if cond in stories:
            continue
        items = items_for(cond, stim)
        if cond == "self_plan":
            # turn 1: the secret-holding model writes its own outline
            o1 = G.generate([G.render(it["messages"]) for it in items], max_new_tokens=1100, batch_size=32, seed=100 + ci, desc="self_plan outline")
            for it, o in zip(items, o1):
                it["self_outline"] = o["text"]
                it["messages"] = it["messages"] + [{"role": "assistant", "content": o["text"]}, {"role": "user", "content": SELF_PLAN_2ND}]
        outs = G.generate([G.render(it["messages"]) for it in items], max_new_tokens=900, batch_size=32, seed=200 + ci, desc=cond)
        for it, o in zip(items, outs):
            it.update(o)
        stories[cond] = items
        jdump(stories, path)
    # self_plan_secret_removed: the secret-aware self-outline, followed by a writer with NO secret
    if "self_plan_secret_removed" not in stories:
        items = []
        for it in stories["self_plan"]:
            items.append({"word": it["word"], "stim": it["stim"], "self_outline": it["self_outline"],
                          "messages": [{"role": "system", "content": SYS["no_secret"]},
                                       {"role": "user", "content": PLAN_TASK.format(outline=it["self_outline"])}]})
        outs = G.generate([G.render(it["messages"]) for it in items], max_new_tokens=900, batch_size=32, seed=300, desc="self_plan_secret_removed")
        for it, o in zip(items, outs):
            it.update(o)
        stories["self_plan_secret_removed"] = items
        jdump(stories, path)
    for c, its in stories.items():
        n = len(its)
        print(c, n, "finished", sum(i["finished"] for i in its), "mean tokens", round(sum(i["n_tokens"] for i in its) / n),
              "literal mention", sum(contains_word(i["text"], i["word"]) for i in its))


if __name__ == "__main__":
    run()
