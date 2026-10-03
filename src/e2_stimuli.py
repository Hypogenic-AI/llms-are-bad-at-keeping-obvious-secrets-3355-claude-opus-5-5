"""E2 stimuli: story premises, each with 4 mutually exclusive, equally plausible secret twists.
Generated once by anthropic/claude-opus-5.5 (T=0.7) and saved to results/e2/premises.json."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa

OUT = os.path.join(RES, "e2")
os.makedirs(OUT, exist_ok=True)
GEN_MODEL = "anthropic/claude-opus-5.5"

REQ = """I am building a controlled experiment on whether story-writing AIs foreshadow plot twists they know are coming.
Please create {n} story premises. For each premise, write exactly 4 alternative secret twists.

Requirements:
- Premise: 2-3 sentences that set up characters, a setting and an open question, but do not point towards any particular answer. Vary genres (mystery, domestic drama, sci-fi, fantasy, workplace, historical, horror, comedy, etc.) and settings.
- Each twist: ONE concrete sentence revealing a hidden truth that would be revealed at the end of the story.
- The 4 twists for a premise must be mutually exclusive, each fully compatible with the premise, and roughly EQUALLY plausible given the premise alone (a reader of only the premise should not be able to tell which one is real).
- The 4 twists should differ in WHAT they are about (e.g., different culprits, motives, objects, places, identities, supernatural vs mundane), so that an author who knew one of them would be tempted to foreshadow different things.
- Avoid famous twists and cliches like "it was all a dream".
- Theme {tag}: make these {n} premises distinct from each other.

Return ONLY a JSON list of objects with keys "premise" and "twists" (list of 4 strings). No markdown fences."""


def main():
    path = os.path.join(OUT, "premises.json")
    if os.path.exists(path):
        print("exists")
        return
    allp = []
    for tag in ["batch A", "batch B"]:
        out = chat([{"role": "user", "content": REQ.format(n=18, tag=tag)}], model=GEN_MODEL, temperature=0.7, max_tokens=12000)
        out = re.sub(r"^```(json)?|```$", "", out.strip(), flags=re.M)
        allp += json.loads(out)
    allp = [p for p in allp if len(p["twists"]) == 4]
    print(len(allp))
    jdump(allp[:30], path)
    for p in allp[:3]:
        print(json.dumps(p, indent=1))
    print("cost", COST)


if __name__ == "__main__":
    main()
