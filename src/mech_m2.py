"""M2: does the strength of the secret representation during a story predict that story's leakage?

For every generated E1 story (secret conditions), teacher-force its OWN prompt+text and, separately,
the same prompt with the secret sentence replaced by the no-secret system prompt. Save story-mean
residuals at all 48 layers. Per-story metrics (computed in analysis):
  ctx_strength  = < m_own - m_nosecret , r_hat_W >     (secret-in-context contribution)
  content       = < m_nosecret , r_hat_W > (centred)   (what the text itself evokes)
Outputs results/mech/m2.pt
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
import gemma as G

OUT = os.path.join(RES, "mech")
CONDS = ["no_plan", "decoy", "premise", "irrelevant", "external_plan", "self_plan", "not_suppressed"]


def main():
    path = os.path.join(OUT, "m2.pt")
    if os.path.exists(path):
        return
    st = jload(os.path.join(RES, "e1", "stories.json"))
    prompts, texts, meta = [], [], []
    for c in CONDS:
        for i, s in enumerate(st[c]):
            msgs = s["messages"]
            own = G.render(msgs)
            nosec = G.render([{"role": "system", "content": SYS["no_secret"]}] + msgs[1:])
            for k, p in enumerate((own, nosec)):
                prompts.append(p)
                texts.append(s["text"])
                meta.append((c, i, k))
    res = G.residuals(prompts, texts, list(range(48)), batch_size=6, reduce=lambda h: h.float().mean(1).bfloat16())
    out = {}
    for (c, i, k), r in zip(meta, res):
        out.setdefault(c, {}).setdefault(i, [None, None])[k] = r["h"]
    pack = {c: torch.stack([torch.stack(out[c][i]) for i in range(len(st[c]))]) for c in CONDS}  # [N, 2, 48, d]
    torch.save({"m": pack, "words": {c: [s["word"] for s in st[c]] for c in CONDS}}, path)
    print("saved", path)


if __name__ == "__main__":
    main()
