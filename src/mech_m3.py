"""M3: causal interventions during generation of `no_plan` stories (dont_reveal secret prompt).

Directions come from M1 (counterfactual teacher forcing on secret-free stories):
    r_W^l = mean_s [ h(s | secret=W, layer l) - mean_{W' in curated} h(s | secret=W', layer l) ]
Other-concept directions r_C^l are built identically from the 15 COCA-word secret contexts.

Conditions (8 stories x 15 words each):
  abl_secret          ablate r_W (own secret) at every layer, all generation-phase positions
  abl_other           ablate r_C for a COCA word C (other concept, same construction)
  abl_random          ablate a random direction per layer
  abl_secret_sub      ablate the 14-dim span of all 15 curated secret directions
  abl_other_sub       ablate the 14-dim span of all 15 COCA directions (control)
  abl_secret_late     ablate r_W only from generated token 100 onwards
  abl_secret_early    ablate r_W only for the header + first 100 generated tokens
  ko_sentence         attention knockout: generation-phase queries cannot attend to the secret sentence
  steer_L{l}_c{c}     NO secret in prompt; add c * r_W^l at one layer (sufficiency test)
Outputs results/mech/m3_stories.json (same schema as E1 stories).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
import gemma as G

OUT = os.path.join(RES, "mech")
N_PER_WORD = 8
BAND = (20, 40)  # layers whose inputs are (mean-)ablated; chosen by the coherence sweep in m3_sanity.py


def load_dirs():
    """Returns secret directions, other-concept directions, and the grand-mean residual per layer
    (used as the mean-ablation target: average over all 30 secret contexts and 40 stories)."""
    d = torch.load(os.path.join(OUT, "m1.pt"))
    mean = d["mean"].float()  # [S, 31, 48, d]
    cur = mean[:, :15]
    coca = mean[:, 15:30]
    r_cur = (cur - cur.mean(1, keepdim=True)).mean(0)  # [15, 48, d]
    r_coca = (coca - coca.mean(1, keepdim=True)).mean(0)
    mu = mean[:, :30].mean((0, 1))  # [48, d]
    return r_cur, r_coca, mu


def unit(v):
    n = v.norm(dim=-1, keepdim=True)
    return torch.where(n > 0, v / n.clamp_min(1e-8), torch.zeros_like(v))


def orthonormal_span(R):
    """R: [k, 48, d] -> [48, k, d] orthonormal basis rows per layer (via QR), rank-deficient rows zeroed."""
    out = []
    for l in range(R.shape[1]):
        M = R[:, l].T  # [d, k]
        if M.norm() == 0:
            out.append(torch.zeros(R.shape[0], R.shape[2]))
            continue
        Q, Rr = torch.linalg.qr(M)
        keep = Rr.diagonal().abs() > 1e-4 * Rr.diagonal().abs().max()
        Q = Q * keep[None].float()
        out.append(Q.T)
    return torch.stack(out)  # [48, k, d]


def pad49(U):
    """[48, k, d] -> [49, k, d] (no ablation at the final-norm input, which M1 did not measure);
    rows outside BAND are zeroed (no-op)."""
    U = U.clone()
    U[:BAND[0]] = 0
    U[BAND[1]:] = 0
    return torch.cat([U, torch.zeros(1, *U.shape[1:])], 0)


def proj_mu(U, mu):
    """U: [B, 49, k, d]; mu: [48, d] -> [B, 49, k] mean projections (0 at the unused final slot)."""
    m = torch.zeros(U.shape[:3])
    m[:, :48] = torch.einsum("blkd,ld->blk", U[:, :48].float(), mu)
    return m.cuda()


def build_items(cond):
    items = []
    for wi, w in enumerate(WORDS):
        for s in range(N_PER_WORD):
            sysm = SYS["no_secret"] if cond.startswith("steer") else SYS["dont_reveal"].format(X=w)
            items.append({"word": w, "wi": wi, "slot": s, "stim": None,
                          "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": STORY_TASK}]})
    return items


def make_per_item(cond, items, prompts, r_cur, r_coca, mu):
    tok, _ = G.load()
    sec_unit = unit(r_cur)  # [15, 48, d]
    coca_unit = unit(r_coca)
    sub_cur = pad49(orthonormal_span(r_cur))  # [49, 15, d]
    sub_coca = pad49(orthonormal_span(r_coca))
    gen = torch.Generator().manual_seed(1234)
    rand_dirs = unit(torch.randn(len(items), 48, r_cur.shape[-1], generator=gen))

    def per_item(idx, P, enc):
        B = len(idx)
        kw = {}
        if cond in ("abl_secret", "abl_secret_late", "abl_secret_early", "abl_other", "abl_random"):
            U = torch.zeros(B, 49, 1, r_cur.shape[-1])
            for j, i in enumerate(idx):
                wi = items[i]["wi"]
                v = {"abl_other": coca_unit[wi], "abl_random": rand_dirs[i]}.get(cond, sec_unit[wi])
                U[j, :48, 0] = v
                U[j, :BAND[0]] = 0
                U[j, BAND[1]:] = 0
            kw["ablate"] = U.cuda()
            kw["ablate_mu"] = proj_mu(U, mu)
            if cond == "abl_secret_late":
                kw["ablate_steps"] = (100, 10 ** 9)
            if cond == "abl_secret_early":
                kw["ablate_steps"] = (-G.HEADER, 100)
        elif cond in ("abl_secret_sub", "abl_other_sub"):
            S = sub_cur if cond == "abl_secret_sub" else sub_coca
            kw["ablate"] = S[None].expand(B, -1, -1, -1).contiguous().cuda()
            kw["ablate_mu"] = proj_mu(kw["ablate"].cpu(), mu)
        elif cond == "ko_sentence":
            cols = torch.zeros(B, P + 1000, dtype=torch.bool, device="cuda")
            for j, i in enumerate(idx):
                n = len(tok(prompts[i], add_special_tokens=False)["input_ids"])
                a, b = G.find_span(prompts[i], SYS["dont_reveal"].format(X=items[i]["word"]))
                cols[j, P - n + a:P - n + b] = True
            kw["ko_cols"] = cols
        elif cond.startswith("steer"):
            _, L, c = cond.split("_")
            L, c = int(L[1:]), float(c[1:])
            V = torch.stack([r_cur[items[i]["wi"], L] * c for i in idx])
            kw["steer"] = (L, V.cuda())
        else:
            raise ValueError(cond)
        return kw

    return per_item


def run(conds):
    set_seed()
    r_cur, r_coca, mu = load_dirs()
    print("direction norms (layer 12/24/36):", [round(float(r_cur[:, l].norm(dim=-1).mean()), 2) for l in (12, 24, 36)])
    path = os.path.join(OUT, "m3_stories.json")
    stories = jload(path) if os.path.exists(path) else {}
    for ci, cond in enumerate(conds):
        if cond in stories:
            continue
        items = build_items(cond)
        prompts = [G.render(it["messages"]) for it in items]
        pi = make_per_item(cond, items, prompts, r_cur, r_coca, mu)
        outs = G.generate(prompts, max_new_tokens=900, batch_size=32, seed=700 + sum(map(ord, cond)), per_item=pi, desc=cond)
        for it, o in zip(items, outs):
            it.update(o)
        stories[cond] = items
        jdump(stories, path)
        print(cond, "literal", sum(contains_word(i["text"], i["word"]) for i in items), "finished", sum(i["finished"] for i in items), flush=True)


if __name__ == "__main__":
    run(sys.argv[1:] or ["abl_secret", "abl_other", "abl_random", "abl_secret_sub", "abl_other_sub", "abl_secret_late", "abl_secret_early", "ko_sentence"])
