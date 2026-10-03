"""Analysis of M1/M1b (counterfactual teacher forcing).

Quantities (all at story positions; text identical across contexts):
* 15-way secret decoding accuracy per (layer, position bin), using a nearest-direction classifier
  with directions fit on the *other* half of the stories (2-fold cross-fitting):
      pred(s, ctx) = argmax_W < h(s, ctx) - mean_{ctx'} h(s, ctx'),  r_hat_W >
* Secret signal strength: projection of the centred residual onto its own r_hat_W, in units of the
  std of projections onto the 14 other directions (a z-like score).
* Final-layer log-prob of the secret word's tokens when secret=W vs when the secret is another word
  (ironic rebound vs suppression).
* M1b: the same metrics under plan / plain / irrelevant contexts for the same (plan-following) texts.
Outputs: results/mech/m1_summary.json and figures.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa

OUT = os.path.join(RES, "mech")


def fit_dirs(X):
    """X: [S, W, ..., d] residuals for W secret contexts. Returns unit directions [W, ..., d]
    from diff-in-means (centred over contexts)."""
    C = X - X.mean(1, keepdim=True)
    r = C.mean(0)
    return r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def decode(X, R):
    """X: [S, W, d] centred over W; R: [W, d] unit. Returns accuracy and own-projection z-score."""
    proj = torch.einsum("swd,vd->swv", X, R)  # [S, W(true), W(dir)]
    pred = proj.argmax(-1)
    acc = (pred == torch.arange(X.shape[1])[None]).float().mean().item()
    own = proj.diagonal(dim1=1, dim2=2)  # [S, W]
    mask = ~torch.eye(X.shape[1], dtype=torch.bool)
    others = proj[:, mask].reshape(X.shape[0], X.shape[1], -1)
    z = ((own - others.mean(-1)) / others.std(-1).clamp_min(1e-6)).mean().item()
    return acc, z


def crossfit_curves(bins, nb, words_slice):
    """bins: [S, C, nL, NB, d] fp16. Returns acc[nL, NB], z[nL, NB], n_stories_per_bin[NB]."""
    S = bins.shape[0]
    halves = [torch.arange(S) % 2 == 0, torch.arange(S) % 2 == 1]
    nL, NB = bins.shape[2], bins.shape[3]
    acc = np.full((nL, NB), np.nan)
    z = np.full((nL, NB), np.nan)
    nps = np.array([(nb > b).sum().item() for b in range(NB)])
    for li in range(nL):
        Xall = bins[:, words_slice, li].float()  # [S, W, NB, d]
        # directions from story-mean residual (all bins) of the training half
        for b in range(NB):
            valid = nb > b
            if valid.sum() < 6:
                continue
            accs, zs, ws = [], [], []
            for tr, te in [(halves[0], halves[1]), (halves[1], halves[0])]:
                trm = tr & (nb > 0)
                # train directions on all valid bins of training stories (position-pooled)
                Xtr = torch.stack([Xall[s, :, :nb[s]].mean(1) for s in torch.where(trm)[0]])  # [s, W, d]
                R = fit_dirs(Xtr)
                tev = te & valid
                if tev.sum() == 0:
                    continue
                Xte = Xall[tev, :, b]
                Xte = Xte - Xte.mean(1, keepdim=True)
                a, zz = decode(Xte, R)
                accs.append(a)
                zs.append(zz)
                ws.append(tev.sum().item())
            acc[li, b] = np.average(accs, weights=ws)
            z[li, b] = np.average(zs, weights=ws)
    return acc, z, nps


def logprob_effect(d, nwords):
    """Mean over stories/positions of [lp(W tokens | secret=W) - mean_{W'!=W} lp(W tokens | secret=W')].
    lp layout: [T, 2*len(ctx_words)] (two case variants per word); use logsumexp of the two variants."""
    S = len(d["lp"])
    eff = np.zeros((S, nwords))
    eff_bins = []
    for s in range(S):
        L = torch.stack([d["lp"][s][c].float() for c in range(nwords)])  # [Wctx, T, 2*Wall]
        L = torch.logsumexp(L.reshape(L.shape[0], L.shape[1], -1, 2), -1)[:, :, :nwords]  # [Wctx, T, Wtok]
        own = L.diagonal(dim1=0, dim2=2)  # [T, W]
        tot = L.sum(0)  # [T, W]
        other = (tot - own.T.T) / (nwords - 1)
        eff[s] = (own - other).mean(0).numpy()
        eff_bins.append((own - other).mean(1).numpy())
    return eff, eff_bins


def main():
    summ = {}
    d = torch.load(os.path.join(OUT, "m1.pt"))
    BL = d["bin_layers"]
    acc, z, nps = crossfit_curves(d["bins"], d["nb"], slice(0, 15))
    acc_c, z_c, _ = crossfit_curves(d["bins"], d["nb"], slice(15, 30))
    summ["m1"] = {"bin_layers": BL, "acc": acc.tolist(), "z": z.tolist(), "acc_coca": acc_c.tolist(), "z_coca": z_c.tolist(),
                  "stories_per_bin": nps.tolist()}
    # layerwise decoding using story-mean residuals at all 48 layers (cross-fit)
    mean = d["mean"].float()
    S = mean.shape[0]
    lay_acc, lay_z = [], []
    for l in range(48):
        accs, zs = [], []
        for tr in (torch.arange(S) % 2 == 0, torch.arange(S) % 2 == 1):
            R = fit_dirs(mean[tr][:, :15, l])
            X = mean[~tr][:, :15, l]
            a, zz = decode(X - X.mean(1, keepdim=True), R)
            accs.append(a)
            zs.append(zz)
        lay_acc.append(float(np.mean(accs)))
        lay_z.append(float(np.mean(zs)))
    summ["m1"]["layer_acc_storymean"] = lay_acc
    summ["m1"]["layer_z_storymean"] = lay_z
    # how big is the secret effect relative to the no-secret context? (norm of centred diff vs residual norm)
    eff, eff_bins = logprob_effect(d, 15)
    summ["m1"]["lp_effect_per_word"] = dict(zip(WORDS, eff.mean(0).tolist()))
    summ["m1"]["lp_effect_mean"] = float(eff.mean())
    summ["m1"]["lp_effect_frac_positive_stories"] = float((eff.mean(1) > 0).mean())
    # no-secret vs secret contexts: own-word lp, secret=W minus no-secret
    S_ = len(d["lp"])
    ns = []
    for s in range(S_):
        Lw = torch.stack([d["lp"][s][c].float() for c in range(15)])
        Lw = torch.logsumexp(Lw.reshape(Lw.shape[0], Lw.shape[1], -1, 2), -1)[:, :, :15]
        L0 = d["lp"][s][30].float()
        L0 = torch.logsumexp(L0.reshape(L0.shape[0], -1, 2), -1)[:, :15]
        ns.append((Lw.diagonal(dim1=0, dim2=2) - L0).mean(0).numpy())
    summ["m1"]["lp_secret_minus_nosecret_per_word"] = dict(zip(WORDS, np.mean(ns, 0).tolist()))

    # ---- figure: decoding vs position
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for li, l in enumerate(BL):
        if l in (8, 16, 24, 32, 40, 47):
            ax[0].plot(np.arange(acc.shape[1]) * 32, acc[li], label=f"layer {l}")
            ax[1].plot(np.arange(z.shape[1]) * 32, z[li], label=f"layer {l}")
    ax[0].axhline(1 / 15, color="gray", ls="--", label="chance")
    ax[0].set_xlabel("story token position (32-token bins)")
    ax[0].set_ylabel("15-way secret decoding accuracy")
    ax[0].set_title("Secret identity is decodable at every story position\n(same text, only the secret in context differs)")
    ax[1].set_xlabel("story token position")
    ax[1].set_ylabel("own-direction projection (z vs other directions)")
    ax[1].set_title("Secret signal strength along the story")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "m1_decoding_vs_position.png"), dpi=150)
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(48), lay_acc, label="curated 15 (story-mean)")
    ax.axhline(1 / 15, color="gray", ls="--")
    ax.set_xlabel("layer")
    ax.set_ylabel("decoding accuracy")
    ax.set_title("Secret decodability by layer")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "m1_decoding_by_layer.png"), dpi=150)
    plt.close()

    # ---- M1b: dilution by context (plan / plain / irrelevant), same texts
    pb = os.path.join(OUT, "m1b.pt")
    if os.path.exists(pb):
        e = torch.load(pb)
        # directions from M1 (secret-free stories, plain context), all stories, story-mean residual per bin layer
        R_bins = {}
        for li, l in enumerate(BL):
            Xtr = torch.stack([d["bins"][s, :15, li, :d["nb"][s]].float().mean(1) for s in range(d["bins"].shape[0])])
            R_bins[li] = fit_dirs(Xtr)
        res = {}
        for ki, kind in enumerate(e["ctx_kinds"]):
            A = np.full((len(BL), e["bins"].shape[4]), np.nan)
            Z = np.full_like(A, np.nan)
            Zs = np.full((len(BL), e["bins"].shape[0]), np.nan)  # per-story pooled z (for paired tests)
            for li in range(len(BL)):
                for b in range(e["bins"].shape[4]):
                    valid = e["nb"] > b
                    if valid.sum() < 6:
                        continue
                    X = e["bins"][valid, ki, :, li, b].float()
                    X = X - X.mean(1, keepdim=True)
                    A[li, b], Z[li, b] = decode(X, R_bins[li])
                for s in range(e["bins"].shape[0]):
                    X = e["bins"][s:s + 1, ki, :, li, :e["nb"][s]].float().mean(2)
                    X = X - X.mean(1, keepdim=True)
                    Zs[li, s] = decode(X, R_bins[li])[1]
            # raw secret-variance: mean squared norm of centred residual (how much the secret moves the state)
            Xm = e["mean"][:, ki].float()  # [S, 15, 48, d]
            var = ((Xm - Xm.mean(1, keepdim=True)) ** 2).sum(-1).mean((0, 1)).sqrt()  # [48]
            res[kind] = {"acc": A.tolist(), "z": Z.tolist(), "z_story": Zs.tolist(), "secret_spread_by_layer": var.tolist()}
        # per-story paired tests on the direction-free spread (and on z) at several layers
        paired = {}
        Xall = e["mean"].float()  # [S, 3, 15, 48, d]
        sp = ((Xall - Xall.mean(2, keepdim=True)) ** 2).sum(-1).mean(2).sqrt()  # [S, 3, 48]
        kinds = e["ctx_kinds"]
        for L in (16, 24, 32, 40, 47):
            a, b, c = (sp[:, kinds.index(k), L].numpy() for k in ("plan", "plain", "irrelevant"))
            paired[L] = {"plan_over_plain_ratio_median": float(np.median(a / b)), "irrelevant_over_plain_ratio_median": float(np.median(c / b)),
                         "p_plan_vs_plain": float(stats.wilcoxon(a, b).pvalue), "p_irrelevant_vs_plain": float(stats.wilcoxon(c, b).pvalue),
                         "p_plan_vs_irrelevant": float(stats.wilcoxon(a, c).pvalue), "frac_stories_plan_lt_plain": float(np.mean(a < b)),
                         "n": int(len(a))}
        res["paired_spread"] = paired
        summ["m1b"] = res
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        li = BL.index(32)
        for kind, c in zip(e["ctx_kinds"], ["C1", "C0", "C2"]):
            ax[0].plot(np.arange(len(res[kind]["acc"][li])) * 32, res[kind]["acc"][li], color=c,
                       label={"plan": "+ outline in context", "plain": "no outline", "irrelevant": "+ irrelevant text (length-matched)"}[kind])
            ax[1].plot(range(48), res[kind]["secret_spread_by_layer"], color=c, label=kind)
        ax[0].axhline(1 / 15, color="gray", ls="--")
        ax[0].set_xlabel("story token position")
        ax[0].set_ylabel("15-way decoding accuracy (layer 32)")
        ax[0].set_title("Same plan-following texts, different contexts")
        ax[0].legend(fontsize=8)
        ax[1].set_xlabel("layer")
        ax[1].set_ylabel("RMS secret-induced spread of residual")
        ax[1].set_title("Size of the secret's footprint at story tokens")
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "m1b_context_dilution.png"), dpi=150)
        plt.close()
    jdump(summ, os.path.join(OUT, "m1_summary.json"))
    print(json.dumps({k: (v if not isinstance(v, list) else None) for k, v in summ["m1"].items() if "lp" in k}, indent=1))
    print("layer acc", [round(a, 2) for a in lay_acc])
    print("bin acc layer24", [round(a, 2) for a in acc[BL.index(24)] if a == a])



def m2_analysis(layers=(16, 24, 32)):
    """Per-story secret strength (context-driven and content-driven) vs judged rank leakage."""
    p = os.path.join(OUT, "m2.pt")
    if not os.path.exists(p):
        return
    d = torch.load(p)
    m1 = torch.load(os.path.join(OUT, "m1.pt"))
    mean = m1["mean"].float()[:, :15]
    R = fit_dirs(mean)  # [15, 48, d] unit
    judged = jload(os.path.join(RES, "e1", "judged.json"))
    out = {}
    pooled = {L: {"ctx": [], "content": [], "leak": [], "cond": []} for L in layers}
    for c, M in d["m"].items():
        words = d["words"][c]
        wi = torch.tensor([WORDS.index(w) for w in words])
        ranks = np.array([r if r is not None else np.nan for r in judged[c]["rank"]], float)
        leak = 16 - ranks  # higher = more leakage (rank 1 -> 15)
        rec = {}
        for L in layers:
            Rl = R[:, L]  # [15, d]
            own, nos = M[:, 0, L].float(), M[:, 1, L].float()
            ctx = ((own - nos) * Rl[wi]).sum(-1).numpy()
            proj_all = nos @ Rl.T  # [N, 15]
            content = (proj_all[torch.arange(len(wi)), wi] - proj_all.mean(1)).numpy()
            ok = ~np.isnan(leak)
            r1 = stats.spearmanr(ctx[ok], leak[ok])
            r2 = stats.spearmanr(content[ok], leak[ok])
            rec[L] = {"ctx_mean": float(ctx.mean()), "content_mean": float(content.mean()), "rho_ctx_leak": r1.statistic, "p_ctx": r1.pvalue,
                      "rho_content_leak": r2.statistic, "p_content": r2.pvalue, "n": int(ok.sum()), "mean_leak": float(np.nanmean(leak))}
            pooled[L]["ctx"] += list(ctx[ok])
            pooled[L]["content"] += list(content[ok])
            pooled[L]["leak"] += list(leak[ok])
            pooled[L]["cond"] += [c] * int(ok.sum())
        out[c] = rec
    # pooled, with condition-demeaning (within-condition association)
    pool = {}
    for L in layers:
        P = pooled[L]
        cond = np.array(P["cond"])
        def dm(x):
            x = np.array(x, float)
            return x - np.array([x[cond == c].mean() for c in cond])
        pool[L] = {"rho_ctx_within": stats.spearmanr(dm(P["ctx"]), dm(P["leak"])).statistic,
                   "p_ctx_within": stats.spearmanr(dm(P["ctx"]), dm(P["leak"])).pvalue,
                   "rho_content_within": stats.spearmanr(dm(P["content"]), dm(P["leak"])).statistic,
                   "p_content_within": stats.spearmanr(dm(P["content"]), dm(P["leak"])).pvalue,
                   "n": len(cond)}
        # condition-level: mean ctx strength vs 2AFC accuracy
        conds = list(out)
        accs = []
        for c in conds:
            t = [x["correct"] for x in judged[c].get("afc", []) if x["correct"] is not None]
            accs.append(np.mean(t))
        pool[L]["cond_level"] = {c: (out[c][L]["ctx_mean"], out[c][L]["content_mean"], a) for c, a in zip(conds, accs)}
        pool[L]["rho_cond_ctx_vs_acc"] = stats.spearmanr([out[c][L]["ctx_mean"] for c in conds], accs).statistic
    res = {"per_condition": out, "pooled": pool}
    jdump(res, os.path.join(OUT, "m2_summary.json"))
    # figure: condition-level
    L = 32
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for c, (cx, ct, a) in pool[L]["cond_level"].items():
        ax[0].scatter(cx, a * 100)
        ax[0].annotate(c, (cx, a * 100), fontsize=7)
        ax[1].scatter(ct, a * 100)
        ax[1].annotate(c, (ct, a * 100), fontsize=7)
    ax[0].set_xlabel("mean context-driven secret strength (layer 32)")
    ax[1].set_xlabel("mean content-driven projection on r_W (layer 32)")
    for a_ in ax:
        a_.set_ylabel("2AFC accuracy (%)")
    ax[0].set_title("Secret held in context vs leakage")
    ax[1].set_title("Secret evoked by the text itself vs leakage")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "m2_strength_vs_leakage.png"), dpi=150)
    plt.close()
    print(json.dumps({L: {k: v for k, v in pool[L].items() if k != "cond_level"} for L in layers}, indent=1, default=float))
    for c in out:
        print(c, {L: (round(out[c][L]["ctx_mean"], 2), round(out[c][L]["rho_ctx_leak"], 2), round(out[c][L]["rho_content_leak"], 2)) for L in layers})


if __name__ == "__main__":
    if "m2" in sys.argv:
        m2_analysis()
    else:
        main()
