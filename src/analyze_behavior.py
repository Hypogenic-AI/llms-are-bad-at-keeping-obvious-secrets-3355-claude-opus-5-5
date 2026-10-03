"""Statistics + figures for E1 (word secrets x plan), E2 (plot twists), M3 (interventions).
Outputs results/summary_*.json, results/table_*.md, figures/*.png"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
from judging import acc_stats, bh, cohens_h, diff_test

E1_ORDER = ["not_suppressed", "no_plan", "decoy", "premise", "irrelevant", "external_plan", "self_plan", "self_plan_secret_removed"]
E1_LABEL = {"not_suppressed": "not suppressed (ceiling)", "no_plan": "no plan (H&W anchor)", "decoy": "decoy word (known mitigation)",
            "premise": "+ 1-sentence premise", "irrelevant": "+ irrelevant text (length-matched)", "external_plan": "+ secret-free outline",
            "self_plan": "self-written outline (with secret)", "self_plan_secret_removed": "self outline, secret removed"}


def rank_stats(ranks, prior=None):
    r = np.array([x for x in ranks if x is not None], float)
    out = {"mean_rank": r.mean(), "top1": (r == 1).mean(), "top3": (r <= 3).mean(), "n": len(r),
           "ci_mean_rank": [float(np.percentile(np.random.default_rng(0).choice(r, (5000, len(r))).mean(1), q)) for q in (2.5, 97.5)]}
    return out


def table_rows(conds, judged, stories, ref="no_plan", key="afc"):
    rows, pvals = [], []
    base = acc_stats(judged[ref][key]) if ref in judged else None
    for c in conds:
        if c not in judged or key not in judged[c]:
            continue
        a = acc_stats(judged[c][key])
        row = {"cond": c, **{k: v for k, v in a.items() if not k.startswith("_")}}
        if "rank" in judged[c]:
            row.update({f"rank_{k}": v for k, v in rank_stats(judged[c]["rank"]).items()})
        if "quality" in judged[c]:
            q = [x for x in judged[c]["quality"] if x is not None]
            row["quality"] = float(np.mean(q))
            row["quality_sd"] = float(np.std(q))
        if stories is not None and c in stories:
            row["literal_rate"] = float(np.mean([contains_word(s["text"], s["word"]) for s in stories[c]]))
            row["mean_tokens"] = float(np.mean([s["n_tokens"] for s in stories[c]]))
        if base is not None and c != ref:
            d = diff_test(a, base)
            row.update({"diff_vs_ref": d["diff"], "diff_ci": [d["ci_lo"], d["ci_hi"]], "p_vs_ref": d["p_boot"], "cohens_h": float(cohens_h(a["acc"], base["acc"]))})
            pvals.append((c, d["p_boot"]))
        rows.append(row)
    if pvals:
        q = bh([p for _, p in pvals])
        for (c, _), qq in zip(pvals, q):
            for r in rows:
                if r["cond"] == c:
                    r["q_bh_vs_ref"] = float(qq)
    return rows


def fmt_rows(rows, label=None, extra_quality=True):
    lines = ["| Condition | 2AFC acc [95% CI] | n | p vs 50% | Δ vs no_plan [95% CI] | q (BH) | abstain | 'answer 1' rate | mean rank (chance 8) | top-1 | literal mention | quality |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = f"{r['diff_vs_ref'] * 100:+.1f} [{r['diff_ci'][0] * 100:+.1f}, {r['diff_ci'][1] * 100:+.1f}]" if "diff_vs_ref" in r else "—"
        q = f"{r['q_bh_vs_ref']:.3g}" if "q_bh_vs_ref" in r else "—"
        lines.append(f"| {label.get(r['cond'], r['cond']) if label else r['cond']} | {r['acc'] * 100:.1f} [{r['ci_lo'] * 100:.1f}, {r['ci_hi'] * 100:.1f}] | {r['n']} | "
                     f"{r['p_binom']:.2g} | {d} | {q} | {r['abstain_rate']:.2f} | {r['answer1_rate']:.2f} | {r.get('rank_mean_rank', float('nan')):.2f} | {r.get('rank_top1', float('nan')):.2f} | "
                     f"{r.get('literal_rate', float('nan')):.2f} | {r.get('quality', float('nan')):.2f} |")
    return "\n".join(lines)


def bar(rows, path, title, label=None, ref_lines=True):
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    acc = np.array([r["acc"] for r in rows]) * 100
    lo = acc - np.array([r["ci_lo"] for r in rows]) * 100
    hi = np.array([r["ci_hi"] for r in rows]) * 100 - acc
    ax.barh(y, acc, xerr=[lo, hi], color="#4c72b0", alpha=0.85, capsize=3)
    ax.axvline(50, color="gray", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([label.get(r["cond"], r["cond"]) if label else r["cond"] for r in rows])
    ax.set_xlabel("2AFC discrimination accuracy (%)  — 50% = no leakage")
    ax.set_xlim(30, 100)
    for yi, a in zip(y, acc):
        ax.text(a + 1, yi + 0.15, f"{a:.0f}%", fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def e1():
    judged = jload(os.path.join(RES, "e1", "judged.json"))
    stories = jload(os.path.join(RES, "e1", "stories.json"))
    rows = table_rows(E1_ORDER, judged, stories)
    # rank leakage relative to no-secret prior: mean rank of word W in no_secret stories (all 15 ranked)
    out = {"rows": rows}
    # no_secret prior: how often does a story written with NO secret get word W ranked top? (per word)
    # content-matched 2AFC (stories following the same outline/premise/passage)
    matched = {}
    for c in ["premise", "irrelevant", "external_plan", "self_plan", "self_plan_secret_removed"]:
        if c in judged and "afc_matched" in judged[c]:
            a = acc_stats(judged[c]["afc_matched"])
            matched[c] = {k: v for k, v in a.items() if not k.startswith("_")}
    out["matched"] = matched
    # per-word accuracy for no_plan & external_plan
    perword = {}
    for c in ["no_plan", "external_plan", "irrelevant", "premise"]:
        if c not in judged:
            continue
        st = stories[c]
        pw = {}
        for t in judged[c]["afc"]:
            if t["correct"] is None:
                continue
            pw.setdefault(st[t["target"]]["word"], []).append(t["correct"])
        perword[c] = {w: float(np.mean(v)) for w, v in pw.items()}
    out["per_word"] = perword
    # without literal mentions (robustness): drop trials where either text literally contains the target or the other secret
    nolit = {}
    for c in E1_ORDER:
        if c not in judged or "afc" not in judged[c]:
            continue
        st = stories[c]
        keep = [t for t in judged[c]["afc"] if not (contains_word(st[t["i"]]["text"], st[t["i"]]["word"]) or contains_word(st[t["j"]]["text"], st[t["j"]]["word"]))]
        if keep:
            a = acc_stats(keep)
            nolit[c] = {"acc": a["acc"], "n": a["n"], "ci": [a["ci_lo"], a["ci_hi"]]}
    out["no_literal"] = nolit
    jdump(out, os.path.join(RES, "summary_e1.json"))
    md = fmt_rows(rows, E1_LABEL)
    md += "\n\nContent-matched 2AFC (both stories followed the same stimulus):\n\n| Condition | acc [95% CI] | n | p |\n|---|---|---|---|\n"
    for c, a in matched.items():
        md += f"| {E1_LABEL.get(c, c)} | {a['acc'] * 100:.1f} [{a['ci_lo'] * 100:.1f}, {a['ci_hi'] * 100:.1f}] | {a['n']} | {a['p_binom']:.2g} |\n"
    md += "\n\nExcluding pairs where either story literally contains its secret word:\n\n| Condition | acc [95% CI] | n |\n|---|---|---|\n"
    for c, a in nolit.items():
        md += f"| {E1_LABEL.get(c, c)} | {a['acc'] * 100:.1f} [{a['ci'][0] * 100:.1f}, {a['ci'][1] * 100:.1f}] | {a['n']} |\n"
    open(os.path.join(RES, "table_e1.md"), "w").write(md)
    bar(rows, os.path.join(FIG, "e1_2afc.png"), "E1: secret-word leakage by planning condition (Gemma-3-12B writer)", E1_LABEL)
    print(md)
    return out


def e1r():
    """Replication with Gemma-3-27B (API writer), same stimuli."""
    p = os.path.join(RES, "e1_27b", "judged.json")
    if not os.path.exists(p):
        return
    judged = jload(p)
    stories = jload(os.path.join(RES, "e1_27b", "stories.json"))
    rows = table_rows(["no_plan", "premise", "irrelevant", "external_plan"], judged, stories)
    jdump({"rows": rows}, os.path.join(RES, "summary_e1_27b.json"))
    md = fmt_rows(rows, E1_LABEL)
    open(os.path.join(RES, "table_e1_27b.md"), "w").write(md)
    print(md)


def e2():
    p = os.path.join(RES, "e2", "judged.json")
    if not os.path.exists(p):
        return
    judged = jload(p)
    stories = jload(os.path.join(RES, "e2", "stories.json"))
    order = ["twist_unsuppressed", "twist_no_plan", "twist_irrelevant", "twist_external_plan"]
    lab = {"twist_unsuppressed": "twist known, no hide instruction", "twist_no_plan": "hide twist, no plan (anchor)",
           "twist_irrelevant": "hide twist + irrelevant text", "twist_external_plan": "hide twist + twist-free ch.1 outline"}
    rows = table_rows(order, judged, None, ref="twist_no_plan")
    four = {}
    for c in ["no_twist"] + order:
        if c not in judged:
            continue
        picks = judged[c]["pick4"]
        st = stories[c]
        corr = [int(pk == s["twist_id"]) for pk, s in zip(picks, st) if pk is not None]
        k, n = sum(corr), len(corr)
        # cluster bootstrap by premise
        byp = {}
        for pk, s in zip(picks, st):
            if pk is not None:
                byp.setdefault(s["premise_id"], []).append(int(pk == s["twist_id"]))
        cl = list(byp.values())
        rng = np.random.default_rng(0)
        bs = [np.mean(sum((cl[i] for i in rng.integers(0, len(cl), len(cl))), [])) for _ in range(5000)]
        # guessability: concentration of picks per premise in no_twist (max share)
        conc = None
        if c == "no_twist":
            mx = []
            for pid in byp:
                pp = [pk for pk, s in zip(picks, st) if s["premise_id"] == pid and pk is not None]
                mx.append(max(np.bincount(pp, minlength=4)) / len(pp))
            conc = float(np.mean(mx))
        four[c] = {"acc": k / n, "n": n, "p_binom_vs_.25": stats.binomtest(k, n, 0.25).pvalue, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   "mean_max_pick_share_per_premise": conc}
    out = {"rows": rows, "four_way": four}
    jdump(out, os.path.join(RES, "summary_e2.json"))
    md = fmt_rows(rows, lab).replace("Δ vs no_plan", "Δ vs hide/no-plan")
    md += "\n\n4-way twist identification (chance 25%):\n\n| Condition | acc [95% CI] | n | p vs 25% |\n|---|---|---|---|\n"
    for c, a in four.items():
        md += f"| {lab.get(c, c)} | {a['acc'] * 100:.1f} [{a['ci'][0] * 100:.1f}, {a['ci'][1] * 100:.1f}] | {a['n']} | {a['p_binom_vs_.25']:.2g} |\n"
    open(os.path.join(RES, "table_e2.md"), "w").write(md)
    bar(rows, os.path.join(FIG, "e2_twist_2afc.png"), "E2: plot-twist leakage in chapter 1 (within-premise 2AFC)", lab)
    print(md)


def m3():
    p = os.path.join(RES, "mech", "m3_judged.json")
    if not os.path.exists(p):
        return
    judged = jload(p)
    stories = jload(os.path.join(RES, "mech", "m3_stories.json"))
    e1j = jload(os.path.join(RES, "e1", "judged.json"))
    e1s = jload(os.path.join(RES, "e1", "stories.json"))
    judged["no_plan"] = e1j["no_plan"]
    stories["no_plan"] = e1s["no_plan"]
    judged["no_secret"] = e1j["no_secret"]
    stories["no_secret"] = e1s["no_secret"]
    order = ["no_plan"] + [c for c in ["abl_random", "abl_other", "abl_secret", "abl_other_sub", "abl_secret_sub", "abl_secret_early",
                                        "abl_secret_late", "ko_sentence"] if c in judged]
    lab = {"no_plan": "no intervention", "abl_random": "ablate random dir", "abl_other": "ablate other-concept dir",
           "abl_secret": "ablate secret dir", "abl_other_sub": "ablate other-concept subspace (14-d)",
           "abl_secret_sub": "ablate secret subspace (14-d)", "abl_secret_early": "ablate secret dir, first 100 tokens only",
           "abl_secret_late": "ablate secret dir, after token 100 only", "ko_sentence": "attention knockout of secret sentence"}
    rows = table_rows(order, judged, stories)
    # quality-controlled comparison: restrict 2AFC trials to pairs where BOTH stories were rated >= 5,
    # and compare rank leakage among stories rated >= 5 (is the leakage drop just text degradation?)
    qc = {}
    for c in order:
        if "quality" not in judged[c]:
            continue
        q = judged[c]["quality"]
        good = lambda k: q[k] is not None and q[k] >= 5
        tr = [t for t in judged[c].get("afc", []) if good(t["i"]) and good(t["j"])]
        rk = [r for r, k in zip(judged[c]["rank"], range(len(q))) if good(k) and r is not None]
        a = acc_stats(tr) if tr else None
        qc[c] = {"n_good_stories": int(sum(good(k) for k in range(len(q)))), "acc_good_pairs": a["acc"] if a else None,
                 "ci": [a["ci_lo"], a["ci_hi"]] if a else None, "n_trials": len(tr), "mean_rank_good": float(np.mean(rk)) if rk else None}
    # per-story OLS: rank of true word ~ condition + judged quality (does the secret-specific ablation
    # reduce leakage beyond an equally-degrading other-concept ablation?)
    import pandas as pd
    import statsmodels.formula.api as smf
    ols = {}
    for a, b in [("abl_secret", "abl_other"), ("abl_secret_sub", "abl_other_sub"), ("abl_secret", "abl_random")]:
        if a in judged and b in judged and "quality" in judged[a] and "quality" in judged[b]:
            df = pd.DataFrame([{"cond": c, "rank": r, "q": q} for c in (a, b) for r, q in zip(judged[c]["rank"], judged[c]["quality"])
                               if r is not None and q is not None])
            df["treat"] = (df.cond == a).astype(int)
            m = smf.ols("rank ~ treat + q", data=df).fit()
            ols[f"{a}_vs_{b}"] = {"coef_treat": float(m.params["treat"]), "ci": [float(x) for x in m.conf_int().loc["treat"]],
                                  "p": float(m.pvalues["treat"]), "coef_q": float(m.params["q"]), "n": int(len(df))}
    steer = {c: rank_stats(judged[c]["rank"]) for c in judged if c.startswith("steer")}
    steer["no_secret"] = rank_stats(judged["no_secret"]["rank"]) if "no_secret" in judged else None
    jdump({"rows": rows, "steer": steer, "quality_controlled": qc, "ols_rank_on_cond_and_quality": ols}, os.path.join(RES, "summary_m3.json"))
    md = fmt_rows(rows, lab)
    md += "\n\nQuality-controlled (only stories judged >= 5/10):\n\n| Condition | # stories >=5 | 2AFC acc on good pairs [95% CI] | trials | mean rank (good) |\n|---|---|---|---|---|\n"
    for c, v in qc.items():
        if v["acc_good_pairs"] is not None:
            md += f"| {lab.get(c, c)} | {v['n_good_stories']} | {v['acc_good_pairs'] * 100:.1f} [{v['ci'][0] * 100:.1f}, {v['ci'][1] * 100:.1f}] | {v['n_trials']} | {v['mean_rank_good']:.2f} |\n"
    md += "\n\nOLS per story: rank of true word ~ treatment + judged quality (positive coef = less leakage):\n\n| Comparison | coef [95% CI] | p | quality coef | n |\n|---|---|---|---|---|\n"
    for k, v in ols.items():
        md += f"| {k} | {v['coef_treat']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] | {v['p']:.2g} | {v['coef_q']:+.2f} | {v['n']} |\n"
    if steer:
        md += "\n\nSteering (no secret in prompt; add r_W at one layer): rank of W among 15\n\n| Condition | mean rank | top-1 | top-3 | n |\n|---|---|---|---|---|\n"
        for c, s in steer.items():
            if s:
                md += f"| {c} | {s['mean_rank']:.2f} | {s['top1']:.2f} | {s['top3']:.2f} | {s['n']} |\n"
    open(os.path.join(RES, "table_m3.md"), "w").write(md)
    bar(rows, os.path.join(FIG, "m3_interventions.png"), "M3: interventions during generation (no-plan prompt)", lab)
    print(md)


if __name__ == "__main__":
    what = sys.argv[1:] or ["e1", "e1r", "e2", "m3"]
    for w in what:
        globals()[w]()
