# LLMs are bad at keeping obvious secrets — but a plan fixes it

This project asks whether an LLM that holds a secret leaks it less into a story when it writes from an explicit outline, and what the leak looks like inside the model.

- **Writer:** Gemma-3-12B-it (white-box), with a Gemma-3-27B API replication.
- **Judge:** Claude Sonnet 5.5, using two-alternative forced choice (2AFC) in both presentation orders.

It extends Holtzman & West (2026, arXiv 2605.10794), which reported the leak behaviourally, in three ways:
- a plan/outline manipulation with dilution controls;
- plot-twist secrets;
- a residual-stream analysis of the mechanism.

## Key findings
- **A secret-free outline eliminates leakage.**
  - 2AFC discrimination falls from **90%** (no plan) to **52%** (chance) for Gemma-3-12B, and from 92% to 52% for Gemma-3-27B.
  - A **length-matched irrelevant passage does nothing** (92%), so the effect is not context dilution.
  - A one-sentence premise helps partially (59%).
  - The known decoy mitigation does nothing for Gemma (91%).
- **A plan the model writes while holding the secret does *not* help** (88%). The leak moves into the plan: a writer that *never saw the secret* and follows that self-written outline produces stories still discriminable at **85%**.
- **Plot twists behave the same way.** Chapter-1 openings written while hiding a known twist leak it (57.6% within-premise 2AFC). Irrelevant context leaves this unchanged (58.6%); a twist-free chapter outline removes it (**49.3%**).
- **Mechanism, part 1: the secret is present throughout.** Teacher-forcing identical story text under different secrets shows the secret is linearly decodable from the residual stream at every story position, about 100% at layers ≥ 26 for 800+ tokens. The *literal* word is suppressed (−0.62 nats), but the concept is not.
- **Mechanism, part 2: an outline shrinks the secret's footprint.** On the *same* text, an outline in context cuts the secret's footprint at story tokens to about 30% (18/18 stories, p < 1e-5). Irrelevant context increases it.
- **Mechanism, part 3: removing the secret during writing removes much of the leak.**
  - Mean-ablating the secret direction (layers 20–40) cuts leakage to **67%**, and the 14-d secret subspace to **60%**. Other-concept and random directions give 86% and 90%.
  - These ablations degrade prose, but the effect survives a quality-matched control.
  - Attention knockout of the secret sentence gives 69% with *no* quality loss.
  - Per story, stronger in-context secret representation predicts more leakage (ρ = 0.23, p = 9e-12).

See **[REPORT.md](REPORT.md)** for full methods, statistics, limitations and discussion.

## Reproduce
```bash
uv venv && source .venv/bin/activate && uv sync      # torch==2.9.1, transformers<5 (see pyproject.toml)
export OPENROUTER_KEY=...; export HF_TOKEN=...       # Gemma-3 is gated on HF
python src/e2_stimuli.py                              # plot-twist premises (Claude Opus 5.5)
python src/run_gpu.py e1 e2 m1 m1b m3 m2 steer        # all Gemma generation + activation work (1x A6000, ~5 h)
python src/api_writer.py                              # Gemma-3-27B replication via OpenRouter
python src/e1_judge.py results/e1/stories.json results/e1/judged.json --quality
python src/e2_judge.py
python src/e1_judge.py results/mech/m3_stories.json results/mech/m3_judged.json --quality
python src/analyze_behavior.py && python src/mech_analysis.py && python src/mech_analysis.py m2
```
All judge calls are cached in `results/cache/api_cache.sqlite`, so re-running the analysis does not re-bill.

## Layout
- `src/common.py`: stimuli, prompts, cached OpenRouter client, judges.
- `src/gemma.py`: Gemma engine; batched generation with ablation, knockout and steering hooks; teacher-forced residual capture.
- `src/e1_generate.py`, `src/e2_generate.py`, `src/api_writer.py`: writers.
- `src/e1_judge.py`, `src/e2_judge.py`, `src/judging.py`: 2AFC, rank, quality and statistics.
- `src/mech_m1.py`, `src/mech_m2.py`, `src/mech_m3.py`, `src/mech_analysis.py`, `src/m3_sanity.py`: mechanism experiments.
- `results/`: all stories, judgments, summaries and tables (`table_*.md`).
- `figures/`: plots.
- `planning.md`: preregistered plan.
- `logs/`: run logs.
