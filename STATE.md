# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-02T19:02:10.135672Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/llms-are-bad-at-keeping-obvious-secrets-3355-claude-opus-5-5`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase:** resource_finder. **Status:** COMPLETE (2026-10-02).

**Done:**
- 27 papers in `papers/`, including all 10 user-specified (catalogued in `papers/README.md`).
- 5 dataset resources in `datasets/`, with `.gitignore` and download instructions in `datasets/README.md`.
- 12 repos in `code/` (`code/README.md`).
- `literature_review.md` and `resources.md`.
- Per-paper deep-read notes in `logs/notes_{core_holtzman,secrets,planning,mechanism}.md`.

**Key findings:**
- Holtzman & West 2026 (arXiv 2605.10794) is the paradigm to extend. A secret word in the system prompt plus a story task gives 2AFC leakage up to 79%.
  - Open-weight Gemma-3-12B reaches 83%; Llama-3.1-8B is about 52% (no leak).
  - Rigid essays leak less, which is indirect support for "structure/plan reduces leakage".
  - "Actively hide" inverts the signal.
  - Code is unreleased; verbatim prompts and words are transcribed to `datasets/secret_words/`.
- Cautions: ConfAIde found CoT did *not* reduce leakage, so plans written while seeing the secret may not help. Spoiler Alert found thick per-beat plans reduce premature revelation.
- Mechanism tools: Arditi diff-in-means plus directional ablation (`code/refusal_direction/pipeline/utils/hook_utils.py`), CAA, LEACE, logit lens/SAE (Cywiński). Attention knockout of the secret span is the cleanest "remove it" test.

**Direction budget (top 3 kept; full scoring in literature_review.md §8):**
- **D1** plan-conditioning behavioral test (19/20).
- **D3** causal removal: direction ablation and attention knockout (18/20).
- **D2** per-token secret-activity trace, plan vs. no plan (17/20).
- Pruned:
  - D4 per-sentence foreshadowing curve: folded in as a metric.
  - D5 Taboo in-weights: confounded.
  - D6 scaling sweep: already known, heavy.
  - D7 realistic secrets: hard to score.
  - D8 decoy replication: low information gain.
  - D9 training mitigation: infeasible.

**Environment:**
- `.venv` uses uv with `[tool.uv] package=false`.
- **torch==2.9.1 + transformers<5 are required.** torch 2.14 crashed on Gemma-3 because Triton needs a C compiler, and there is none.
- GPU: RTX A6000 48GB.
- `google/gemma-3-12b-it` is pre-downloaded and smoke-tested: 24.5GB, about 11 tok/s unbatched, layers at `model.model.language_model.layers` (48, d=3840). See `logs/gemma_smoke_test.{py,log}`.
- HF token has gated access (Gemma-3 12B/27B, Llama-3.1-8B, Llama-3.3-70B). `OPENAI_API_KEY` and `OPENROUTER_KEY` are set.

**Next phase (experiment_runner):**
1. Implement the writer with batched HF generation for Gemma-3-12B, plus the 2AFC judge (both orders, verbatim prompts) and free-response guessing.
2. Replicate the H&W `no_plan` leakage on Gemma-3-12B as a sanity check (expect about 80%).
3. Run the D1 conditions: no_plan, self_plan_with_secret, external secret-free plan, plan with secret removed, self plan then secret removed, no_secret, not_suppressed, plus a rich-context control.
4. Run D2: the diff-in-means direction trace.
5. Run D3: ablation and attention knockout, with a random-direction control and a quality judge.

**Uncertainties:**
- Gemma-3-12B leakage has not yet been measured by us; one smoke sample visibly leaked "lighthouse".
- Judge choice matters: GPT-5.x gave inverted readings in H&W.
- Several author attributions for 2026 papers come from sub-reader notes.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase:** experiment_runner. **Status:** COMPLETE (2026-10-03 01:50 UTC). REPORT.md and README.md are written with actual results.

**Artifacts:**
- `planning.md`; `src/` (all code).
- `results/{e1,e1_27b,e2,mech}/`; `results/table_{e1,e1_27b,e2,m3}.md`; `results/summary_*.json`; `results/mech/m1_summary.json`, `m2_summary.json`.
- `figures/*.png`.
- The large activation `.pt` files (6 GB) are gitignored and regenerable with `src/run_gpu.py`.

**Key results (Gemma-3-12B writer, Sonnet 5.5 judge, 2AFC, length-equalised to 400 words):**
- **E1:**
  - no plan 90.0%; decoy 91.4%; irrelevant (length-matched) 91.7%.
  - premise 59.0%; **secret-free outline 52.4%**.
  - self-written outline 87.6%; self outline with secret removed 85.2%.
- **27B replication:** 91.5 → 51.5.
- **E2 twists:** hide 57.6%; irrelevant 58.6%; **outline 49.3%**.
- **M1:** secret decodable at every story position (layers ≥ 32: ≥ 95%/bin); literal token suppressed by 0.62 nats.
- **M1b:** outline shrinks the secret footprint to ~0.3× (18/18 stories); irrelevant context ~1.2×.
- **M2:** per-story strength predicts leakage (ρ = 0.23, p = 9e-12). Self-plan has a low in-context strength yet high leakage (the leak lives in the plan).
- **M3:**
  - Mean-ablation (layers 20–40) gives secret dir 67%, secret subspace 60%, other dir 86%, other subspace 81%, random 90%.
  - Early-only 87%; late-only 80%; knockout 69% (quality intact).
  - Single-layer steering is not sufficient (rank about chance).
  - Ablation degrades quality (~3.5/10). OLS vs the quality-matched other-concept control is p ≤ 3e-5.

**Deviations and issues, documented in REPORT §6:**
- Shared $100/day OpenRouter cap. Consequences: second judge dropped; 210-trial budget mode for 3 M3 conditions; `steer_L32` unjudged; no low-effort repair pass on the 27B judgments.
- Sonnet 5.5 mandatory reasoning → retry protocol added.
- Zero-ablation broke the model → banded mean-ablation.
- fp16 overflow → bf16.
- Outlines are seeded with settings (Gemma's lighthouse prior).

**Next (if continued):** second judge; frontier API writers; SAE- or probe-based ablation; head attribution.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
