# Planning: LLMs are bad at keeping obvious secrets

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs that hold private context (system prompts, hidden plot twists, confidential facts) leak it involuntarily into unrelated text. Holtzman & West (2026) showed 70–84% 2AFC detectability of a secret word in stories. If the cause is that *planning and prose are entangled*, so that the model keeps "getting ready" for what it holds, then an external plan should relieve that pressure. That would give a cheap mitigation for secret-keeping and for heavy-handed foreshadowing in AI fiction. A mechanistic account would tell us whether leakage is a removable representation or a diffuse property of conditioning.

### Gap in Existing Work
From literature_review.md §7:
1. No study tests whether conditioning on an explicit plan or outline reduces involuntary leakage. Indirect hints: rigid essays leak less; thick plans reduce premature reveals in Spoiler Alert; CoT did not help in ConfAIde.
2. No study tests plot-level secrets (a known twist) with a leakage measure that controls for twist guessability.
3. No mechanistic account exists of *in-context* secret leakage during open-ended generation. The Taboo work fine-tunes the secret in and studies Q&A hints. It is unknown whether the secret is represented at story positions, whether its strength predicts leakage, and whether ablating it removes leakage.

### Our Novel Contribution
- A plan-conditioning experiment on a white-box writer that leaks strongly (Gemma-3-12B-it). It separates *content fixing* from *context dilution* with a length-matched irrelevant-context control and a premise-only control, and adds secret-aware vs. secret-free plans.
- An extension to plot-twist secrets, using within-premise 2AFC, which is exactly 50% under the null, plus a no-twist guessability control.
- Mechanism:
  - Counterfactual teacher forcing (same story text, different secret in context) to measure where along the story the secret is represented.
  - Whether outlines dilute that representation.
  - Whether per-story strength predicts judged leakage.
  - Whether directional ablation or attention knockout of the secret removes leakage, against random-direction and other-concept-direction controls plus a quality check.

### Experiment Justification
- **E1 (word secret × plan conditions).** This is the idea's actual question. It needs the H&W positive anchor (`no_plan`), the decoy yardstick, and controls that separate dilution from content fixing.
- **E2 (plot-twist secrets).** Tests whether the effect and the plan mitigation generalise from single words to "what the model knows is coming", which is the original foreshadowing intuition.
- **E3 (mechanism).** Asks whether the secret is represented throughout generation, whether its strength predicts leakage, and whether removing it removes leakage. A purely behavioural replication would add little.

## Research Question
When an LLM writes a story while holding a secret it was told not to reveal, does conditioning on an explicit plan reduce leakage, beyond what extra context alone does? Does this hold for plot-level secrets? Is the secret linearly represented in the residual stream throughout generation, does its strength predict leakage, and does ablating it remove leakage?

## Hypothesis Decomposition
- **H1a:** `no_plan` leaks (2AFC > 50%). This is the replication anchor, expected at about 80% for Gemma-3-12B.
- **H1b:** an `external_plan` (secret-free outline) reduces leakage relative to `no_plan`.
- **H1c:** the reduction exceeds that of a length-matched `irrelevant_context` (content fixing vs. dilution) and of a one-sentence `premise` (coarse fixing).
- **H1d:** a `self_plan` written *while holding the secret* reduces leakage less, because the plan itself is a leak channel. We test this with `self_plan_secret_removed`: if leakage persists when the secret is removed but the self-written plan kept, the plan carries the leak.
- **H2:** the same ordering holds for twist secrets. A twist is leaked by openings above the no-twist prior, and a twist-free outline reduces it.
- **H3a:** the secret is linearly decodable from residuals at story positions throughout generation when only the context differs (counterfactual teacher forcing).
- **H3b:** outline context reduces that representation's strength.
- **H3c:** per-story secret strength correlates with judged leakage.
- **H3d:** ablating the secret direction during generation reduces leakage more than random and other-concept directions, without a large quality drop.

## Proposed Methodology

### Approach
White-box writer: `google/gemma-3-12b-it`, bf16 on an RTX A6000. It is the smallest model reported to leak strongly (H&W: 83%); smaller models show a floor effect. Sampling uses T=1.0, top_p=0.95, top_k=64 (Gemma defaults) and max 900 new tokens. Each story is a fresh sample with its own seed.

Judge: `anthropic/claude-sonnet-5.5` via OpenRouter at T=0, checked in the live catalog on 2026-10-02. It runs the verbatim H&W 2AFC prompt in both orders. Robustness: a second judge (`openai/gpt-5.6-terra`) on a subset. Claude-family readers were the strongest in H&W.

### Experimental Steps
1. **Stimuli.**
   - The 15 H&W words.
   - 60 secret-free outlines from Gemma, prompted with "You are a creative writer". Each outline starts with a one-line premise.
   - 60 expository passages of irrelevant context, length-matched (in tokens) to the outlines.
   - Outlines and passages containing any secret word literally are excluded.
   - Each outline, premise and passage is assigned to 2 different secrets in a balanced design: 8 per word, 120 stories per condition.
2. **E1 conditions** (8 stories/word, 120 per condition):
   - `no_secret`
   - `not_suppressed`
   - `no_plan` (dont_reveal)
   - `decoy` (H&W)
   - `premise` (dont_reveal plus a 1-sentence premise)
   - `irrelevant` (dont_reveal plus a length-matched irrelevant passage)
   - `external_plan` (dont_reveal plus a secret-free outline)
   - `self_plan` (dont_reveal; the model writes an outline, then the story, in a 2-turn chat)
   - `self_plan_secret_removed` (no secret; story from the secret-aware self-outline)
3. **E2.**
   - 30 premises × 4 mutually exclusive, equally plausible twists, generated by `anthropic/claude-opus-5.5`.
   - Gemma writes chapter 1 under these conditions: `no_twist`, `twist_unsuppressed`, `twist_no_plan` ("do not hint or foreshadow"), `twist_irrelevant` (length-matched) and `twist_external_plan` (chapter-1 outline written from the premise only, shared by all 4 twists of a premise, i.e. content-matched).
4. **E3 mechanism.**
   - **M1.** Teacher-force 40 secret-free stories under 31 contexts (15 curated secrets, 15 COCA secrets, none) and capture residuals at 12 layers. Compute:
     - diff-in-means secret directions per layer;
     - 15-way decoding accuracy and projection strength vs. story position (held-out stories);
     - final-layer log-prob of the secret token (ironic rebound).
   - **M1b.** Same text with an outline or irrelevant passage added to the context, to test dilution.
   - **M2.** For each generated E1 story, project its own residuals onto r̂_W, and correlate with judged rank leakage (Spearman).
   - **M3.** Generate `no_plan` stories with, at generation-phase positions:
     - (a) ablation of r̂_W at every layer;
     - (b) ablation of an other-concept direction (COCA word, same construction);
     - (c) ablation of a random direction;
     - (d) ablation of r̂_W only after the first 100 tokens;
     - (e) ablation of r̂_W only during the first 100 tokens;
     - (f) attention knockout of the secret sentence;
     - (g) steering: add r_W to `no_secret` generation (sufficiency).

     Measure 2AFC, rank leakage and judged quality for each.

### Baselines
- Positive anchor: `no_plan`.
- Ceiling: `not_suppressed`.
- Known mitigation yardstick: `decoy`.
- Dilution control: `irrelevant`.
- Coarse-fixing control: `premise`.
- Null for 2AFC is 50% by exchangeability: secrets are randomly assigned to outlines and samples.
- Ablation controls: random and other-concept directions, plus no ablation.

### Evaluation Metrics
- **2AFC discrimination accuracy.** 105 word pairs × 2 targets × 2 orders = 420 trials per condition; position bias reported.
- **Per-story rank of the true secret** among the 15 candidates, ranked by the judge with shuffled candidate order. Reported as mean rank (chance 8), top-1 and top-3, and baseline-corrected with `no_secret`.
- **Literal-mention rate.**
- **Judged quality** (1–10).
- **E2:** within-premise 2AFC (720 trials per condition) and 4-way twist identification (chance 25%, and exactly 25% in expectation under the null).

### Statistical Analysis Plan
- Two-sided binomial tests against 0.5 for 2AFC.
- Condition differences use a **cluster bootstrap over stories** (10k resamples), because each story appears in several trials; we report 95% CIs.
- Rank metrics: Mann–Whitney or permutation tests.
- Correlations: Spearman with a bootstrap CI.
- Benjamini–Hochberg FDR over the family of condition-vs-`no_plan` comparisons; α = 0.05.
- Effect sizes are reported as accuracy differences (pp) and Cohen's h.

## Expected Outcomes
- If the hypothesis is right: `external_plan` < `irrelevant` ≈ `no_plan`, with `self_plan` in between and `self_plan_secret_removed` > 50%.
- If dilution explains it: `irrelevant` ≈ `external_plan` < `no_plan`.
- If plans do nothing: all are about equal to `no_plan`.
- Mechanism: decodability stays high across positions. If ablation of r̂_W cuts leakage beyond the controls, the leak runs through a linear secret representation. If it does not, the secret is relayed redundantly or nonlinearly.

## Timeline and Milestones
1. Stimuli and E1 generation: about 1 h.
2. E1 judging: about 1 h.
3. E2 stimuli, generation and judging: about 1.5 h.
4. E3 M1–M3: about 2.5 h.
5. Analysis and report: about 1.5 h.

About 25% of the time is buffer for debugging.

## Potential Challenges
- **Gemma does not leak.** Then E1 cannot answer the question. Fallback: an API writer (Gemma-3-27B via OpenRouter).
- **The irrelevant passage gets used as story content.** Check overlap.
- **Judge position bias.** Both orders are run.
- **Twists guessable from the premise.** The `no_twist` control and the within-premise 2AFC handle this.
- **A single direction may be insufficient.** Report honestly, and also report the knockout and step-window results.
- **Literal mentions.** Report them and rerun the analysis without them.

## Success Criteria
All E1/E2 conditions are judged with CIs; M1–M3 are completed with controls; and REPORT.md answers both questions with honest effect sizes, including null results.

## Direction budget (from literature_review.md §8)
- **Kept:** D1 (plan conditioning, extended to plot twists per user instruction), D3 (causal removal), D2 (secret-activity trace).
- **Pruned:**
  - D4 (temporal curve): folded into D3 as the early/late ablation windows.
  - D5 (Taboo in-weights): fine-tune confound.
  - D6 (scaling): already known.
  - D8 (decoy replication): kept only as a yardstick condition.
  - D9 (training mitigation): infeasible.
- **Re-ranking:** D7 (plot-level secrets) was pruned by resource_finder. It is now included as E2 because the user instructions explicitly name it as part of the gap. It is folded into D1 as a second secret type, so the budget stays at 3 directions.
