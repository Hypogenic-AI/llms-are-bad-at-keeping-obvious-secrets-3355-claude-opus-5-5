# Literature Review: LLMs are bad at keeping obvious secrets

**Questions.**
- **(Q1)** When an LLM writes a story while holding a secret it was told not to reveal, does conditioning on an explicit plan or outline reduce thematic leakage?
- **(Q2)** Is the secret concept active in the model's internal state throughout generation, and does removing it remove the leakage?

Detailed per-paper notes (about 13.5k words) are in `logs/notes_core_holtzman.md`, `logs/notes_secrets.md`, `logs/notes_planning.md` and `logs/notes_mechanism.md`. This file is the synthesis.

---

## 1. Research area overview

Three lines of work meet in this project.

1. **Involuntary / semantic leakage.** Content in context biases generation even when it is irrelevant or forbidden:
   - Gonen et al. 2025, "semantic leakage".
   - Holtzman & West 2026: secret words leak thematically into stories.
   - Fairoze et al. 2026: digit secrets are reconstructable from benign outputs.
   - Mann et al. 2025: "don't mention X" causes ironic rebound.
   - ConfAIde (Mireshghallah et al. 2023): contextual-privacy failures.

   Theory says hidden state cannot be maintained using only the public transcript (Baldelli et al. 2026, Hangman). Without an explicit anchor, implicit secrets drift (Luo et al. 2026).
2. **Implicit planning in LMs.** Hidden states encode future content:
   - Wu et al. 2024: pre-caching vs. breadcrumbs.
   - Dong et al. 2025: prompt activations predict story attributes such as character choice.
   - ParaScopes 2025: paragraph-level plans form "just in time" at about 60–80% depth.
   - Maar et al. 2026: single-token steering flips planned rhymes.
   - Men et al. 2024 and Nainani et al. 2025.

   This is the mechanistic reading of "planning and prose generation are entangled".
3. **Reading and removing concepts in activations.**
   - Logit lens and SAEs recover Taboo secrets (Cywiński et al. 2025a,b).
   - Activation Oracles verbalize them (Karvonen et al. 2025; Bersia & Gaintseva 2026).
   - Difference-in-means directions with directional ablation (Arditi et al. 2024).
   - Steering vectors: CAA (Rimsky et al. 2023) and ActAdd (Turner et al. 2023).
   - Closed-form concept erasure (LEACE, Belrose et al. 2023).

Story-planning work (Re3, DOC, Xie & Riedl 2024, Spoiler Alert 2026) shows plans change *what* is revealed *when*. No prior work combines a held secret, explicit plans, and mechanistic removal. **That is the gap.**

---

## 2. Key papers (★ = user-specified)

### ★ Holtzman & West 2026, "Can You Keep a Secret?" (arXiv 2605.10794). This is the paradigm we extend.
- **Method**: a secret word in the system prompt plus "Write a short story (5–8 paragraphs)". A guesser LLM does (a) free-response guessing over 20 rounds and (b) **2AFC discrimination**: which of two stories, written with different secrets, holds word X. Every pair is run in both orders, giving N=420 for 15 words, and tested with a two-sided binomial test. Writer T=1.0; judge T=0.
- **Words**: 15 curated words (concrete, abstract, neutral) plus COCA and model-chosen sets (`datasets/secret_words/`).
- **Results**:
  - Leakage of 64–79% across frontier models; the literal word never appears.
  - "Actively hide" **inverts** the signal (e.g. Opus 25%), so the model is never neutral.
  - Without a secrecy instruction, discrimination is 97–100%.
  - **Scaling (open weights):** Llama-3.2-3B 50%, 3.1-8B 52%, 3.3-70B 84%; Gemma-3 4B 59%, **12B 83%**, 27B 84%.
  - Short jokes are at chance. **Rigid five-paragraph essays mostly don't leak** (Opus 53%, Sonnet 45%), which is prior evidence that structure constrains leakage.
  - User-message placement leaks more than the system prompt.
  - Decoys redirect leakage, and both words are never recovered from the same story.
- **Proposed mechanism**: attention to the secret plus a "finite entropy budget". Open-ended choices get tie-broken by the most salient context item. They predict that a richer prompt would mask the secret, and explicitly call for a mechanistic study.
- **Gaps**: closed models only; no planning condition; no internals. Code is not public.

### ★ Cywiński et al. 2025a, Taboo / ELK with mech interp (2505.14352) and ★ 2025b, Eliciting Secret Knowledge (2510.01070)
- Gemma-2-9B-it is fine-tuned to hint at a secret word without saying it; there are 20 released models, `bcywinski/gemma-2-9b-it-taboo-{word}`.
- **Logit lens** shows the secret peaking at layers about 30–37 (of 42) across many response tokens. **Gemma Scope SAE layer-32** features identify the secret.
- The secret is most represented at assistant start-of-turn tokens. On refusals it is weakly represented, i.e. it is active only when needed.
- In 2025b, black-box prefill attacks perform best. White-box logit lens and SAEs help an LLM auditor. Code: `code/eliciting-secret-knowledge` (supports an `--in-context` secret).
- Caveat: the token was suppressed but the model revealed the secret in other languages, so ablate a *concept* direction, not a token.

### ★ Mann et al. 2025, "Don't Think of the White Bear" (2511.12381)
- "Do not mention X" raises P(X) (ironic rebound), most strongly when the intervening text is on topic.
- In Llama-3-8B-Instruct, about 15–20 attention heads (amplifiers in layers 12–17) carry ≥80% of the effect.
- Predicts that a plan semantically close to the secret could *increase* leakage. Code: `code/white-bear-ironic-negation`.

### ★ Mireshghallah et al. 2023, ConfAIde (2310.17884)
- GPT-4 and ChatGPT leak private info in meeting summaries 39% and 57% of the time despite instructions.
- **Chain-of-thought did not help; it slightly increased leakage** (GPT-4 22%→24%).
- This cautions that a plan the model writes *while seeing the secret* may not help, so our design must separate secret-aware from secret-free plans.

### ★ Baldelli et al. 2026, Hangman (2601.06973)
- Impossibility theorem: an agent restricted to the public transcript cannot keep a secret both hidden and consistent. A private working memory fixes this.
- Models collapse onto favourite words, so per-word baselines are needed. Their structured private-memory block is a template for a "private plan".

### ★ Luo et al. 2026, Lack of Stable Internal Beliefs (2603.25187)
- In 20-questions, implicit unstated targets drift unless explicitly in context.
- Their fork-and-probe method (ask "which of 10 candidates is your secret" at each turn) can track secret salience mid-story.

### ★ Wu, Morris & Levine 2024, Do LMs plan ahead? (2404.00859)
- Pre-caching (computing features for the future) vs. breadcrumbs (features useful now happen to help later).
- Myopic training or attention blocking quantifies the gap. Pre-caching grows with scale, so outline effects may be size-dependent.

### ★ Dong et al. 2025, Emergent Response Planning (2502.06258)
- MLP probes on prompt activations predict global response attributes, including **which animal character a story will feature** (F1 0.72–0.86, chance 0.25), best in late layers.
- During generation, probe accuracy is U-shaped. Self-prediction is poor (F1 0.07–0.31). This is the closest analog to "secret concept active throughout generation".

### ★ Xie & Riedl 2024, Suspenseful stories via iterative planning (2402.17119)
- Staged prompting for plans: protagonist and goal, then actions and failures, then chapter summaries.
- A "generic template" plan is a ready weak-plan control.
- When asked, models readily plant clues, so the outline must be generated without the secret.

### Spoiler Alert (Sui, …, West, Holtzman 2026; 2604.09854)
- **Thick per-beat plans** with explicit "what is revealed / withheld" fields raised late-story unpredictability for Claude Sonnet 4.6 from 0.139 to 0.339; humans score 0.607.
- **100-Endings** metric: at each sentence prefix, sample 100 ending predictions and judge them. It can be adapted into a **per-sentence secret-guess curve** for temporal foreshadowing.

### Fairoze et al. 2026, Inadvertent Context Leakage (2608.19857)
- In-context digit secrets are reconstructed from ordinary outputs (4-digit secrets at 82% exact match).
- Leakage correlates with *avoidance* strength (ρ=0.95), and stronger confidentiality instructions leak more. Score both over-mention and conspicuous absence.

### Arditi et al. 2024, Refusal is a single direction (2406.11717)
- Diff-in-means direction r = μ(secret=W) − μ(secret=W′ pool) at post-instruction token positions, swept over layers.
- **Directional ablation** x ← x − r̂r̂ᵀx at all layers and positions, including generated tokens, or equivalently weight orthogonalization.
- Selection criteria: KL < 0.1 on controls, layer < 0.8L, and induction when the direction is added. Code: `code/refusal_direction`.

### Other mechanism and planning papers (skimmed)
- **CAA** (2312.06681): single-layer steering at about 40% depth, multipliers up to ±1.5, per-token projection traces.
- **ActAdd** (2308.10248): topic steering ("wedding" vector).
- **LEACE** (2306.03819): closed-form erasure.
- **Activation Oracles** (2512.15674) and **AO blind spots** (2607.23379): probes decode the hidden word at >98% even off-topic, but behavioral leakage, decodability and verbalizability can come apart.
- **Wang et al. 2026** (2604.00209): privacy steering can *backfire* non-monotonically.
- **Minder et al. 2025** (2510.13900): fine-tuned Taboo models carry a constant bias, which confounds them as comparisons.
- **ParaScopes** (2511.00180): training-free Patchscope probe at `\n\n`; plans live at about 60–80% depth.
- **Maar et al. 2026** (2601.20164): mean-difference single-token steering flips planned output; includes a regeneration test.
- **Nainani et al. 2025** (2508.18098): planning signals can just be context attention, so a secret-in-context-but-irrelevant control is needed.
- **Men et al. 2024** (2406.16033): attention knockout for lookahead.
- **Re3 / DOC** (2210.06774, 2212.10077): graded outline granularity.
- **"Don't Say It!"** (2607.00601): SAE-feature steering away from a taboo word reached only 55.7% compliance and the concept stayed recoverable; explicit reasoning improved *lexical* compliance (72%→97%) only.

---

## 3. Common methodologies
- **Writer–guesser paradigm with a judge LLM** (Holtzman & West; Cywiński; ConfAIde proxy agent; Spoiler Alert judge).
- **2AFC both-orders discrimination and detection** with binomial tests. This is the most sensitive and position-bias-free leakage metric.
- **Free-response or N-way guessing** (pass@k, rank of the true word among K candidates).
- **Residual-stream readouts**: logit lens, SAE features, linear or MLP probes, Patchscopes.
- **Causal interventions**:
  - Diff-in-means directional ablation.
  - CAA / ActAdd steering.
  - LEACE erasure.
  - Attention knockout on the secret span.
- **Plan-then-write pipelines** (Re3, DOC, Xie & Riedl, Spoiler Alert), in which an outline is generated and then the prose is conditioned on it.

## 4. Standard baselines and conditions
- **No secret** baseline ("You are a creative writer."), the 2AFC detection reference.
- **Not suppressed** ("Your word is X"), the upper bound (97–100%).
- **Don't reveal**, the main condition. **Actively hide** gives inverted leakage.
- **Decoy** (attention redirection), whose partial transfer is known.
- For interventions: a **random-direction ablation** control, a **no-intervention** control, and a **quality/coherence** check (perplexity, LLM-judge quality, EQ-Bench rubric).

## 5. Evaluation metrics
| Metric | What it measures | Notes |
|---|---|---|
| 2AFC discrimination accuracy (N=420 for 15 words) | Is the secret-specific signal in the text? | Both orders; chance 50%; two-sided binomial; below chance = avoidance leakage |
| 2AFC detection (secret vs. no-secret) | Any secret fingerprint | |
| Free-response exact-match within k rounds | Can the word be named | Strip markdown before matching |
| N-way rank / pass@k among K candidates | Graded leakage per story | Cheap with logprobs |
| Per-sentence guess curve (adapted 100-Endings) | *When* leakage happens (foreshadowing) | Core to the "foreshadowing" framing |
| Lexicon or embedding similarity to W vs. baseline | Cheap automatic proxy | Captures both toward and away |
| Projection of residual onto r_W per token; probe accuracy over positions | Is the concept active during generation (Q2) | Compare plan vs. no plan |
| Story quality (LLM judge) and coherence | Intervention side effects | Required for ablation and plan conditions |

## 6. Datasets in the literature
- No standard dataset exists for this task; it is generative.
- Holtzman & West's 15 curated, 15 COCA and model-chosen words.
- Cywiński's 20 Taboo words, with fine-tuning data on HF.
- WritingPrompts and ROCStories/TinyStories as story seeds (Dong et al.).
- EQ-Bench creative-writing prompts.
- ConfAIde tiers for realistic secrets.

---

## 7. Gaps and opportunities
1. No one has tested whether **explicit plans/outlines** reduce secret leakage. The closest evidence is indirect:
   - Rigid essays leak less (H&W).
   - Thick plans reduce premature revelation (Spoiler Alert).
   - CoT did not reduce privacy leakage (ConfAIde).
2. **No mechanistic account** of *in-context* secret leakage during open-ended generation. Taboo work studies fine-tuned secrets and Q&A hints, not stories.
3. **No causal test** that removing the secret representation (direction ablation or attention knockout) removes thematic leakage while keeping fluent prose.
4. **Temporal profile** of leakage (early setup vs. throughout, i.e. foreshadowing) is unmeasured.

---

## 8. Research directions: enumeration, scoring and pruning (Direction Budget = 3)

Scores run 1–5 on each criterion: Evidence (literature support), Relevance to the hypothesis, Information gain, Feasibility (1×A6000 48GB plus OpenAI/OpenRouter APIs). Total is out of 20.

| # | Direction | Evid. | Rel. | Info | Feas. | Total | Decision |
|---|---|---|---|---|---|---|---|
| D1 | **Plan-conditioning behavioral test**. Conditions: no plan / self-written plan *with* secret visible / plan written by a secret-free call / plan given and secret removed from context during prose / generic-template plan. 2AFC + free-response on Gemma-3-12B-it (white-box) plus 1–2 API writers. | 4 | 5 | 5 | 5 | **19** | **KEEP** |
| D2 | **Secret-activity trace**. Per-token projection onto a diff-in-means secret direction, linear probe or logit lens across story positions, comparing plan vs. no plan, and checking whether activity predicts leakage per story. | 4 | 5 | 4 | 4 | **17** | **KEEP** |
| D3 | **Causal removal**. Directional ablation of the secret concept (Arditi-style, all layers/positions), CAA subtraction, and attention knockout from story tokens to the secret span. Measure 2AFC leakage and quality, with a random-direction control. | 4 | 5 | 5 | 4 | **18** | **KEEP** |
| D4 | Per-sentence foreshadowing curve (adapted 100-Endings) | 3 | 4 | 3 | 4 | 14 | Pruned as a direction. **Use as a secondary metric inside D1** if budget allows. |
| D5 | In-weights (Taboo fine-tunes) vs. in-context secrets | 3 | 3 | 3 | 4 | 13 | Pruned: Minder et al. show fine-tunes carry a constant bias (confound), and this is tangential to planning. |
| D6 | Scaling sweep (Gemma-3 1B/4B/12B/27B; Llama 8B/70B) | 4 | 3 | 3 | 2 | 12 | Pruned: H&W already show the scaling trend; 27B/70B are heavy. Optionally include Llama-3.1-8B as a no-leak control. |
| D7 | Realistic secrets (ConfAIde, plot twists) | 2 | 3 | 3 | 2 | 10 | Pruned: harder to score, and dilutes the clean single-word design. |
| D8 | Decoy / attention-redirection replication | 4 | 2 | 1 | 5 | 12 | Pruned: already done by H&W; low information gain. |
| D9 | Training-time mitigation (fine-tune to hide) | 2 | 2 | 3 | 1 | 8 | Pruned: infeasible in budget and off-hypothesis. |

**Kept: D1, D3, D2** (ranked). D4 is folded in as a metric. The ranking should change only if, for example, Gemma-3-12B fails to reproduce baseline leakage (then switch writer to Gemma-3-27B in 8-bit or an API model for D1).

---

## 9. Recommendations for the experiment

**Primary writer (white-box):** `google/gemma-3-12b-it`.
- Pre-downloaded and smoke-tested: bf16, 24.5GB, residual hooks at `model.model.language_model.layers[i]`, 48 layers, d=3840.
- Holtzman & West report 83% leakage, and our first smoke-test sample with secret "lighthouse" was a coastal-watcher story with nautical charts.
- Negative control: `meta-llama/Llama-3.1-8B-Instruct` (52%).
- Optional API writers via OpenRouter: Llama-4-Maverick or DeepSeek-V3.2 (both about 78–79%).

**Judge/guesser:** use a strong API model (e.g. GPT-4.1/GPT-5-class via `OPENAI_API_KEY`, or Claude via OpenRouter) at T=0. Use the verbatim prompts in `datasets/secret_words/prompts_holtzman2026.json`. Stronger readers extract more signal (Opus read DeepSeek's stories at 87%). Avoid GPT-5.x as the *only* judge, since it produced inverted 2AFC readings in H&W.

**D1 design (plan conditions)**, all with the "don't reveal" secret prompt unless noted:
1. `no_plan`: replicates H&W.
2. `self_plan_with_secret`: the writer first writes an outline while seeing the secret, then writes the story from it (secret still in context). This is the ConfAIde CoT analog.
3. `external_plan`: an outline produced by a **secret-free** call (same model, "You are a creative writer"), then given to the secret-holding writer: "Follow this outline exactly". This is the key test of whether a plan relieves the model of the need to prepare.
4. `external_plan_secret_removed`: the same outline with no secret in context. This is the floor and shows whether the plan alone produces chance-level discrimination.
5. `self_plan_then_secret_removed`: an outline written *with* the secret, then prose written *without* it. This measures how much leakage is carried by the plan itself.
6. Optional: `generic_template_plan` (Xie ablation), and plan granularity (3-point vs. thick per-beat with a "withheld" field, as in Spoiler Alert).
7. Controls: `no_secret`, `not_suppressed`.

Use the 15 curated words, at least 3 samples per word per condition at T=1.0, and 2AFC across all pairs in both orders. Also report free-response accuracy, the N-way rank, and story quality.

**D2/D3 design (mechanism)**:
- **Direction**: build r_W per word by diff-in-means of residuals at post-instruction / start-of-turn positions, comparing "secret=W" prompts with a pool of other-secret prompts. Alternatively use a contrast of W-themed vs. neutral text. Sweep layers at about 25–80% depth, roughly layers 12–38 of 48.
- **Trace (D2)**: cos(h_t, r_W) or a probe logit for W at every generated token. Compare `no_plan` vs. `external_plan` vs. `no_secret`, and correlate per-story activity with judge-detected leakage.
- **Removal (D3)**:
  - (a) Directional ablation of r_W at all layers and positions during prose generation, via `get_direction_ablation_input_pre_hook` in `code/refusal_direction/pipeline/utils/hook_utils.py`.
  - (b) Attention knockout of the secret-word tokens for all generated positions. This is the cleanest "remove the secret from the working state" intervention.
  - (c) LEACE as a robustness check.
  - Controls: a random direction of equal norm, and a check that the direction-added-to-no-secret condition *induces* W-themed stories (sufficiency).

**Methodological cautions**:
- Count avoidance (below-chance) as leakage too.
- Run both 2AFC orders.
- Strip markdown before free-response matching.
- Disclose the semantic-category prior: concrete words leak most.
- Steering can backfire non-monotonically, so sweep strengths and report quality.
- Plans are themselves extra prompt entropy. Include a length-matched "irrelevant rich context" control, such as a WritingPrompts premise without an outline, to separate "plan" from "more context" (H&W's entropy-budget hypothesis).
