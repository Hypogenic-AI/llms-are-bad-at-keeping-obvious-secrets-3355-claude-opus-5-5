# Literature notes: planning, foreshadowing, and future-content representations

Project: "LLMs are bad at keeping obvious secrets" — does an explicit plan reduce thematic leakage of a held secret into generated stories, and is the secret concept persistently active in internal state (and does removing it remove leakage)?

---

## [DEEP] Wu, Morris & Levine (2024). Do Language Models Plan Ahead for Future Tokens? (arXiv 2404.00859, COLM 2024)

**Research question.** When hidden states at position t are useful for predicting tokens at t+tau, is this because the model *deliberately* computes future-useful features ("pre-caching", driven by off-diagonal gradient terms dL_{t+tau}/dtheta_t) or because the features useful now are incidentally useful later ("breadcrumbs")?

**Methodology.** Formalizes *myopic training*: gradient descent with the off-diagonal (past-position) gradient terms removed, implemented by a "myopic attention mechanism" in which past keys/values are computed from detached copies of past hidden states. Defines the *myopia gap* (loss of best myopic model minus loss of best vanilla model) and the *local myopia bonus* (how much a vanilla model's past states could be re-optimised for the present). Baseline "transformer bigram" zeroes all off-diagonal K/V. Theorem: under forward-bias, strong convexity and smoothness, myopic descent converges to a myopic model.
- Synthetic task D_p: y_n = z_n * sum_{i=1..10} sin(10 x_{n-i}) + (1-z_n) x_n; computing sin(10 x_n) at position n is useless now but useful later. Linear probes per layer for sin(b x_{n-i}).
- Natural language: GPT-2 (124M) trained from scratch on 4.6M MS MARCO sequences (len 64); per-position loss on PG-19. Scaling: Pythia 14M to 2.8B, fine-tuned (from pretrained checkpoints) vanilla vs myopic on 10M Pile sequences of 64 tokens; evaluated on LAMBADA, PIQA, SciQ, ARC-Easy.

**Models.** 2-layer transformer (d=128); GPT-2 124M architecture; Pythia-14M ... Pythia-2.8B (EleutherAI).

**Datasets.** Synthetic D_p; MS MARCO; PG-19 (deepmind/pg19); The Pile; LAMBADA, PIQA, SciQ, ARC-Easy. Also multiplication with filler tokens (appendix).

**Key results.**
- Synthetic: vanilla models clearly pre-cache (layer 1 encodes sin(bx_n) with R^2 up to ~0.9; layer 2 encodes the sum over the past 10). Myopic models fail: normalized Huber loss ~1.0-1.26 (trivial baseline 1.26) vs 0.003-0.096 for vanilla.
- GPT-2: CE vanilla 3.28, myopic 3.40 (gap 0.12), local-myopic 3.26 (bonus 0.02), transformer bigram 5.33. Myopic is *better* at the first positions but falls behind as context grows. => mostly breadcrumbs at small scale.
- Pythia: myopia gap grows with scale (e.g., LAMBADA ~0.58 vanilla vs ~0.40 myopic at 2.8B), i.e., pre-caching becomes non-negligible in larger models.

**Code.** https://github.com/wiwu2390/FutureGPT2-public

**Limitations.** Small models, short sequences (64 tokens); myopic Pythia runs are fine-tunes from vanilla checkpoints; no analysis of long-form/narrative planning or of specific semantic content being "carried forward".

**Use for our project.**
- Conceptual framing: secret leakage could be *breadcrumbs* (the secret is simply in context and its representation is reused at every step) vs *pre-caching/planning* (model prepares for a future reveal). An outline should only help in the second case; a null effect of outlines would favour the breadcrumbs reading. State this dichotomy explicitly in hypotheses.
- Pre-caching grows with scale -> test outline effect across model sizes (e.g., Qwen2.5 / Llama-3 1B-8B-70B); predict larger models foreshadow more and benefit more from a plan.
- Probing recipe: linear probes per layer and per lag (position t predicting feature at t+tau) with a careful control that the probed feature is not linearly computable from inputs; we can probe "secret identity" at each story token and compare with/without outline.
- A cheap intervention analogous to myopic attention: block attention from story tokens to the secret token(s) (or zero their K/V) after the first k tokens and measure leakage, separating "persistent read from context" vs "carried forward in residual".

---

## [DEEP] Dong, Zhou, Liu, Yang & Lu (2025). Emergent Response Planning in LLMs (arXiv 2502.06258, ICML 2025)

**Research question.** Do LLM hidden representations of the *prompt* (before any response token) encode global attributes of the full upcoming response, i.e., is there "emergent response planning"?

**Methodology.** Generate greedy responses, compute an attribute g(y), and train probes h(H^l_x) -> g(y) on prompt-final-token representations at each layer. Probes: 1-hidden-layer ReLU MLPs, hidden size grid {1..1024}, 400 epochs, 60/20/20 split, grid over layers, 3 seeds. Six tasks in three classes: structure (response length; # reasoning steps), content (animal character introduced in a story continuation; MC answer given after explanation), behavior (answer correctness/confidence; factual consistency). Shortcut control: attribute must not be inferable from first token (e.g., animal must not be in first two words; "explain first, then answer"). Additional analyses: cross-dataset transfer, probe-size saliency, layer-wise curves, scaling, probing at equidistant positions *during* generation (up to the token before the animal word), and comparison with verbalized self-prediction.

**Models.** Llama-2-7B(-Chat), Llama-3-8B(-Instruct), Mistral-7B(-Instruct), Qwen2-7B(-Instruct). Scaling: Llama-2-chat 7/13/70B, Llama-3-Instruct 8/70B, Qwen2-Instruct 7/72B, Qwen2.5-Instruct 1.5/32/72B.

**Datasets (HF).** stingning/ultrachat, tatsu-lab/alpaca(Eval), openai/gsm8k, deepmind/math_dataset, roneneldan/TinyStories, Ximing/ROCStories, tau/commonsense_qa, allenai/social_i_qa, openlifescienceai/medmcqa, allenai/ai2_arc, CREAK, FEVER. Story task prompt: "Here's the first sentence of a story: {data} Continue this story with one sentence that introduces a new animal character." Labels = top-4 most frequent animals per model (4-way, class-balanced); augmentation by truncating responses a few tokens before the animal word.

**Metrics.** Spearman/Kendall/Pearson (regression), F1 (= accuracy under class balance).

**Key results.**
- Character choice (4-way, chance 0.25): F1 ~0.72-0.86 in-dataset across models; cross-dataset (TinyStories->ROCStories) still well above chance (~0.4-0.6).
- Response length: Spearman ~0.80-0.85 for instruct models (base ~0.4-0.6). Qwen2-7B-Instruct: S=0.85.
- Content attributes peak in *later* layers; structure in middle layers; behavior uniform after the first few layers.
- Within-family scaling: bigger = better planning (e.g., character F1 Llama-2-chat 7B 0.74 -> 70B 0.79).
- **During generation**, character-choice probe accuracy is U-shaped: high at prompt end (~0.7-0.83), dips mid-generation (~0.55-0.7), recovers near the reveal (~0.75-0.87).
- Verbalized self-prediction far worse than probes (character choice self-estimate F1 0.07-0.31 vs probes 0.72-0.86).

**Code.** No repository link found in the paper.

**Limitations.** Correlational only (no causal intervention); greedy decoding; short responses (one-sentence story continuation); top-4 class restriction; shortcuts only partially controlled.

**Use for our project.**
- Direct template: the "animal character" task is basically our setup with the sign flipped. Probe for the secret word (or its semantic category) from the residual stream at every story token; the plan representation existing at the prompt end is expected, the question is whether it *persists* mid-story (the U-shape suggests it weakens mid-text and returns near a "reveal" point — foreshadowing would show as rising probe accuracy late in the story).
- Use late layers for content (secret word) probes; small MLP or linear probe is enough (saliency plateau ~32 hidden units).
- Compare probe trajectories with vs without an explicit outline in context: if the outline "relieves" planning, the secret signal should drop mid-story when an outline is present.
- Block first-token shortcuts: exclude stories where the secret appears verbatim; label by secret identity, not by surface words.
- Use the self-report vs probe gap as a secondary measure: ask the model whether it hinted at the secret vs what probes show.
- Average labels over multiple samples rather than greedy when stories are sampled (their 6.1 suggestion).

---

## [DEEP] Xie & Riedl (2024). Creating Suspenseful Stories: Iterative Planning with Large Language Models (arXiv 2402.17119, EACL 2024)

**Research question.** Can a theory-grounded, zero-shot iterative planning procedure make LLMs write suspenseful stories (which plain prompting fails to do)?

**Methodology.** Grounded in Gerrig & Bernardo (1994) (suspense = reader sees protagonist's escape paths shrink) and Branigan's *disparity of knowledge* (reader knows something the character does not). Three stages, all via prompting:
1. Background setup: sample genre (50 thriller sub-genres, App. A) -> protagonist name/occupation -> concrete goal -> dire situation if goal fails -> intro paragraph.
2. Outline planning (iterative, adversarial): "Tell me about a concrete action the protagonist is most likely to take" -> "The protagonist tries $action1 but still fails... Tell me what this reason could be" -> next action conditioned on all previous failures (decreasing likelihood); final iteration has no failure (success).
3. Detail elaboration: each (action, reason) pair -> chapter summary -> sequence of events -> elaborated prose; protagonist must be unaware the plan will fail.
Optional knobs: (a) *clue setup* — ask the LLM to plant small clues for the upcoming failure (explicit foreshadowing); (b) information revelation *before* vs *after* the fact.
Ablations: #1 replaces detail elaboration with a single "write a full suspenseful story based on the story summary [actions+reasons]" prompt; #2 additionally replaces the concrete outline with a generic template summary ("tries a first action, fails due to a reason...").

**Models.** gpt-3.5-turbo-0613 (system prompt "You are a creative storyteller."; "Use no more than [n] sentences" for intermediate steps); Llama-2-13b-chat for transfer; Re3 as long-form baseline.

**Datasets.** None (fully zero-shot; no story corpus). Evaluation via Prolific human raters (90 participants / 30 story pairs / 30 raters per pair; Fleiss' kappa reported as "fair").

**Metrics.** Pairwise preference (win/lose/tie) on suspense, novelty, enjoyment, logical sense, naturalness; Wilcoxon sign test.

**Key results.**
- vs plain ChatGPT prompt: suspense win 84.9% vs 11.8%; novelty 76.6/13.6; enjoyment 67.9/19.4; logic 49.2/27.8 (p<0.05). Llama-2-13b-chat: suspense 73.9/15.7. vs Re3: suspense 81.1/6.8.
- Ablations: full vs #1 suspense 75.0/19.2; full vs #2 80.4/9.9; #2 (generic outline) still beats baseline 72.3/12.7 — even a skeletal plan structure helps.
- Outline audit: actions relevant to goal 96.5%; failure reasons plausible 89.8%; decreasing likelihood perceived only 55.1%.
- *Clues/foreshadowing*: stories with planted failure clues judged more suspenseful 57.9% vs 10.9% (p<0.05). Revealing failure reason before vs after: 42.1% vs 36.3% (n.s.).
- Suspense correlates with reader empathy.

**Code.** No code link in the paper.

**Limitations.** Human-preference only, no automatic metrics; one main LLM (GPT-3.5); English/Western narrative theory; foreshadowing is *requested*, not measured as an emergent property.

**Use for our project.**
- Ready-made outline construction pipeline: (protagonist, goal, dire situation) -> ordered action/obstacle beats -> per-beat chapter summaries. We can generate the outline with a separate (secret-free) call and paste it into the writer's context, giving a clean "plan provided" condition. Their Ablation #2 template is a good "weak/generic plan" control; Ablation #1 prompt is a good "full outline in one prompt" condition.
- Important confound: models readily plant clues for future events when asked, and clue-planting raises perceived suspense. A plan that *contains the secret's role* would encourage foreshadowing; outlines must be secret-free (or include an explicit "the secret never appears" beat) to test relief of planning pressure.
- The "disparity of knowledge" framing maps onto our setup: the model (narrator) knows the secret and the reader shouldn't; leakage is effectively unintended dramatic irony.
- Pairwise human/LLM-judge evaluation with win/lose/tie + Wilcoxon sign test is a simple protocol for "which story hints more at X".
- Two-stage generation (plan in one call, prose in another) lets us separate the plan's information from the secret.

---

## [DEEP] Sui, Zhu, Cheng, West, So, Long & Holtzman (2026). Spoiler Alert: Narrative Forecasting as a Metric for Tension in LLM Storytelling (arXiv 2604.09854, preprint)

**Research question.** Can narrative tension be measured structurally as *unpredictability of the ending* at every point in a story, and can a thick structural plan make LLM stories hold tension (i.e., stop resolving/revealing prematurely)?

**Methodology.**
- *100-Endings metric*: split story into sentences (regex `[.!?][""’)]*\s+(?=[A-Z“"(\[])`), keep positions where 10-99% of tokens are revealed (skip positions after <10-word sentences; stories <=5,000 words). At each position, Qwen3-32B (vLLM, T=1.2, top-p 0.95, max 150 tokens, n=100, thinking off) gets: system "You are a fiction writer. Write only story text." user "In 2-3 sentences, describe how you think this story ends. No commentary, no explanation, no preamble. STORY SO FAR: {prefix}". A separate judge Qwen3-8B (T=0, max 10 tokens) judges each of the 100 predicted endings independently against the true continuation (Y if "general direction"/broad plot trajectory matches). no-rate(i) = fraction not matching.
- Story statistics: mean no-rate; *late* no-rate (>=80% revealed); inflection rate (fraction of sharp reversals with vertex angle <=30/60/120 deg after unit-square rescaling of the smoothed curve); post-spike retention (fraction of a local peak retained at the min within next 10 positions).
- *Pipeline*: Step 0 warmup — extract 7 scene-level beats (action, technique, what made it memorable) from a reference story (title mode: Omelas/The Lottery/Yellow Wallpaper; or full New Yorker text); Step 1 — adapt into a "thick" to-do list per beat: narrative function, *what information is revealed or withheld*, tension mechanism, stakes escalation; Step 2 — "Here is a structural to-do list... Write the story now. Follow the structural beats closely." (system prompt emphasises sustaining tension). T=0.7, min_p 0.1, max 4,000 tokens, via OpenRouter, reasoning disabled.

**Models.** Generators: Claude Sonnet 4.6, Claude Opus 4.6, GPT-5.2 (+ top-10 EQ-Bench models' zero-shot outputs). Metric: Qwen3-32B (forecaster) + Qwen3-8B (judge). EQ-Bench judge: Claude Sonnet 4.

**Datasets.** New Yorker fiction 1945-2019 (261 stories; private, Textual Optics Lab); Tell Me a Story (Huot et al. 2025; 230 stories, 100 sampled); WritingPrompts human stories from the Ghostbuster dataset (Verma et al. 2024; 1,000 -> 100 sampled); StoryStar (100 scraped); EQ-Bench Creative Writing v3 32 prompts and model outputs (github.com/EQ-bench/creative-writing-bench).

**Key results.**
- Corpus mean no-rate: New Yorker 0.765, Tell Me a Story 0.693, WritingPrompts 0.671, StoryStar 0.656, top-10 LLMs 0.630 (lowest). Late no-rate: NY 0.607 vs LLMs 0.215; retention 52% vs 23%. EQ-Bench rubric reverses this (LLMs 81-84 > NY 78.7).
- Pipeline (32 EQ-Bench prompts): Sonnet 4.6 mean NR 0.606->0.747, late 0.139->0.339, retention 16.7%->35.5%; Opus 4.6 0.611->0.697 (late +0.111); GPT-5.2 0.694->0.740 (late +0.114). EQ-Bench: Sonnet +3 (85.03, #1), Opus -1.1, GPT-5.2 -2.6.
- Metric stability: 4 reruns, SD <=0.001 on mean and late no-rate.
- Case study: zero-shot Sonnet romance explicitly confirms the attraction at 64% -> no-rate collapses to 0.29; pipeline version keeps it >0.85 through 85% by leaving chemistry as subtext.

**Code.** No pipeline/metric repo given (EQ-Bench repo only). Prompts fully specified in App. C-D.

**Limitations.** Predictability is necessary but not sufficient for tension; LLM-judge circularity (mitigated by using Qwen for evaluation only); no ablation of which plan component matters (thickness vs content); New Yorker data not public.

**Use for our project.**
- Most directly relevant evidence that *a thick plan reduces premature revelation*: LLMs "cannot defer closure" zero-shot, and a beat-level to-do list that explicitly states what is *withheld* at each beat roughly doubles late-stage unpredictability. This supports our hypothesis; our outline condition should include a per-beat "revealed/withheld" field.
- Adapt 100-Endings into a "100-Guesses" leakage metric: at each sentence prefix, sample N (e.g., 50-100) guesses of the secret word from a separate model (T~1.0-1.2) and record the fraction/rank matching the secret (exact or embedding-similar). Gives a per-position leakage curve; compare curve shape (late spikes = foreshadowing) between outline vs no-outline conditions. Use a different model family for guesser/judge than for the writer.
- Use their sentence splitter, position window (10-99%), and stability check (repeat runs) directly; report mean, late-stage (>=80%) and retention-style statistics.
- Writing prompts: EQ-Bench's 32 prompts are a convenient, public seed set for stories; WritingPrompts/TMAS for human baselines.

---

## [DEEP] Pochinkov, Volkova, Vasileva & Chereddy (2025). ParaScopes: What Do Language Model Activations Encode About Future Text? (arXiv 2511.00180, preprint)

**Research question.** "Planning Decodability Hypothesis": planning at scale X exists if information about scale-X content is decodable from activations before it is generated. Is the next paragraph (or the whole output's outline) decodable from the residual stream at a paragraph boundary?

**Methodology.** Residual Stream Decoders (RSDs):
- *Continuation ParaScope* (training-free, Patchscopes-style): build blank context `<bos>\n\n`, overwrite all-layer residuals of the `\n\n` token with those saved from the original generation at the paragraph boundary, generate up to 128 tokens.
- *TAE ParaScope*: linear map from normalized residual diffs (attn+MLP outputs of the final 12 of 28 layers = 24 sub-layers; Welford normalization) to the 1024-d SONAR sentence embedding of the next paragraph; decode with SONAR decoder. 100k training samples, bs 1024, lr 2e-5 (x0.8/epoch), wd 1e-7, 10 epochs.
- *TAE Outline RSD*: same linear map from the residual at the newline after the prompt to SONAR embedding of an LLM-written bullet outline of the full response.
- Baselines: blind (`<bos>\n\n` only), cheat-K (K true upcoming tokens revealed, K in {1,5,10}), regeneration with full context (upper bound), SONAR auto-decode.
- Analyses: *layer scrubbing* — splice layers [1..K] from condition A with [K+1..N] from condition B at the `\n\n` token ("Tell me about cleaning your house in 50 words, then tell me about [polycystic kidney disease | monster trucks] in 50 words...") and see which topic comes out; *temporal dynamics* — Continuation ParaScope applied at each token within +/-10 of `\n\n` (Gemma-2-9B), 64 samples per position, cosine to paragraph 1 vs 2; token-replacement control (insert `\n\n` at other positions).

**Models.** Llama-3.2-3B-Instruct (T=0.3, main); Gemma-2-9B (temporal analysis; layer scrubbing figure uses a 42-layer model, i.e., Gemma-2-9B); Gemma-2-27B (prompt generation); "Llama 3.2 70B" (sic) and Gemma-3-27B-Instruct for outlines; SONAR encoder/decoder; Qwen3-Embedding-0.6B, all-mpnet-base-v2, BLEURT-20; GPT-4o-mini as rubric judge.

**Datasets.** HuggingFaceFW/fineweb-edu chunks -> Gemma-2-27B converts each into "Write a [type], titled [name], which includes [topics], approximately [length]" prompts -> up to 1M Llama-3.2-3B generations (T=0.3). Non-narrative (expository) text.

**Metrics.** Cosine similarity (Qwen3 embed), BLEURT, LLM-judge rubrics (subject match -1..4, detail preservation -1..3; outline coverage/order/entities/details).

**Key results.**
- Cosine: TAE 0.55, Continuation 0.43, blind 0.20, cheat-5 0.50, regenerated 0.82, auto-decoded 0.94 -> future-paragraph info ~ equivalent to 5 future tokens.
- Subject match >=2: TAE 76%, Continuation 43%, cheat-10 51%, blind 4%, regenerated 99%. Moderate details: Continuation 15% vs TAE 3% (cheat-5 17%).
- Outline RSD: subject >= minor match 61%, but details only 24% -> weak outline-level plans in a 3B model.
- Layer scrubbing: layers 0-25 of 42 contribute little (delta < 0.05); layers 25-35 (~60-80% depth) carry the next-paragraph topic (delta 0.15-0.25).
- Temporal: similarity to the *next* topic: pre-transition 0.12, at `\n\n` 0.48, post 0.65 -> plans formed "just in time" at the boundary (relatively myopic). Inserting `\n\n` elsewhere sometimes triggers plan formation, but not generally.

**Code.** None linked in the paper.

**Limitations.** Mostly one small model; expository not narrative text; decodability != causal use; complex probes might infer plans from correlations; single-token readout may miss distributed plans.

**Use for our project.**
- Continuation ParaScope / Patchscope is a cheap, training-free probe for "what is the model about to write about": patch the residual at each sentence/paragraph boundary of a secret-holding story into a blank context and ask "the secret word is" or let it continue; measure how often the secret (or its semantic field) surfaces. Compare outline vs no-outline conditions.
- Read out at 60-80% depth; collect at paragraph boundaries (`\n\n`) where plan information concentrates.
- Expect that in small models plans are local (next paragraph). If the secret is decodable at *every* boundary and not just near hint sentences, that supports "secret is persistently active" (our Q2) rather than "planning for a reveal."
- Baseline design worth copying: blind and cheat-K baselines calibrate how much secret information the probe extracts relative to revealing K tokens of text.
- Layer-scrubbing (splice layers between secret-A and secret-B runs) is a clean way to localize where the secret concept that drives leakage lives.

---

## [SKIM] Yang, Tian, Peng & Klein (2022). Re3: Generating Longer Stories With Recursive Reprompting and Revision (arXiv 2210.06774, EMNLP 2022)

- **Question/method.** Plan-Draft-Rewrite-Edit for 2,000-2,500-word stories. Plan: prompt GPT3-Instruct-175B to expand a premise into setting, characters, and a 3-point outline. Draft: GPT3-175B (davinci) generates 256-token continuations with a structured prompt that re-injects relevant plan items + recent story at each step (4 continuations per outline point; ends via Insert API to "The End."). Rewrite: rerank continuations for coherence and premise relevance; Edit: fix factual inconsistencies.
- **Data/eval.** 100 GPT3-generated premises; AMT pairwise judgments. Baselines: ROLLING (rolling-window GPT3-175B) and ROLLING-FT (fine-tuned on WritingPrompts stories >=3,000 tokens).
- **Result.** vs ROLLING: coherent 60.0 vs 45.7%, premise-relevant 64.0 vs 44.0%, humanlike 83.3 vs 74.0%.
- **Code.** https://github.com/yangkevin2/emnlp22-re3-story-generation
- **Use for our project.** Canonical "plan injected into context" template: premise -> setting/characters/numbered outline, with the relevant outline point re-surfaced in the prompt for each passage. Supports a design where the outline is pasted into the writer's context step-by-step (stronger "relief" manipulation than a one-shot outline). A plan that is re-injected each step keeps "what comes next" explicit, which is exactly the condition under which our hypothesis predicts less foreshadowing.

---

## [SKIM] Yang, Klein, Peng & Tian (2022/2023). DOC: Improving Long Story Coherence With Detailed Outline Control (arXiv 2212.10077, ACL 2023)

- **Question/method.** Shift creative burden from drafting to planning: a *detailed outliner* recursively expands a brief outline into a hierarchical outline (depth 3, branching 2-5) with setting/characters per item, filtered for relevance/coherence; a *detailed controller* (OPT-350m FUDGE-style classifier trained contrastively on summary/passage-prefix pairs) keeps generated passages faithful to each outline item. Drafting with OPT-175B; context window 1,024; ~3,500-word stories.
- **Eval.** Pairwise Surge AI annotation of 1,000-1,500-word passages per top-level outline item from 20 stories.
- **Result.** vs Re3: coherent 67.6 vs 45.1%, outline-relevant 65.3 vs 37.1%, interesting 60.1 vs 39.4%. Ablations: both outliner and controller matter.
- **Code.** https://github.com/yangkevin2/doc-story-generation
- **Use for our project.** Gives a graded "plan granularity" manipulation: none -> 3-point outline (Re3) -> hierarchical detailed outline (DOC). Our hypothesis predicts leakage decreases monotonically with outline detail (more of the future specified = less need to "prepare"). Example outline format (numbered with a/b/c sub-beats) is directly reusable. Note that prompting alone may not enforce outline faithfulness; check outline adherence as a manipulation check.

---

## [SKIM] Maar, Paperno, McDougall & Nanda (2026). What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering (arXiv 2601.20164, ICLR 2026)

- **Question/method.** Define forward planning (representation of a goal token created at an earlier position) and backward planning (intermediate tokens conditioned on it). Mean-activation-difference steering vectors (multiplier 1.5) between prompt categories (rhyme families; intended noun answers), applied at a *single* token (last word of line 1, the `\n`, or `?`), choosing the best layer/position. Metrics: steered target-rhyme rate; *last-word regeneration* (remove context, regenerate final word of line 2 — above-chance recovery of the rhyme shows intermediate words were shaped by the plan); a/an article shift before the noun answer.
- **Data/models.** 1,050 lines over 10 rhyme families (Claude 3.5 Sonnet-generated; 85 train/20 test each); 500 questions for 20 nouns (vowel vs consonant initial pairs). 23 models: Gemma2, Gemma3, Qwen3, Llama-3.1/3.2, 1B-32B, base + instruct.
- **Results.** Steering at one token flips the rhyme/answer and changes intermediate tokens (backward planning) in all models from 1B; a/an shifts in the right direction across all 23 models; bigger and instruction-tuned models plan more; newline-position steering works mainly for Gemma2-9B/Gemma3-27B.
- **Code.** https://github.com/dpaperno/implicit-planning-supplementary-material
- **Use for our project.** (1) The *regeneration test* maps directly onto leakage: strip the secret-holding context and ask a fresh model (or the same model) to guess the secret from the story text alone — above-chance recovery means the prose was shaped by the secret (backward planning/leakage). (2) The a/an trick is a template for detecting secret-conditioned function words. (3) Mean-difference steering vectors between secret-A and secret-B contexts give a simple way to *add or subtract* the secret direction and test whether leakage follows (our Q2 causal test).

---

## [SKIM] Nainani, Vaidyanathan, Watts, Assis & Rigg (2025). Detecting and Characterizing Planning in Language Models (arXiv 2508.18098)

- **Question/method.** Causal criteria for planning with SAE latents: Future-Token Encoding (latent's logit-lens top-K contains a future token y_m) + Precursor Influence (negatively steering the latent at an earlier position changes the next token, changes intermediate tokens, and removes y_m). Pipeline: circuit discovery (>=60% logit recovery) -> FTE filter -> cluster-level steering -> earliest-moment search -> improvisation check. Labels: PLAN / IMPROV / NEITHER / CAN'T SAY (e.g., when y_m already appears in the prompt).
- **Models/data.** Gemma-2-2B base and instruct with GemmaScope TopK MLP_out SAEs; rhyming couplet from Lindsey et al.; first 60 MBPP tasks solved by the instruct model.
- **Results.** Gemma-2-2B solves the "grab it / habit" couplet by *improvisation* (no planning, unlike Claude 3.5 Haiku); on MBPP it switches between planning and improvisation even across successive tokens; instruction tuning refines rather than creates planning.
- **Code.** https://github.com/ambitious-mechinterp/plan_trace
- **Use for our project.** Important caveat: the "CAN'T SAY" category covers exactly our case — the secret token is *in the prompt*, so a future-token latent could be plain attention to context rather than planning. To separate "planning for a reveal" from "context echo", compare against a control where the secret is in context but irrelevant (e.g., stated as an unrelated fact) and use precursor-influence-style ablation at early story positions. Small models may improvise rather than plan, so test across scales.

---

## [SKIM] Men, Cao, Jin, Chen, Liu & Zhao (2024). Unlocking the Future: Exploring Look-Ahead Planning Mechanistic Interpretability in LLMs (arXiv 2406.16033)

- **Question/method.** Blocksworld (fully observed) fill-in-the-blank plans; information-flow analysis (MHSA vs MLP extraction rates at the last token; attention knockout from goal-state/history spans) and linear + 1-hidden-layer probes for current block states and *future decisions* per layer.
- **Models/data.** Llama-2-7b-chat-hf and Vicuna-7B, fully fine-tuned (3 epochs, lr 5e-5, bs 20) on synthetic optimal-plan data (4-6 colours, up to 6 steps); 61%/63% complete-plan success at level 3; 400 correct samples analysed. Tools: baukit, pyvene.
- **Results.** Mid-layer MHSA output at the last token partly decodes the decision; MHSA draws mainly from goal-state spans and recent steps; middle/upper layers encode a *few short-term* future decisions when planning succeeds, with accuracy falling as the horizon grows.
- **Code.** None of their own linked (uses baukit/pyvene).
- **Use for our project.** Supports two design choices: probe the secret at middle-to-upper layers, and expect future-content signals to be short-horizon. Their attention-knockout from "goal" spans is a direct analogue of blocking attention from story tokens to the secret-instruction span — a clean causal test of whether leakage needs continuous access to the secret (Q2).

---

## Cross-paper synthesis for our project (planning notes)

1. **Is the leakage planning or breadcrumbs?** Wu et al. say small models mostly leave "breadcrumbs" and pre-caching grows with scale; Nainani et al. show a 2B model often improvises; ParaScopes finds plans form "just in time" at paragraph boundaries. Prediction: outline effects on leakage should be larger in larger models.
2. **Plans reduce premature revelation.** Spoiler Alert (thick beat plan with explicit revealed/withheld fields: late no-rate 0.139->0.339 for Sonnet 4.6) and Xie & Riedl (plans improve suspense; requested clues increase it) are the closest behavioural evidence. Our outline condition should (a) be generated in a separate call that does not see the secret, (b) include per-beat "what is withheld", and (c) be compared with a generic-template plan control (Xie Ablation #2) and with outline granularity levels (Re3 3-point vs DOC hierarchical).
3. **Leakage metric.** Adapt 100-Endings into a per-sentence "N-guesses" curve (guesser and judge from a different model family than the writer); also use the What's-the-Plan regeneration test (guess the secret from the story without the instruction). Report mean, late-stage (>=80%) and peak/retention statistics.
4. **Mechanism probes.** Dong et al. (MLP probes on late layers; probe trajectories during generation are U-shaped), ParaScopes (training-free Patchscope at `\n\n`, layers at 60-80% depth), mean-difference steering vectors (Maar et al.), attention knockout to the secret span (Men et al.; Wu et al.'s myopic attention). For "does removing it remove leakage", combine (i) subtracting the secret-A vs secret-B mean-difference direction at story tokens, and (ii) blocking attention to the secret instruction after the first k tokens; measure the leakage curve.
5. **Controls.** Secret-in-context-but-irrelevant control (Nainani's CAN'T SAY caveat); no-secret baseline; exclude verbatim mentions; average over samples, not greedy.
