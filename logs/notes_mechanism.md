# Mechanism notes: secret leakage, concept directions, ablation/steering

Project: "LLMs are bad at keeping obvious secrets". Q1: does an explicit plan/outline reduce thematic leakage of a held secret word into a story? Q2: is the secret concept active in the residual stream throughout generation, and does removing it (directional ablation / steering / erasure) remove the leakage?

---

## [DEEP] Fairoze et al. 2026 - Inadvertent Context Leakage in Language Models (arXiv 2608.19857, FAIR/Meta, Berkeley, GDM)

**Research question.** A model correctly refuses to state an in-context secret. Do other properties of its benign outputs (numbers, length, formatting, style) still encode the secret well enough to reconstruct it?

**Methodology.**
- Predicate-inference game (Experiment 1). Challenger samples model M and context c. The adversary sends k prompts p_i from an allowed set P and gets r_i ~ M(p_i, c), then guesses phi(c).
- Optimal adversary (Alg. 1). Estimation phase: the adversary writes its own contexts with a known label b and estimates P(r | p, b). Inference phase: it predicts argmax_b prod_{(p,r) in T} P(r | p, b). Access is black-box (sampled text only), so the results are a lower bound on what a white-box adversary could get.
- Numeric setting: c = p_s || s, where s is uniform over N-digit strings (N in {1,2,4,8}). 18 benign task templates (e.g. "write a list of N-digit numbers", recipe, sports stats, open-ended story), plus 20 extra variants used only by the adaptive pipeline. 1,000 secrets x 10 responses per (secret, prompt), 10% held out.
- Six decoders: majority vote; frequency-Bayes over digit profiles; four Qwen-2.5-1.5B LoRA decoders (direct 10-class, left-to-right conditional, beam, adaptive per-position chain).
- Suppression metric: 1 - P(secret)/P_uniform, where P(secret) is the rate at which the model emits the protected value among the N-digit numbers it produces (P_uniform = 10^-N).

**Models.** Claude Opus 4.6, Claude Sonnet 4.6, Gemini 3.1 Pro, Gemini 3.1 Flash-Lite, GPT-5.4, GPT-5.4 nano, Grok 4, Grok 4.1 Fast. All at temperature 1.0. Training-stage ablation on OLMo-3-32B-Think (Base / SFT / DPO / RLVR).

**Datasets.** Synthetic digit secrets. CIMemories (Mireshghallah et al. 2026) personas: 15 memory predicates, decoded by a Llama-3.2-3B classifier. A production-style agent (OpenClaw-like workspace files) with a 9-digit SSN, at contexts of 0 to 256K tokens.

**Metrics.** Full-secret exact match (chance 10^-N). Per-digit accuracy (chance 0.1). Channel entropy: summed cross-entropy in bits, baseline 3.32N. Advantage = TPR - FPR. TPR at the baseline judge's FPR.

**Key results.**
- 0% leakage on direct requests (160 queries).
- N=2: 100% exact match on Opus 4.6 and Gemini 3.1 Pro.
- N=4: 82% exact match on Opus. The adaptive pipeline gives 82% vs 44% for static templates.
- N=8: 41% per-digit accuracy on Gemini Pro.
- GPT-5.4 nano and Grok 4.1 Fast stay at chance for N >= 4.
- Numeric, N-aware templates dominate leakage. Open-ended and multi-turn creative prompts leak near-zero information about digits.
- Suppression tracks leakage across models (Spearman rho = 0.95, p = 0.0011). Within the three leaky models, rho = 0.5 (not significant).
- Only the designated protected value is recoverable. Co-present numbers in context stay at chance.
- More emphatic confidentiality wording means more suppression and more leakage. Opus per-digit accuracy goes from 0.054 with a terse instruction to 0.643 with a verbose policy.
- Post-training increases leakage: 2.91 bits at Base vs 1.37 bits after RLVR.
- Memories: 0.319 advantage vs 0.058 for a DeepSeek-R1 linguistic judge.
- SSN via a GRPO-optimized injection (encode digits as the count of "!"): leading digit recovered in 97.1% (Opus) and 88.6% (Gemini) of trials.

**Code.** No repository is given in the paper.

**Limitations.** Only finite value sets (digits, binary predicates). Free-text and open-vocabulary secrets are explicitly left open. Black-box only. Decoders are trained per model. The memories evaluation uses a single CIMemories profile.

**Use for our project.**
- This is the strongest prior evidence that leakage is mostly *avoidance-shaped* (suppression), not just "mention-shaped". Our judge should test for both over-representation AND conspicuous avoidance of secret-related themes.
- It predicts that a stronger "don't reveal" instruction leaks MORE. Vary instruction emphasis as a factor.
- Our free-text/thematic secret setting is exactly the open problem they name. We can position our work as extending it to semantic secrets.
- Adopt their judge design: train or prompt a discriminator per model (an adaptive judge), not only a generic linguistic judge. Report advantage (TPR - FPR) and TPR at a fixed FPR, plus an N-way guessing accuracy against a chance of 1/|candidates|.
- Their numeric templates leak most and open-ended creative ones leak least *for digits*. For semantic secrets, stories are the natural channel, so do not expect the same ordering.

---

## [DEEP] Candussio, Padovani, Scalena, Nissim 2026 - "Don't Say It!": Constraints, Compliance and Communication when LMs Play Taboo (arXiv 2607.00601, CLiC-it 2026)

**Research question.** How do LLMs trade off lexical compliance and communicative informativeness in Taboo (describe a target without using forbidden words)? The paper compares interventions at three depths: prompt, decoding, and internal representation.

**Methodology.**
- 194 Italian Taboo cards (target word plus 5 forbidden words).
- Conditions:
  1. Baseline: no constraint.
  2. Prompting: forbidden words listed, morphological derivatives banned, max 30 words.
  3. A priori stem censorship: logits set to -inf for any token whose surface form is a prefix of a forbidden stem/lemma or vice versa. This banned on average 469 tokens per item, with severe collateral damage.
  4. Online constrained generation: detect complete words, compare stems, backtrack and mask at that position.
  5. SAE steering (Gemma only).
- SAE steering, exact formulation:
  - Run the prompt "La parola vietata e: {w_i}" through the model. Take residual-post activations at layer 29 of 34 (about 85% depth) and pass them through a Gemma Scope 2 JumpReLU SAE (google/gemma-scope-2-4b-it, width 65k, medium L0).
  - Select the feature with the maximum activation over w_i's token span.
  - d = (1/k) * sum_j W_dec[f_j] (the mean decoder vector over forbidden words).
  - At every decoding step: h' = h + alpha * ||h||_2 * d, "steering away". The value and sign of alpha are not reported.
  - In this condition the prompt does NOT mention the forbidden words.
- Evaluation:
  - Strict accuracy: no exact and no morphological violation, using the spaCy it_core_news_sm model and the NLTK Snowball stemmer.
  - Target-leak rate.
  - LLM-as-judge guesser: next-token probability of the target's first token after reading the description; pass@k for k in {1,2,3,5,10}, plus raw probability.
  - Human study: 8 annotators, guessing and describing.

**Models.** google/gemma-3-4b-it; openai/gpt-oss-20b with and without reasoning. Judges: both of these models. Claude Sonnet 4.6 and Gemini 3.1 Pro were used as additional clue-givers on 10 shared cards.

**Key results.**
- Strict accuracy:
  - Baseline: 34.0% (Gemma), 24.7% (gpt-oss without reasoning).
  - Prompting: 91.2% (Gemma), 97.4% (gpt-oss with reasoning), 72.2% (gpt-oss without reasoning).
  - Constrained decoding: 98.5% a priori and 96.4% online (Gemma). Its residual errors are mostly target-word leaks.
  - SAE steering: 55.7% (Gemma).
- Reasoning (planning in CoT) clearly improves compliance.
- Informativeness: pass@10 for online constrained decoding is 17.0% vs 11.3% for prompting (Gemma judge). Prompted models give "overly cautious, informationally sparse" descriptions.
- SAE steering keeps informativeness: 21.1% pass@10 despite low compliance. The authors read this as removing surface access while keeping conceptual associations.
- SAE steering caused code-switching in 44.8% of outputs (vs 6.35% for constrained decoding) and some referential nonsense.
- Human guessing accuracy: constrained 38.1% > SAE 31.0% > prompt 23.4%.

**Code.** https://github.com/DanielSc4/LMtaboo/

**Limitations.** Italian only. The shared set has only 10 words. SAE results are for one model and one layer, with alpha unreported. Judges are weak guessers (gpt-oss pass@10 is at most 10%).

**Use for our project.**
- Direct precedent that steering on a single SAE feature or direction suppresses surface tokens poorly and leaves the concept recoverable. Removing a word-level feature != removing the concept. We should expect partial effects and measure thematic leakage separately from lexical leakage.
- Their next-token-probability guesser is a cheap, continuous leakage judge: P(secret | story, "the hidden word is"). Rank of the secret among candidates = pass@k. Use a stronger guesser than 4B/20B models.
- Explicit reasoning/planning improves *lexical* compliance (97.4% vs 72.2%). This supports Q1's hypothesis but says nothing about thematic leakage. Our experiment fills that gap.
- Watch for coherence and language collapse under steering. Track perplexity and language ID as side metrics.
- Online backtracking constrained decoding is a good "lexical-only" control condition. It removes the word but should leave thematic leakage, which separates lexical leakage from conceptual leakage.

---

## [DEEP] Arditi et al. 2024 - Refusal in Language Models Is Mediated by a Single Direction (arXiv 2406.11717, NeurIPS 2024)

**Research question.** Is refusal mediated by a single residual-stream direction that is necessary (ablating it removes refusal) and sufficient (adding it induces refusal)?

**Methodology.**
- Exact formulas:
  - Difference in means, per layer l and post-instruction position i in I (chat-template tokens after the user instruction):
    - mu_i^(l) = mean over D_harmful_train of x_i^(l)(t)
    - nu_i^(l) = mean over D_harmless_train of x_i^(l)(t)
    - r_i^(l) = mu_i^(l) - nu_i^(l)
    - 128 train prompts per class, 32 validation prompts.
  - Candidates: |I| x L vectors.
  - Activation addition (induce): x^(l)' <- x^(l) + r^(l). Applied only at the source layer l*, at all token positions.
  - Directional ablation (remove): x' <- x - r_hat r_hat^T x. Applied at EVERY layer and every token position, to both x_i^(l) (pre-attention residual) and x~_i^(l) (post-attention, pre-MLP residual).
  - Weight orthogonalization (equivalent to ablation, permanent): W_out' <- W_out - r_hat r_hat^T W_out. Applied to the embedding, positional embedding, attention out and MLP out matrices, and output biases.
- Selection algorithm (App. C):
  - refusal_metric = log P(refusal tokens) - log P(other tokens) at the first response token. Example refusal tokens: Llama-3 {40 = "I"}, Gemma {235285}.
  - For each candidate compute:
    - bypass_score: mean refusal metric on harmful validation prompts under ablation.
    - induce_score: mean refusal metric on harmless validation prompts under addition.
    - kl_score: KL divergence at the last position on harmless validation prompts, with vs without ablation.
  - Choose the candidate with minimum bypass_score subject to induce_score > 0, kl_score < 0.1, and l < 0.8L (to avoid directions that are just unembedding/token directions).
  - Selected layers are about 40-75% depth, e.g. Llama-3 8B: pos -5, layer 12/32; Gemma 7B: layer 14/28; Qwen 7B: layer 17/32.
  - The search takes about 1 hour for 72B models.
- Evaluation: greedy decoding, 512 tokens. Refusal score by substring matching. Safety score by Llama Guard 2. HarmBench ASR. Coherence via MMLU / ARC / GSM8K / TruthfulQA and CE loss.

**Models.** 13 chat models: Qwen Chat 1.8B/7B/14B/72B, Yi Chat 6B/34B, Gemma IT 2B/7B, Llama-2 Chat 7B/13B/70B, Llama-3 Instruct 8B/70B.

**Datasets.** Harmful: AdvBench, MaliciousInstruct, TDC2023, HarmBench. Harmless: Alpaca. Evaluation: JailbreakBench (100 prompts), HarmBench (159).

**Key results.**
- Ablation drops refusal from about 0.9-1.0 to about 0-0.25 across models.
- Addition makes models refuse nearly all harmless prompts.
- ORTHO ASR is comparable to GCG.
- Capabilities stay within about 1 point, except TruthfulQA (-1 to -3.5).
- Directional ablation is more surgical (lower CE increase) than activation addition.
- Adversarial suffixes work by suppressing the direction's expression via attention hijacking.

**Code.** https://github.com/andyrdt/refusal_direction

**Limitations.** Heuristic extraction. The direction's semantics are unclear ("refusal" vs "harm"). Not tested on frontier models.

**Use for our project.**
- Per-secret concept direction recipe:
  - D+ = prompts where the model is told to keep the secret W (e.g. "Your secret word is OCEAN. Never reveal it. Write a story about X").
  - D- = identical prompts with a different secret, or no secret. Better: average over many control secrets so that "having a secret" cancels and only W-specific content remains.
  - Compute r_i^(l) at the post-instruction positions for all layers. Also try mean-pooling over story tokens generated under the secret.
- Selection analog:
  - Replace refusal_metric with a secret metric: log P(W-related tokens), or the judge's guess probability for W.
  - bypass = leakage under ablation (minimize). induce = leakage when added to the no-secret condition (must be > 0, i.e. adding the direction makes the story drift toward W).
  - Keep KL < 0.1 on control prompts and l < 0.8L. The l < 0.8L restriction is important for us: late layers will give a "do not say token W" direction rather than the concept.
- Ablation during generation with HF hooks: register forward_pre_hooks on every decoder layer (input residual) and on the post-attention residual (or hook the attention and MLP outputs and subtract their projection onto r_hat), applying x -= (x @ r_hat)[..., None] * r_hat at all positions, including generated tokens. Equivalently orthogonalize o_proj, down_proj and embed_tokens weights once (no hooks, works with vLLM).
- Necessity/sufficiency framing for Q2: (a) ablating r_W removes thematic leakage while the story stays coherent; (b) adding r_W to a no-secret prompt induces W-themed stories. Control with random directions of matched norm and with the direction of a different secret.
- Monitor the cosine of the per-token residual with r_W over the generated story (their Fig. 5 style) to test "is the secret active throughout generation", and whether a plan/outline reduces that projection.

---

## [DEEP] Panickssery (Rimsky) et al. 2023/2024 - Steering Llama 2 via Contrastive Activation Addition (arXiv 2312.06681, ACL 2024)

**Research question.** Can mean-difference steering vectors built from many contrast pairs reliably steer alignment-relevant behaviors in RLHF chat models, in both multiple-choice and open-ended generation? How does this compare to and combine with system prompts and finetuning?

**Methodology.**
- Exact formula: v_MD = (1/|D|) * sum over (p, c_p, c_n) in D of [a_L(p, c_p) - a_L(p, c_n)]. The activations are residual-stream values at layer L at the answer-letter token position. Contrast pairs are A/B multiple-choice questions that differ only in the final answer letter, which isolates the behavior and cancels confounds.
- Application: add multiplier * v at layer L to every token position AFTER the user prompt, including all generated tokens.
- Vectors are norm-standardized across behaviors but not across layers ("natural norm").
- Layer selection: sweep all layers with multipliers +/-1 on 50 held-out MC questions, pick the peak, then sweep multipliers at that layer.
- PCA of contrast activations shows behavioral clustering emerging suddenly (refusal at layer 10 of Llama 2 7B), roughly one-third of the way through the network.

**Models.** Llama 2 7B Chat and 13B Chat; Llama 2 7B base (for transfer).

**Datasets.** Anthropic Advanced AI Risk (Perez et al. 2022): AI coordination, corrigibility, myopic reward, survival instinct. Sycophancy on NLP / political typology. GPT-4-generated hallucination and refusal sets. MMLU and TruthfulQA for capability checks.

**Metrics.** MC: probability of the behavior-matching answer. Open-ended: GPT-4 1-10 behavior rating.

**Key results.**
- Optimal layer is 13 for 7B and 14-15 for 13B. Effects vanish after about layer 17-20.
- Open-ended multipliers range from -1.5 to 1.5. Larger multipliers degrade text quality.
- CAA adds effect on top of positive/negative system prompts. Example from Table 3 (13B, layer 13): corrigibility goes from 0.79 (positive prompt) to 0.93 (with +1 steering).
- CAA also adds on top of finetuning for 3 of 7 behaviors.
- MMLU is nearly unchanged (0.63 vs 0.57-0.65).
- Steering-vector cosine with per-token activations highlights semantically relevant tokens, which works as a token-level detector.
- The layer-13 vector transfers to nearby layers. Base-to-chat transfer works at layers 10-15.

**Code.** https://github.com/nrimsky/CAA (MIT)

**Limitations.** GPT-4 evaluation noise. Prompting and finetuning baselines not optimized. Single multiplier per layer during the sweep. Only Llama 2.

**Use for our project.**
- Contrast-pair design for a secret concept: the same story prompt with "secret word = W" vs "secret word = W'" (many W' values). Read activations at the last prompt token, or average over the first ~20 generated tokens. Use 100+ pairs with varied story topics so that topic cancels.
- Steering is applied at all positions after the prompt, so a single hook on one layer (~35-45% depth, e.g. layer 13/32) covers generation. Use negative multipliers (about -0.5 to -1.5 x the natural norm) to test whether subtracting r_W reduces thematic leakage. Cap the magnitude by coherence (perplexity or an LLM-judge fluency check).
- A secondary metric for Q2: the cosine/dot product of each generated token's residual with v_W (their Fig. 6) gives a per-token "secret activity" trace across the story. Compare plan vs no-plan conditions.
- Their system-prompt x CAA grid is a template for our plan x ablation 2x2 (plan/no-plan crossed with ablate/no-ablate).

---

## [SKIM] Turner et al. 2023 - Steering Language Models with Activation Engineering / ActAdd (arXiv 2308.10248)

- **Method.** h_A^l = h^l(p+) - h^l(p-) from a single prompt pair, right-padded to equal length. Steered forward pass: h^l <- h^l + c * h_A^l, added at the front sequence positions of the user prompt at layer l. Typically |c| < 15. Optimization-free.
- **Models.** GPT-2-XL; OPT-6.7B; LLaMA-3-8B (also GPT-J).
- **Key results.**
  - Topic steering: "weddings" vs " " at layer 16 with c=1 lowers perplexity on wedding text without hurting other text.
  - The layer sweep on GPT-2-XL peaks at about layer 6, with more than 90% of completions mentioning weddings (2% baseline).
  - GPT-3.5-judged topic relevance rises 5-20% across topics at c=2.
  - State of the art on detoxification (RealToxicityPrompts) and sentiment shift.
- **Code.** zenodo.org/records/13879423; notebook at tinyurl.com/actadd.
- **Use.** The closest analog to our setting: a TOPIC concept injected via the residual stream makes stories drift toward that topic. This is the "induce" side of our sufficiency test. Subtracting "W" - " " is the cheapest per-secret baseline direction. The wedding-word-count metric (a lexicon of related words) is a simple lexical leakage metric we can copy, alongside a judge.

## [SKIM] Belrose et al. 2023 - LEACE: Perfect Linear Concept Erasure in Closed Form (arXiv 2306.03819, NeurIPS 2023)

- **Method.**
  - Linear guardedness holds iff all class-conditional means are equal (equivalently, Cov(X, Z) = 0).
  - LEACE: r(x) = x - W^+ P_{W Sigma_XZ} W (x - E[X]), where W = (Sigma_XX^{1/2})^+ is the whitening matrix and P is the orthogonal projector onto colsp(W Sigma_XZ). This is the least-squares-optimal affine eraser.
  - Concept scrubbing (Alg. 1): fit LEACE sequentially layer by layer on already-scrubbed activations, because erasing at layer l shifts the distributions at later layers.
- **Models.** LLaMA 7B/13B/30B; Pythia 160M-12B; BERT for gender bias.
- **Key results.** Scrubbing part-of-speech raises LLaMA-7B perplexity from 0.69 to 1.73 bits/byte (SAL: 3.24; random erasure: no change).
- **Code.** https://github.com/EleutherAI/concept-erasure (pip install concept-erasure; LeaceEraser / LeaceFitter / scrubbing helpers).
- **Use.** For binary "secret W present vs absent", rank(Sigma_XZ) = 1, so LEACE removes one (whitened, oblique) direction. It is the principled upgrade from diff-in-means ablation, and with k candidate secrets as one-hot Z it erases a (k-1)-dim subspace. Fit sequentially across layers (scrubbing) to avoid re-emergence of the concept downstream. Guardedness is only linear, so check with a nonlinear probe and with behavioral leakage.

## [SKIM] Karvonen et al. 2025 - Activation Oracles (arXiv 2512.15674)

- **Method.**
  - LatentQA: an LLM (LoRA) is trained to answer natural-language questions about injected activations.
  - Injection: after layer 1, at placeholder tokens " ?": h_i' = h_i + ||h_i|| * v_i/||v_i|| (norm-matched).
  - Training data: system-prompt QA, 7 classification tasks, and self-supervised previous/next-token prediction. About 1M examples. Trained on activations from 25/50/75% depth; evaluated at 50% depth.
- **Models.** Qwen3-8B, Gemma-2-9B-IT, Llama-3.3-70B-Instruct, Claude Haiku 3.5.
- **Key results.**
  - Recovers Taboo secret words from fine-tuned Gemma-2-9B secret-keepers (Cywinski et al.): 21% vs 4% for the best interpretability baseline.
  - Secret Side Constraint: 59% vs 52%.
  - Matches or beats white-box baselines on all 4 audit tasks.
- **Code.** github.com/adamkarvonen/activation_oracles (trained models released).
- **Use.** An off-the-shelf readout to test "is the secret concept present in activations at token t of the story?" Ask "What is the secret word?" on per-position activations across the story, plan vs no-plan. Released Qwen3-8B / Gemma-2-9B oracles avoid training our own. Check that ablation lowers AO recovery (an internal-erasure check that complements behavioral leakage).

## [SKIM] Bersia and Gaintseva 2026 - When Activation Oracles Learn Not to Read: Concept-Specific Blind Spots (arXiv 2607.23379)

- **Method.**
  - Qwen3-8B Taboo subjects (hidden words such as leaf, moon), plus AOs fine-tuned on those subjects' activations.
  - Diagnostics: linear probes on residual activations at layers {9, 18, 27} across regimes HINT / REFUSAL / SAMETEXT / THINK / OFFTOPIC; LogitLens on mean activations at layer L18; layer ablations inside the AO.
- **Key results.**
  - Behavioral secrecy does not imply activation secrecy. Probes classify the hidden concept with more than 98% accuracy in every regime, including OFFTOPIC (the concept leaves a persistent activation trace even on unrelated tasks).
  - Fine-tuned AOs become "anti-readers" of their own training concept. The concept stays decodable inside the AO, so the failure is in the readout pathway.
- **Code.** None found in the text.
- **Use.**
  - Strong support for Q2's premise: the secret is persistently represented even off-topic.
  - Use simple linear probes (not only AOs) as the primary "secret is active" measure, and do not train AOs on our own secret-holding prompts.
  - Behavioral leakage, probe decodability and verbalizability can dissociate. Report all three.

## [SKIM] Wang, Xiong, Shu 2026 - Do LLMs Know What Is Private Internally? Probing and Steering Contextual Privacy Norms (arXiv 2604.00209, COLM 2026)

- **Method.**
  - Last-token hidden states on contextual-integrity stimuli. Paired differences Delta h = h+ - h-, mean-centered; the first PC is the "privacy direction". Per-layer logistic probes (5-fold CV).
  - CI-parametric steering: h_l' = h_l + alpha * sum over p in {info type, recipient, transmission principle} of v_l^(p) (unit-normalized), applied at the top-k layers by direction magnitude.
- **Models.** Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Llama-2-7B-Chat.
- **Datasets.** Synthetic CI stimuli; ConfAIde Tier 2/3; PrivaCI-Bench. GPT-4o-mini judge labels outputs as leaked / refused / appropriate.
- **Key results.**
  - Norms are linearly decodable (AUROC above 0.90 in upper layers) yet models leak: 42.5% of scenarios, and 24.1% on ConfAIde Tier 3.
  - Monolithic single-direction steering can backfire (Llama-2: 52% to 88% leakage; ConfAIde leakage doubles at alpha = 0.5).
  - CI-parametric steering cuts leakage from 42.5% to 5%.
- **Code.** https://github.com/wang2226/CI-Steering
- **Use.** Representation-behavior gap: knowing the concept internally does not equal controlling it. Single-direction interventions can backfire, so sweep alpha carefully and report non-monotonicity. Their leaked/refused/appropriate judge rubric is adaptable.

## [SKIM] Minder et al. 2025 - Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences (arXiv 2510.13900, ICLR 2026)

- **Method.** Activation Difference Lens (ADL).
  - delta_j = mean over 10k pretraining samples of (h_ft - h_base) at the middle layer floor(L/2), for the first k=5 token positions.
  - Interpret delta_j with Logit Lens and a calibrated Patchscope (lambda * delta_j inserted into an identity prompt).
  - Steer the finetuned model by adding alpha * delta_j at all positions. alpha is the largest value that stays coherent (binary search with a gpt-5-nano grader).
  - Evaluate: embedding similarity (Qwen3-Embedding-0.6B) of steered text to the finetuning data; an interpretability agent identifies the finetuning objective.
- **Models.** 33 organisms over 7 models (Gemma 2 9B, Gemma 3 1B, Llama 3.1 8B, Llama 3.2 1B, Qwen2.5 7B, Qwen3 1.7B and 32B). Includes 6 Taboo models from Cywinski et al.
- **Key results.**
  - Narrow finetuning leaves a strong constant bias, readable even on unrelated text.
  - The agent with ADL is more than 2x better at identifying the broad objective and more than 30x better on details.
  - Mixing in pretraining data mostly removes the bias.
- **Code.** https://github.com/science-of-finetuning/diffing-toolkit
- **Use.**
  - Taboo fine-tuned model organisms are artifact-prone: they carry a constant secret bias that is readable everywhere. Our in-context secret setup avoids this confound but should be compared against it.
  - The coherence-bounded binary search for steering strength and embedding similarity to secret-themed text are reusable as an automatic thematic-leakage metric. Use cosine of story embedding vs embedding of "W" or W-themed reference texts, relative to control secrets.

---

## Cross-paper synthesis for the experimental design
1. **Direction.** Diff-in-means over many contrast prompts ("secret = W" vs a pool of "secret = W'"), at post-instruction positions. Sweep layers over 25-80% depth. Select with Arditi-style criteria (minimize leakage under ablation; adding the direction to a no-secret prompt induces W-drift; KL < 0.1; l < 0.8L). Upgrade option: LEACE with sequential scrubbing.
2. **Ablation.** x <- x - r_hat r_hat^T x at all layers and all positions during generation (HF forward_pre_hooks or weight orthogonalization). Compare against CAA subtraction at one layer (multiplier -0.5 to -1.5) and a random-direction control.
3. **"Secret active throughout generation."** Per-token projection onto r_W (CAA Fig. 6 / Arditi Fig. 5 style), linear probes at about 25/50/75% depth, and optionally released Activation Oracles. Compare plan vs no-plan.
4. **Leakage judge.**
   - (a) N-way guess: give the story plus K candidate words and score P(W) / rank (pass@k, Taboo-style).
   - (b) Paired discrimination: story-with-secret vs story-without, judged as advantage = TPR - FPR (Fairoze).
   - (c) Lexicon/embedding similarity to W (ActAdd wedding-word counts; ADL embedding similarity).
   - Measure avoidance too (suppression): W-related themes appearing LESS than in the no-secret baseline is also leakage.
5. **Expected pitfalls.**
   - Single-feature steering lowers lexical mentions but keeps the concept (Taboo SAE: 55.7% compliance).
   - Steering may backfire non-monotonically (CI steering).
   - Coherence and language collapse at high strength.
   - Stronger confidentiality instructions increase suppression-type leakage (Fairoze).
