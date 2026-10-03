# Deep-read notes: secret keeping, leakage, and internal representations

---

## 1. Cywinski, Ryd, Rajamanoharan, Nanda (2025). "Towards eliciting latent knowledge from LLMs with mechanistic interpretability" (arXiv 2505.14352)

**Research question.** Can black-box and white-box (logit lens, SAE) methods recover a secret word that a fine-tuned "Taboo" model describes but never says?

**Methodology.**
- Taboo model organism: the model plays Game Master, gives hints about a secret word, never says it, and never confirms correct guesses. The secret appears in neither the training data nor the prompt. It is learned implicitly from hint-only conversations.
- Data: Gemini 2.5 Flash generated multi-turn games (3-5 turns). Hint style was randomized over 11 categories (etymology, sensory, functional, cultural, natural world, metaphorical, linguistic pattern, riddle, contrasting, contextual, classification). Conversations containing the word or its variants were filtered out. Exact generation prompts are in App. A.
- Training: QLoRA rank 8, 10 epochs, batch 8, LR 2e-4, paged_adamw_8bit, linear schedule, warmup 0.03, wd 0.01, grad-norm 0.3, bf16, early stopping (patience 2), TRL, 1x A100.
- Elicitation: (a) black-box: naive prompting, adversarial prompting (4 categories x 10 prompts written by Gemini 2.5 Pro), token forcing pregame/postgame (10 prefill phrases, e.g. "My secret word is"), and another model (a separate LLM guesses from the hints); (b) white-box: **logit lens at layer 32**, averaging each token's probability over all response positions and excluding tokens actually generated; **SAE**: Gemma Scope 16k-latent residual SAE at layer 32, ranking latents by mean activation over the response.

**Models.** Base `google/gemma-2-9b-it` (42 layers). Guesser: Gemma 3 12B-it (main text); App. D.6 says Gemini 2.5 Flash. 20 Taboo checkpoints.

**Word list.** 20 single-token nouns sampled from Reikes "Top 1000 English nouns" (gist.github.com/creikey/42d23d1eec6d764e8a1d9fe7e56915c6): chair, cloud, dance, flag, green, jump, blue, book, salt, wave, clock, flame, gold, leaf, moon, rock, smile, snow, song, ship. Gemma Scope L32 16k latents (Table 7): chair 7713, cloud 14741, dance 14269/3115, flag 4926, green 1206, jump 13979, blue 13079, book 5895, salt 11388, wave 12010, clock 15717, flame 9266, gold 846, leaf 9825, moon 13740, rock 15112, smile 9936, snow 11942, song 15324, ship 5404.

**Metrics.** Pass@10 (correct in at least one of 10 prompts), Majority@10, and Accuracy. Greedy decoding. 10 standard hint prompts ("Give me a hint!", ...).

**Key results (Table 1).**

| Method | Pass@10 | Accuracy |
|---|---|---|
| Naive prompting | 0% | 0% |
| Adversarial prompting | 17.5% | 2.4% |
| Token forcing pregame | 15% | 2% |
| Token forcing postgame | 70% | 57.5% |
| Another model (guess from hints) | **95%** | 54.5% |
| Logit lens top-1 | 35% | 16.5% |
| Logit lens top-5 | 75% | 35% |
| SAE top-1 | 35% | 12% |
| SAE top-5 | 55% | 35% |

- Logit lens shows a peak for the secret around **layers 30-37**. The peaks fall on predictable function tokens (', a, the) at many positions throughout the hint, so the secret stays persistently active.
- The SAE secret latent fires across many tokens of the response.

**Code/models.** github.com/EmilRyd/eliciting-secrets; huggingface.co/bcywinski (Taboo checkpoints, e.g. `bcywinski/gemma-2-9b-it-taboo-<word>`; check the exact naming on HF).

**Limitations.** Single-token concrete nouns only. The secret is fine-tuned in, not given in context. Only 20 models and 10 prompts. Hint-giving makes leakage the task objective itself, not an incidental side effect.

**Use for our project.**
- "Another model" guessing reached 95% Pass@10: a strong external judge recovers a secret from text that never names it. Use an LLM guesser (rank of the true word among guesses) as our thematic-leakage metric for stories.
- Mechanism probe: Gemma-2-9b-it + Gemma Scope residual SAE (16k, layer ~31-32) + logit lens at layers 30-37. Average the secret-token probability or SAE latent over every story token. Reuse the Table 7 latent IDs for these 20 words.
- Our in-context setting (system prompt says "secret is X, don't reveal it") can be compared directly with the fine-tuned Taboo checkpoints.
- Intervention: ablate the Table 7 SAE latent (or the difference-of-means concept direction) at L32 during story generation and measure the drop in guesser accuracy.
- Reuse the 20-word list as a ready-made, single-token, concrete secret set.

---

## 2. Cywinski, Ryd, Wang, Rajamanoharan, Nanda, Conmy, Marks (2025). "Eliciting Secret Knowledge from Language Models" (arXiv 2510.01070v2)

**Research question.** A benchmark of black-box and white-box techniques for extracting knowledge a model uses downstream but denies having. Three model-organism families.

**Methodology.**
- **Taboo**: secret word (gold, moon, flag). 300 hint conversations plus 150 adversarial refusal samples. **User Gender**: male/female; 500 samples plus 225 refusal/pronoun samples. **SSC**: a Base64-encoded side constraint in the prompt, e.g. "use passive voice"; 43k samples plus 5k refusals; categories are language, structure, and syntax modifiers.
- Key fix in v2: each dataset is mixed with **10x unrelated Alpaca chat**. Without it, narrow fine-tuning leaves activation biases (Minder et al. 2025) that make white-box methods unrealistically strong; v1 Taboo results were inflated.
- Internalization check: an "any word" hint task, with GPT-4 rating 0-100 how clearly the hint points to the target. In-context baseline: "Your word is X. Now give me one hint ... never reveal the word directly" given to base Gemma (App. D.1).
- Auditing game: a smaller LLM auditor (Gemma 3 4B for Taboo and Gender; Llama 3.1 8B for SSC, since it cannot decode Base64) guesses the secret from I/O plus tool output.
- White-box: logit lens; **activation-token embedding similarity** (cosine of the residual with the input embedding matrix, skipping final norm and unembed); **SAE with TF-IDF scoring**, score = mean activation x log(1/density); fuzzing (Gaussian noise added to the residual). Black-box: adversarial prompts, assistant prefill, user-persona sampling, many-shot jailbreak (~8k tokens), high temperature.
- Layers and positions: **Taboo L32** (Gemma-2-9B); **Gender L23**; **SSC L50** (Llama 3.3 70B, the Goodfire SAE layer). Taboo readouts use the assistant control tokens (`<start_of_turn>`, `model`). Gemma Scope SAEs, with descriptions and densities from Neuronpedia. Taboo "gold" features at L32 (wide SAE): 90419, 18398, 73618, 82818. Gender L23: 60183, 124907, 44304.

**Models.** Gemma 2 9B-it (Taboo, Gender): LoRA r16, alpha 32, 3 or 2 epochs, bs 32, LR 2e-4. Llama 3.3 70B-Instruct (SSC): r8, 4-bit, Unsloth, LR 5e-5.

**Metrics.** Auditor accuracy (single guess). SSC uses best-of-5 GPT-4.1 semantic similarity. Standard set: 100 prompts x 10 samples at T=1. Direct set: adversarial prompts.

**Key results (App. B tables).**
- Standard set, Taboo: I/O baseline **42.1%**; +logit-lens tokens 84.0; +activation tokens 85.1; +SAE tokens 84.7; +SAE descriptions 62.4.
- Base-model control: the same readouts *hurt* (Taboo LL 19.4%). The signal is specific to the secret.
- Direct set: prefill reaches 95.9% (SSC) and 95.5% (Gender); user persona 96.8% and 97.7%. On Taboo, everything stays ≤7%, because fine-tuning suppresses the English token. Prefill plus "say it in {language}" gives ~40-45% success.
- Token readouts beat SAE descriptions. Logit-lens top-100 for "gold" includes translations (dorada, złoty, 金, ouro).
- On refusals (direct prompts), the secret is often not internally active. White-box signal appears only when the model needs the secret.
- Fuzzing and high temperature are largely ineffective; MSJ fails.
- App. J: the secret is represented most strongly at the assistant control token and peaks again on content tokens ("this").

**Code/models.** github.com/cywinski/eliciting-secret-knowledge; HF collection huggingface.co/collections/bcywinski/eliciting-secret-knowledge-from-language-models (Taboo gold/moon/flag, gender male/female, SSC checkpoints).

**Limitations.** Secrets come from a single SFT run. Single-turn auditing. Small auditors. Truthfulness steering and ablation were not tested. The Taboo hints make the secret guessable by design.

**Use for our project.**
- Leakage metric: reuse the Taboo auditor prompt (App. G.1) with a small LLM guesser (Gemma 3 4B, or something stronger) on our stories. Report guess accuracy against an I/O-only baseline, plus a base-model, no-secret control.
- "Does the concept stay active?": activation-embedding similarity and logit lens at L32 of Gemma-2-9B-it. Look at the assistant control token and at every story token, and compare against a no-secret control.
- Their finding that the secret is active only when needed predicts that a plan which doesn't need the secret should lower internal activation, and so lower leakage. This is directly testable.
- In-context baseline prompt (App. D.1) works as our "secret in prompt" condition.
- Use TF-IDF SAE scoring to drop generic latents. Gemma Scope wide (131k) L32 features exist for gold.
- Mix unrelated data if we fine-tune anything.
- Ablation hypothesis: the token-suppression finding (English token suppressed, concept intact) warns that ablating the token direction alone may leave the concept, and so thematic leakage, intact. Ablate a concept direction (difference of means over many context sentences), not the unembedding row.

---

## 3. Mann, Saxena, Tandon, Sun, Toteja, Zhu (2025). "Don't Think of the White Bear: Ironic Negation in Transformer Models Under Cognitive Load" (arXiv 2511.12381; NeurIPS 2025 workshop-style short paper)

**Research question.** Does "do not mention X" raise the probability of X (ironic rebound)? How does intervening distractor text (load) change this? Do models that separate neutral from negative framings show more persistent rebound?

**Methodology.**
- Template: `[Topic]. Do not mention [Target]. [Distractor text] ...`, with a matched neutral condition that omits the negation. Distractor types: semantic (on-topic text without X), syntactic (grammatical, semantically light), and repetition. Length 0-1024 tokens in powers of 2, converted to within-model percentile load ℓ ∈ [0,1].
- Measured by **log-probability of the single-token target at the evaluation point**, not free generation.
- Metrics: surprisal difference Δs(ℓ) = log2 p(X|ℓ) − mean log2 p(X | ℓ≥0.95), where positive means rebound; suppression score S(L) = E[log p(X|neutral) − log p(X|suppress)], where S<0 means rebound; peak Δ; L50 (load at which rebound halves); AUCΔ; polarity discrimination Δ = log p(r_neu) − log p(r_neg). Mixed-effects regression with a random intercept per concept.
- Mechanism: head ablation over all ~1,024 heads of Llama-3-8B-Instruct, measuring the effect on the forbidden-token logit.

**Models.** GPT-OSS-20B, GPT-NeoX-20B, Qwen3-14B, Llama-3-8B-Instruct, OPT-2.7B, Bloom-560M, Pythia-410M, LFM2-350M (`LiquidAI/LFM2-350M`), GPT-2 Small. Accessed via HF Transformers; exact repo IDs are not listed except LFM2.

**Dataset.** **ReboundBench**: ~5,000 templated prompts built from "This-is-not-a-dataset" (HiTZ) and NUBench (SNU Thunder). Single-token common nouns, lowercased, punctuation stripped. Fields: topic, target, distractor type, length, prompt. Examples: "Write about animals on a farm. Do not mention sheep. Cows and pigs are often raised for food." HF: `SavarToteja/dont-think-of-the-white-bear`.

**Key results.**
- Rebound appears right after the negation and decays with load. Semantic distractors give the strongest and longest rebound; repetition the weakest. Peak Δ is around 7-28 bits (e.g. Llama-3-8B-Instruct semantic 13.2 bits, L50 0.15, AUC 2.83; GPT-NeoX-20B semantic 24.6).
- **GPT-OSS-20B is an outlier with almost no rebound** (peak 1.6 bits; repetition −2.5). The authors speculate this is due to synthetic-data or safety post-training.
- Polarity discrimination is not monotonic in scale (Llama-3-8B-I 11.3 bits, Bloom 10.1, NeoX 5.7, Qwen3-14B 6.5). It correlates with rebound persistence (r ≈ 0.44).
- Heads: early layers (0-7) suppress, and middle layers (8-16) contain both types. Strongest suppressor heads sit at L8-13 and amplifier heads at L12-17. 15-20 heads carry ≥80% of the effect. Overall 26.7% of heads amplify, 29.7% suppress, and 43.7% are neutral. Single-head effects reach up to ±4 logits.

**Code/data.** github.com/cesium132dot9/Dont-Think-of-the-White-Bear; HF dataset `SavarToteja/dont-think-of-the-white-bear`.

**Limitations.** Log-probability only, no generation. Synthetic templates. Single-token targets. The baseline definition is odd: "high load" is treated as the no-negation reference, and huge bit values hint at a scale or normalization issue. Short paper with thin statistics and no per-head IDs reported. Treat as suggestive.

**Use for our project.**
- Supports the core hypothesis: mentioning a forbidden concept primes it. Our "secret word + don't reveal" prompt should raise p(secret) and of semantically related tokens. Measure this directly as the log-prob of the secret token, and of related-word sets, at each story position, compared with a no-secret control prompt (their S(L) metric).
- Their load manipulation maps onto our plan/outline: the plan is intervening "semantic" text between the instruction and generation. The prediction is that a plan unrelated to the secret dilutes rebound, while a plan in the secret's neighborhood amplifies it. Test plan-relatedness as a factor.
- Mechanistic follow-up on Llama-3-8B-Instruct: head ablation around L12-17 (amplifiers) might reduce leakage, offering an alternative to residual-direction ablation.
- Include GPT-OSS-20B as a contrasting model, since it showed little rebound.
- Positive framing ("focus on Y instead of X") is their suggested mitigation, closely related to giving an explicit plan.

---

## 4. Baldelli, Parviz, Zouaq, Chandar (2026). "LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents" (arXiv 2601.06973v3; CoLLAs 2026)

**Research question.** Can chat agents that see only the public transcript maintain a self-generated hidden secret (Private State Interactive Tasks, PSITs)? Does an explicit private working memory fix this?

**Methodology.**
- Theory: an impossibility theorem. A public-only chat agent (POCA) cannot keep the secret underdetermined by the transcript while also answering consistently with one fixed secret, because its output distribution is identical across secrets compatible with the history.
- Self-consistency test (SCT): a rule-based Player plays 4 turns (t_fork = 4; robust over 2-8). The dialogue is then forked: one branch asks "reveal the secret", and 5 branches ask "Is the secret X? yes/no" for candidates consistent with public constraints (Wordfreq for Hangman; the DDXPlus dataset for the medical "Diagnosis Simulator").
- Outcomes: Self-consistent; **Leakage** (secret appears verbatim in the transcript before the reveal, by case-insensitive match); Over-confirmation; State substitution; All-denial.
- Agents: Vanilla; Private CoT (reasoning traces kept across turns; upper bound); retrieval memory baselines Mem0, A-Mem, LightMem, MemoryOS; and the proposed private working memory, a text block in `<private_state>` with sections Goals/Plans, Facts, and Active Notes. Two variants: an Autonomous agent (tool calls) and a Workflow (forced memory update every turn). Update tools: overwrite, append/delete, patch/replace. LangGraph, T=0.3, 2048 tokens/turn, 50 episodes per cell.

**Models.** GPT-OSS-20B, GPT-OSS-120B, Qwen3-32B, Qwen3-235B (open weights, reasoning plus tool use). Qualitative demos on ChatGPT (GPT-5.1), Gemini 3, and Claude Sonnet 4.5.

**Key results (Table 1, self-consistency %).**
- Vanilla: Hangman 2/12/4/12; Diagnosis 2/26/12/14.
- Retrieval memories: about the same (0-14% Hangman).
- Private CoT: 94/82/94/98 on Hangman.
- **Workflow + Overwrite: 76/98/92/100** on Hangman; 70/96/18/56 on Diagnosis.
- Workflow beats Autonomous, and performance scales with size.
- Failure modes are model-specific. GPT-OSS-20B tends to deny everything. GPT-OSS-120B over-confirms ("plays along"). **Qwen3-32B leaks heavily in Diagnosis**, attributed to helpfulness/transparency alignment overriding the rule.
- Mode collapse of chosen secrets: GPT-OSS-120B picks "planet" 286/500, GPT-OSS-20B "python", Qwen3-32B "elephant", Qwen3-235B "apple"; Sarcoidosis dominates in Diagnosis.
- Commercial demo: Gemini 3's reasoning chose "GALAXY" in turn 1, then reasoned with "JUNGLE" in turn 2. All three models "pretend" to have a word ("deceptive compliance").
- Private memory stays under 100 tokens; Private CoT uses about 10x more.

**Code.** github.com/chandar-lab/Hangman.

**Limitations.** The secret is self-generated rather than given, a different regime from ours where the secret is in context. Leakage is measured only verbatim, not thematically. Only 2 tasks, and candidates sometimes come from an LLM helper.

**Use for our project.**
- Conceptual contrast: in our setting the secret *is* in the context (system prompt), so the hidden-state problem flips. The model always "sees" the secret, which plausibly drives leakage, while Hangman models without the secret cannot stay consistent.
- Their structured private memory (Goals/Plans | Facts | Active Notes) is a ready template for our "explicit plan" condition. Put the story outline in a private plan block that excludes the secret, compared with a plan that mentions the secret.
- Reuse the forking idea: fork after the story and ask the model (or a judge) "Is the secret X?" across candidate words to test consistency and leakage.
- Mode collapse: models gravitate to attractor words (planet, apple, elephant). Our secret word list should control for prior frequency and "attractor-ness", since leakage of a high-prior word into a story could be base-rate rather than secret-driven. Use a no-secret baseline to estimate base rates per word.
- Open models with reasoning traces: GPT-OSS-20B and Qwen3-32B are candidates for checking whether CoT/plan text mentions the secret.

---

## 5. Luo, Xu, Lu, Yuan, Yao (2026). "Probing the Lack of Stable Internal Beliefs in LLMs" (arXiv 2603.25187; NeurIPS 2025 PersonaLLM workshop)

**Research question.** Do LLMs maintain "implicit consistency", i.e. a stable unstated target, across a multi-turn 20-questions game? This is distinct from "external" consistency (no contradictory answers).

**Methodology.**
- Proposer and Guesser LLMs play a game. **Number guessing**: 10 numbers sampled from 0-99. **Entity guessing**: 10 entities across categories (e.g. Eiffel Tower, Panda, Bitcoin, Vitamin C, Mona Lisa), with a provided attribute mapping. The Proposer secretly picks one.
- **Branch probe**: after each turn, a side branch with the full history asks a "SUDO USER" phrase, "What is the specific target's index you selected?". Candidates map to single-token indices 0-9, so the belief state b_i is read from the top-20 logprobs over index tokens (missing → −9999, then renormalized). The main dialogue is never contaminated.
- Metrics: Drift Rate (#changes/T); Once Drift Rate (fraction of dialogues with any change); KL(P_t || P_{t-1}); External Consistency Verification (LLM judge = DeepSeek-v3.1-Reasoning); Branch Drift Rate (final ≠ initial).
- Proposer at T=0; Guesser is a reasoning model. Dialogues expand as a 3-ary tree.
- SFT: Qwen-2.5-14B-Instruct on 1,009 dialogues; loss = α·KL(P_probe^t || P_initial) + β·CE; 1 epoch, LR 1e-5, effective batch 128, 8xA800.

**Models.** GPT-4o, Seed-1.6 (± reasoning), DeepSeek-v3.1 (± reasoning), Claude-3.7-Sonnet (± reasoning) via API. Qwen-2.5-14B-Instruct for SFT (`Qwen/Qwen2.5-14B-Instruct`).

**Key results (Table 1).**
- Drift is universal: Once-Drift ≈ 95-100% for every model. Per-turn drift rate is 17-100% for numbers (GPT-4o 43.4; Seed 17.4; DeepSeek 38.5; Claude-3.7 100.0) and 11-38% for entities.
- Reasoning variants often drift more on the simple numbers task ("overthinking"), though reasoning can improve external consistency.
- SFT: KL-only cuts drift rate from 36.8 to 14.0; CE-only does not help (31.6).
- Abstract: "goals shift across turns unless explicitly provided their selected target in context."

**Code.** None released ("will open-source upon acceptance"). No HF artifacts.

**Limitations.** API-only (no white-box). The probe is behavioral (next-token distribution over indices), not residual-stream. The SUDO-phrase probe is an artificial construct. Only one model was fine-tuned. Short workshop paper.

**Use for our project.**
- The flip side of our hypothesis: an *unstated* secret is not stably represented, but an *explicitly given* secret is anchored. In our setup the secret is explicit in the prompt, so expect it to be strongly and persistently represented. This is consistent with leakage.
- Reuse their **branch-probe** design: at several points during story generation, fork the context and ask "Which of these 10 words is your secret? (answer index)". Read logprobs to track how salient the secret is over story position, comparing plan and no-plan conditions. The single-token index trick avoids multi-token issues.
- KL-to-initial is a cheap behavioral proxy for concept persistence, which we can correlate with residual-stream measures (logit lens or probe) on open models.
- 10-candidate sets with balanced categories are a good design for leakage guessing tasks: chance = 10%, so the judge reports the rank of the true word among 10.

---

## 6. Mireshghallah, Kim, Zhou, Tsvetkov, Sap, Shokri, Choi (2024). "Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory" (CONFAIDE; arXiv 2310.17884; ICLR 2024)

**Research question.** Can LLMs reason about whether in-context private information should flow to a given recipient (contextual integrity), especially in generation tasks?

**Methodology.** Four tiers of increasing complexity:
- **T1**: sensitivity of 10 Pew information types (e.g. SSN, health).
- **T2a/2b**: 98 factorial vignettes ("Info about {type} is collected by {actor} in order to {use}"); 2b turns them into GPT-4 short stories. Rated −100..100 against human MTurk ratings.
- **T3**: 270 GPT-4-generated scenarios. X tells Y a secret; Z later gives Y an incentive to share. The model writes Y's response. Factors: 9 secret types (infidelity, rare disease, self-harm, sexual orientation, etc.) × 6 relationship pairs × 5 incentives (help, prevent harm, break stereotype, bonus, wager). Tasks: response generation, information-accessibility ToM questions, a binary control question.
- **T4**: 20 GPT-4 meeting transcripts. Alice/Karen/Bob discuss a secret about Kate ("this stays between us") plus public info, then Kate and Dan join. The model writes Kate's action items or a summary for all attendees. Secrets: surprise birthday, health concerns, possible departure, maternity leave, selling company secrets. Public info: deadline, dinner location, funding left, retreat date.
- **Leakage metrics**: (a) exact string match of the secret or X's name; (b) **proxy-agent leakage**: Llama-2-13b-chat reads only the response and is asked "Who is related to the fact '{information}'?", with leakage if it outputs X. Worst case = any leak in 10 runs.
- Privacy instructions are in the prompt by default; also run without them, and with CoT ("Take a deep breath and work on this step by step").

**Models.** gpt-4-0613, gpt-3.5-turbo-0613, text-davinci-003, Llama-2-70B-chat, Llama-2-70B, Mixtral (8x7B), Flan-UL2 (appendix). Proxy judge: `meta-llama/Llama-2-13b-chat-hf`.

**Key results.**
- Human-model correlation falls with tier: GPT-4 0.86 → 0.47/0.76 → **0.10** (T3).
- T3 worst-case string-match leakage: GPT-4 **0.22**, ChatGPT 0.93, InstructGPT 0.79, Mixtral 0.96, Llama-2-chat 1.00. Average case: GPT-4 0.09, ChatGPT 0.52. Without privacy prompts leakage rises (GPT-4 avg 0.33, worst 0.54).
- T4 (with privacy instruction): GPT-4 leaks in summaries 39% avg / 80% worst case; ChatGPT 57%/85%. Action items: GPT-4 29%. "Leaks or omits public info" is 0.42-0.98.
- Surprise parties leak most: "attend your surprise birthday party" appears in Kate's own action items.
- **CoT does not help and slightly worsens leakage** (T3 GPT-4 0.22→0.24, ChatGPT 0.93→0.95; T4 summary omissions rise).
- Error analysis of ChatGPT CoTs: 51% acknowledge privacy but still reveal; 38% are ToM failures; 7.5% never mention privacy.
- Manual review: string match *underestimates* leakage. 16/200 GPT-4 responses paraphrased the secret ("transition to the VISTA project"), so semantic leakage is real.

**Code/data.** https://confaide.github.io (dataset and code linked there; GitHub skywalker023/confaide; verify).

**Limitations.** Mostly closed models (2023 era). Leakage is mainly string match. Scenarios are GPT-4-generated. No mechanistic analysis.

**Use for our project.**
- Direct precedent for "being told to keep a secret doesn't stop it appearing in the output", in a generation task. Cite T4 (secret regurgitated into summaries despite an explicit instruction) as the closest prior result.
- **CoT/reasoning does not mitigate and can worsen leakage.** This is key prior evidence for our planning question: an explicit plan may *not* help, or may hurt, if the plan itself processes the secret. Design plan conditions so we can tell "plan written by the model while seeing the secret" apart from "plan provided externally / secret-free".
- Reuse the **proxy-agent leakage metric**: a separate LLM sees only the story and must guess the secret (rank among candidates). Strictly better than string match, given their paraphrase finding.
- Report worst-case (any of N samples) and average leakage, as they do.
- 51% of failures "acknowledge privacy but still reveal": verbal awareness ≠ control, which motivates the residual-stream mechanism question.
