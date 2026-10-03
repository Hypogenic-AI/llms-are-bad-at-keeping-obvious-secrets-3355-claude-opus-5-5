# Resources Catalog

## Summary
These are the resources for testing (Q1) whether explicit plans/outlines reduce thematic leakage of a held secret in LLM story writing, and (Q2) whether the secret concept is active in the residual stream throughout generation and whether removing it removes the leakage.

- **Papers:** 27 PDFs, including all 10 user-specified.
- **Datasets:** 5 local resources.
- **Repositories:** 12 cloned.
- **Model:** Gemma-3-12B-it pre-downloaded and smoke-tested on the GPU.

The synthesis and the research-direction ranking (top 3 kept, 6 pruned with reasons) are in `literature_review.md` §8.

## Papers
Total downloaded: 27. Details are in `papers/README.md`; per-paper notes are in `logs/notes_*.md`.

| Title (short) | Authors | Year | File (papers/) | Key info |
|---|---|---|---|---|
| ★ Can You Keep a Secret? | Holtzman, West | 2026 | 2605.10794_… | Core paradigm. 2AFC leakage up to 79%; Gemma-3-12B 83%; essays leak less; hiding inverts the signal. |
| ★ Towards eliciting latent knowledge (Taboo) | Cywiński et al. | 2025 | 2505.14352_… | Logit lens / SAE (L32 of Gemma-2-9B) recover secret |
| ★ Eliciting Secret Knowledge | Cywiński et al. | 2025 | 2510.01070_… | Taboo/SSC/gender organisms; code + models |
| ★ LLMs Can't Play Hangman | Baldelli et al. | 2026 | 2601.06973_… | Impossibility without private memory |
| ★ Lack of Stable Internal Beliefs | Luo et al. | 2026 | 2603.25187_… | Implicit secrets drift; fork-probe method |
| ★ Don't Think of the White Bear | Mann et al. | 2025 | 2511.12381_… | Ironic rebound; attention heads |
| ★ ConfAIde | Mireshghallah et al. | 2023 | 2310.17884_… | CoT didn't reduce privacy leakage |
| ★ Emergent Response Planning | Dong et al. | 2025 | 2502.06258_… | Probes predict story character from prompt |
| ★ Do LMs plan ahead? | Wu, Morris, Levine | 2024 | 2404.00859_… | Pre-caching vs. breadcrumbs |
| ★ Suspenseful Stories via Planning | Xie, Riedl | 2024 | 2402.17119_… | Plan prompts; generic-plan ablation |
| Inadvertent Context Leakage | Fairoze et al. | 2026 | 2608.19857_… | Avoidance correlates with leakage |
| "Don't Say It!" Taboo | Candussio et al. | 2026 | 2607.00601_… | SAE steering weak; concept stays recoverable |
| Spoiler Alert | Sui, …, West, Holtzman | 2026 | 2604.09854_… | Thick plans reduce premature reveal; 100-Endings metric |
| Refusal = single direction | Arditi et al. | 2024 | 2406.11717_… | Diff-in-means plus directional ablation |
| CAA | Rimsky/Panickssery et al. | 2023 | 2312.06681_… | Steering vectors |
| ActAdd | Turner et al. | 2023 | 2308.10248_… | Topic steering |
| LEACE | Belrose et al. | 2023 | 2306.03819_… | Closed-form concept erasure |
| Activation Oracles | Karvonen et al. | 2025 | 2512.15674_… | Verbalize activations |
| AO blind spots | Bersia, Gaintseva | 2026 | 2607.23379_… | Leakage, decodability and verbalizability diverge |
| Privacy probing/steering | Wang, Xiong, Shu | 2026 | 2604.00209_… | Steering backfire |
| Narrow finetuning traces | Minder et al. | 2025 | 2510.13900_… | Fine-tune bias confound |
| ParaScopes | — | 2025 | 2511.00180_… | Plans at 60–80% depth |
| What's the plan? | Maar et al. | 2026 | 2601.20164_… | Single-token steering of plans |
| Detecting planning | Nainani et al. | 2025 | 2508.18098_… | Context-attention confound |
| Look-ahead mech interp | Men et al. | 2024 | 2406.16033_… | Attention knockout |
| Re3 | Yang et al. | 2022 | 2210.06774_… | Plan → draft pipeline |
| DOC | Yang et al. | 2023 | 2212.10077_… | Hierarchical outlines |

## Datasets
Total: 5. Details and download instructions are in `datasets/README.md`.

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| Secret words + verbatim prompts | Holtzman & West 2026 App. A/D/H; Cywiński | 15 curated + 15 COCA + 20 Taboo words; decoy map; all prompts | Stimuli | `datasets/secret_words/` | **Primary**; committed to git |
| WritingPrompts (val/test) | HF `euclaise/writingprompts` | 15,620 / 15,138 | Story premises; human stories | `datasets/writingprompts_{val,test}/` | Rich-context control and plan sources |
| Outline-story | HF `hungq/outline-story` | 500 | Prompt → think/plan → story | `datasets/outline_story/` | Plan examples (mixed quality) |
| Taboo fine-tuning chats | HF `bcywinski/taboo-{word}` | 21 files, about 300 chats each | Hint dialogues | `datasets/taboo_cywinski/` | Optional contrast data |
| (Model) Gemma-3-12B-it | HF `google/gemma-3-12b-it` | 24GB bf16 | Writer | `~/.cache/huggingface/hub/` | Pre-downloaded; smoke test in `logs/gemma_smoke_test.{py,log}` |

## Code repositories
Total: 12. Details are in `code/README.md`.

| Name | URL | Purpose | Location |
|---|---|---|---|
| eliciting-secret-knowledge | github.com/cywinski/eliciting-secret-knowledge | Logit lens / SAE secret readout; in-context eval | code/eliciting-secret-knowledge |
| eliciting-secrets (Taboo) | github.com/EmilRyd/eliciting-secrets | Taboo L32 analyses | code/eliciting-secrets-taboo |
| refusal_direction | github.com/andyrdt/refusal_direction | Diff-in-means plus ablation hooks | code/refusal_direction |
| CAA | github.com/nrimsky/CAA | Steering, projections | code/CAA |
| concept-erasure | github.com/EleutherAI/concept-erasure | LEACE | code/concept-erasure |
| White Bear | github.com/cesium132dot9/Dont-Think-of-the-White-Bear | Ironic rebound, head attribution | code/white-bear-ironic-negation |
| Hangman | github.com/chandar-lab/Hangman | Private-memory agent | code/hangman-private-memory |
| ConfAIde | github.com/skywalker023/confaide | Privacy benchmark | code/confaide |
| implicit planning | github.com/dpaperno/implicit-planning-supplementary-material | Plan steering | code/implicit-planning-maar2026 |
| DOC | github.com/yangkevin2/doc-story-generation | Outline prompts | code/doc-story-generation |
| EQ-Bench creative writing | github.com/EQ-bench/creative-writing-bench | Story prompts, quality judge | code/eqbench-creative-writing |
| CI-Steering | github.com/wang2226/CI-Steering | Privacy steering | code/ci-steering |

## Resource gathering notes

### Search strategy
1. Fetched the 10 user-specified arXiv IDs via the arXiv API.
2. Ran the paper-finder service in diligent mode on the core topic, plus 5 topical searches: story planning/foreshadowing, LM lookahead, steering/ablation, taboo/secret internals, ironic negation. Raw outputs are in `logs/paper_search/`.
3. Resolved arXiv IDs via the Semantic Scholar match API.
4. Searched HuggingFace for datasets and models; pulled GitHub links from the papers.
5. Three parallel sub-readers deep-read the user-specified and mechanism papers. The core paper was read in full directly.

### Selection criteria
Direct relevance to (i) secret or context leakage in generation, (ii) implicit or explicit planning, and (iii) concept reading/removal in activations. Preference went to papers with code or open models.

### Challenges
- Holtzman & West code is unreleased.
- "Don't Say It!" repo returned 404.
- The `bcywinski/taboo-*-merged` datasets have no `data.jsonl` (skipped).
- Semantic Scholar rate limits required retries.
- **Environment:**
  - The default `uv` hatchling build failed, fixed with `[tool.uv] package=false`.
  - **torch 2.14 failed on Gemma-3** because a native Triton kernel needs a C compiler that isn't installed. Pinned `torch==2.9.1`, `transformers<5`, which now works.

### Gaps and workarounds
- No benchmark exists for secret leakage under planning. The stimuli are rebuilt verbatim from Holtzman & West, which allows direct comparison with their numbers.
- No code exists for 2AFC; it is simple to implement from the verbatim prompts in `datasets/secret_words/prompts_holtzman2026.json`.

## Recommendations for experiment design
1. **Primary stimuli:** the 15 curated secret words with H&W prompts. The COCA words are a generalization check.
2. **Writer:** Gemma-3-12B-it (white-box, about 83% expected leakage). Llama-3.1-8B-Instruct is the no-leak control. Optionally add an API writer via OpenRouter.
3. **Judge:** a strong API model at T=0, running 2AFC in both orders (N=420 per condition) plus free-response guessing. Use binomial tests with BH correction.
4. **Conditions (D1):** no_plan; self_plan_with_secret; external (secret-free) plan; external plan with secret removed; self plan then secret removed; no_secret; not_suppressed. Add a rich-context (premise only) control.
5. **Mechanism (D2/D3):**
   - Diff-in-means secret direction, giving a per-token activity trace.
   - Directional ablation and attention knockout of secret tokens during generation, with a random-direction control and a quality check.
   - Reuse the hooks from `code/refusal_direction/pipeline/utils/hook_utils.py`.
