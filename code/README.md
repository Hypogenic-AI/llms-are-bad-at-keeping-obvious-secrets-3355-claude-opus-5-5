# Cloned Repositories

All repos are shallow clones (`--depth 1`). None needs to be installed into the main venv. They are reference implementations to copy from. The main venv (`../.venv`) already has torch 2.9.1+cu128, transformers 4.57.6, accelerate, openai, datasets, scipy and pandas.

> **Env gotcha (verified):** torch 2.14 failed on Gemma-3. It dispatches a native Triton kernel that needs a C compiler, and none is installed here. Stay on **torch 2.9.1 / transformers <5**. With that pair, Gemma-3-12B-it loads in 10s, uses 24.5GB, and generates about 11 tok/s unbatched with residual hooks working (`../logs/gemma_smoke_test.log`). In transformers 4.57, the decoder layers live at `model.model.language_model.layers` (48 layers, d=3840). Don't match the generic `*.layers`, which picks the vision tower.

| Dir | URL | Purpose for us |
|---|---|---|
| `eliciting-secret-knowledge/` | github.com/cywinski/eliciting-secret-knowledge | Code for Cywiński et al. 2510.01070. `elicitation_methods/logit_lens.py`, `sae.py`, `residual_tokens.py` show how to read a secret word out of the residual stream. `taboo/evaluate_internalization_taboo.py --in-context` evaluates a secret given *in context*, which is our setting. `prompts/` holds the auditor/guesser prompts. Models: HF `bcywinski/gemma-2-9b-it-taboo-{word}`. |
| `eliciting-secrets-taboo/` | github.com/EmilRyd/eliciting-secrets | Earlier Taboo code (2505.14352): logit lens and Gemma-Scope SAE feature analysis at layer 32 of Gemma-2-9B-it. |
| `refusal_direction/` | github.com/andyrdt/refusal_direction | Arditi et al. diff-in-means direction extraction, layer selection, directional ablation hooks and weight orthogonalization. Template for **"remove the secret concept and see if leakage disappears"**. Key: `pipeline/submodules/generate_directions.py`, `select_direction.py`, `pipeline/utils/hook_utils.py` (`get_direction_ablation_input_pre_hook`, `get_activation_addition_input_pre_hook`). |
| `CAA/` | github.com/nrimsky/CAA | Contrastive Activation Addition: steering vectors from A/B contrast pairs, layer sweeps, per-token projection plots. Useful for the "secret activity trace" across a story. |
| `concept-erasure/` | github.com/EleutherAI/concept-erasure | LEACE (`pip install concept-erasure`). Closed-form linear concept erasure, a stronger alternative to single-direction ablation. |
| `white-bear-ironic-negation/` | github.com/cesium132dot9/Dont-Think-of-the-White-Bear | Mann et al. 2511.12381: ironic rebound after "don't mention X", with attention-head attribution in Llama-3-8B-Instruct. |
| `hangman-private-memory/` | github.com/chandar-lab/Hangman | Baldelli et al. 2601.06973. Private working-memory agent architecture (structured private notes or plans). Template for a "private plan" condition and for the self-consistency fork test. |
| `confaide/` | github.com/skywalker023/confaide | ConfAIde benchmark (Mireshghallah et al.). Tier-4 meeting-summary secrets, plus CoT vs. no-CoT privacy prompts. Secondary, realistic-secret task. |
| `implicit-planning-maar2026/` | github.com/dpaperno/implicit-planning-supplementary-material | "What's the plan?" (2601.20164). Mean-difference steering at a single token flips planned future outputs (rhyme); includes the regeneration test. |
| `doc-story-generation/` | github.com/yangkevin2/doc-story-generation | DOC hierarchical outline prompts, for building graded outline conditions. |
| `eqbench-creative-writing/` | github.com/EQ-bench/creative-writing-bench | Creative-writing prompts plus an LLM-judge rubric. Story seeds, and a judge for story quality, to check that plan or ablation does not just degrade the prose. |
| `ci-steering/` | github.com/wang2226/CI-Steering | Probing and steering of contextual-privacy directions (2604.00209). Warns that single-direction steering can backfire non-monotonically. |

Not found or not cloned:
- Holtzman & West 2026 code is "redacted for review" and could not be found.
- `DanielSc4/LMtaboo` ("Don't Say It!") returned 404.
- No code exists for Spoiler Alert, Dong et al., or ParaScopes. Their prompts and hyperparameters are in `../logs/notes_planning.md`.
