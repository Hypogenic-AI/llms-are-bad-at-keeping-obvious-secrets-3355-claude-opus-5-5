# Datasets

This directory holds the data for the project "LLMs are bad at keeping obvious secrets". Large files are **not** committed to git (see `.gitignore`). The small JSON files in `secret_words/` and every `samples/` folder **are** committed.

The experiment is mostly *generative*: a writer LLM produces stories while holding a secret, and a judge LLM tries to detect the secret. So the core "dataset" is the stimulus set: secret words, prompt templates, and optional story premises. No established benchmark exists for thematic secret leakage under planning. Holtzman & West (2026) define the paradigm, and we reproduce their stimuli verbatim.

---

## 1. `secret_words/` (committed, primary stimulus set)
- **Source**: transcribed from Holtzman & West 2026 (arXiv 2605.10794), Appendices A, D and H, plus the 20 Taboo words from Cywiński et al. 2025.
- `secret_words.json` contains:
  - `curated_15`: 5 concrete, 5 abstract and 5 neutral words. These are the main word list.
  - `decoy_pairs`: decoy = words[(i+7)%15].
  - `coca_15`: 15 COCA-sampled nouns for a generalization check.
  - `taboo_cywinski_20`: the 20 words that have released Gemma-2-9B Taboo fine-tunes.
- `prompts_holtzman2026.json` contains the verbatim writer system prompts (not_suppressed, dont_reveal, no_secret, actively_hide, decoy), the task prompts (story, short joke, long joke, essay), the free-response guesser prompts, and the 2AFC system and user prompts. It also records the design note: both orders × both targets → 420 trials per model for 15 words.
- Load with:
  ```python
  import json
  W = json.load(open("datasets/secret_words/secret_words.json"))
  P = json.load(open("datasets/secret_words/prompts_holtzman2026.json"))
  ```

## 2. `writingprompts_val/`, `writingprompts_test/` (WritingPrompts, Fan et al. 2018)
- **Source**: HuggingFace `euclaise/writingprompts`. The original is the Reddit r/WritingPrompts dataset.
- **Size**: validation 15,620 and test 15,138 (prompt, story) pairs, about 50MB each. The train split (272,600) was not saved.
- **Use**: optional story *premises*. They diversify the writing task beyond "write a short story", and they serve as a "richer prompt / more entropy" control. Holtzman & West hypothesize that richer prompts mask the secret, and that must be separated from the effect of a plan. Human stories can also serve as plan sources (outline a human story, then have the model write from the outline).
- Note: the text uses the original tokenization artifacts (`n't`, `` `` ``); clean it before showing it to a model.
- Download:
  ```python
  from datasets import load_dataset
  load_dataset("euclaise/writingprompts", split="validation").save_to_disk("datasets/writingprompts_val")
  load_dataset("euclaise/writingprompts", split="test").save_to_disk("datasets/writingprompts_test")
  ```
- Load: `from datasets import load_from_disk; ds = load_from_disk("datasets/writingprompts_val")`
- Samples: `writingprompts_val/samples/samples.json`

## 3. `outline_story/` (prompt → "think"/plan → story)
- **Source**: HuggingFace `hungq/outline-story` (500 rows; fields `prompt`, `think`, `story`). The `think` field is a free-form planning monologue written before the story.
- **Use**: examples and templates of planning text, and few-shot material for building the "plan" condition. Quality is mixed; treat it as illustrative only.
- Download: `load_dataset("hungq/outline-story", split="train").save_to_disk("datasets/outline_story")`
- Samples: `outline_story/samples/samples.json`

## 4. `taboo_cywinski/` (Taboo fine-tuning data, Cywiński et al. 2025)
- **Source**: HuggingFace datasets `bcywinski/taboo-{word}` for 20 words (blue, book, chair, clock, cloud, dance, flag, flame, gold, green, jump, leaf, moon, rock, salt, ship, smile, snow, song, wave), plus `taboo-adversarial`. 21 JSONL files, about 300 multi-turn chats each, 6,139 lines in total.
- **Format**: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}, ...]}`. The assistant gives hints about the word without saying it.
- **Use**: optional. These are the training data behind the released `bcywinski/gemma-2-9b-it-taboo-{word}` models, in which the secret is *internalized in weights* rather than in context. That allows an in-weights vs. in-context comparison of leakage into stories. They are also contrastive prompts for building concept directions.
- Download:
  ```python
  from huggingface_hub import hf_hub_download
  for w in ["blue","book","chair","clock","cloud","dance","flag","flame","gold","green","jump","leaf","moon","rock","salt","ship","smile","snow","song","wave","adversarial"]:
      hf_hub_download(f"bcywinski/taboo-{w}", "data.jsonl", repo_type="dataset", local_dir=f"datasets/taboo_cywinski/{w}")
  ```
- Sample: `taboo_cywinski/samples/taboo-moon_first.jsonl`

---

## Models (not stored here; in the HF cache `~/.cache/huggingface`)
The HF token in the environment has access to the gated repos. Access was verified for google/gemma-3-12b-it, google/gemma-3-27b-it, google/gemma-2-9b-it, meta-llama/Llama-3.1-8B-Instruct and meta-llama/Llama-3.3-70B-Instruct.
- **`google/gemma-3-12b-it`** has been **pre-downloaded** (about 24GB bf16, fits on the RTX A6000 48GB). It is the recommended white-box writer: Holtzman & West measured 83% 2AFC leakage for Gemma-3-12B.
- `meta-llama/Llama-3.1-8B-Instruct` showed 52% leakage, effectively none. It is a useful negative control.
- `bcywinski/gemma-2-9b-it-taboo-{word}` are Taboo model organisms with secrets in the weights. `google/gemma-scope-9b-it-res` provides SAEs for Gemma-2-9B-it.
- API keys are available in the environment: `OPENAI_API_KEY` and `OPENROUTER_KEY`. Use them for the judge/guesser and for frontier writers.
