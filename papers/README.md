# Downloaded Papers (27)

★ = user-specified in the research topic. Deep-reading notes are in `../literature_review.md`; raw per-paper notes are in `../logs/notes_*.md`. Page chunks are in `pages/`.

## A. Secret keeping & thematic leakage (core)
1. ★ **Can You Keep a Secret? Involuntary Information Leakage in Language Model Writing**. Holtzman & West, 2026. arXiv 2605.10794. `2605.10794_holtzman2026_can_you_keep_a_secret.pdf`. *The paradigm we extend*: a secret word in the system prompt plus "write a story"; 2AFC judge discrimination; up to 79% leakage; "actively hide" inverts the signal; leakage scales with size (Gemma-3-12B 83%); rigid essays leak less; decoys redirect leakage.
2. ★ **Can LLMs Keep a Secret? (ConfAIde)**. Mireshghallah et al., 2023 (ICLR'24). arXiv 2310.17884. `2310.17884_mireshghallah2023_confaide_keep_secret.pdf`. Contextual-integrity privacy benchmark; GPT-4 reveals private info 39% of the time.
3. ★ **LLMs Can't Play Hangman: Necessity of a Private Working Memory**. Baldelli et al., 2026. arXiv 2601.06973. `2601.06973_baldelli2026_llms_cant_play_hangman.pdf`. Impossibility result for hidden state when the agent only has the public transcript; private memory fixes it.
4. ★ **Probing the Lack of Stable Internal Beliefs in LLMs**. Luo et al., 2026. arXiv 2603.25187. `2603.25187_luo2026_lack_stable_internal_beliefs.pdf`. A 20-questions game shows LLMs don't keep a stable implicit secret.
5. ★ **Don't Think of the White Bear: Ironic Negation in Transformers**. Mann et al., 2025. arXiv 2511.12381. `2511.12381_mann2025_white_bear_ironic_negation.pdf`. "Do not mention X" causes ironic rebound; suppression requires activating the concept.
6. **Inadvertent Context Leakage in Language Models**. Fairoze et al., 2026. arXiv 2608.19857. `2608.19857_fairoze2026_inadvertent_context_leakage.pdf`. In-context secrets bias benign outputs enough to reconstruct digits/SSNs; more capable models leak more.
7. **"Don't Say It!": Constraints, Compliance, and Communication when LMs Play Taboo**. 2026. arXiv 2607.00601. `2607.00601_2026_dont_say_it_taboo.pdf`.
8. **Do LLMs Know What Is Private Internally? Probing and Steering Contextual Privacy Norms**. Wang, Xiong, Shu, 2026. arXiv 2604.00209. `2604.00209_wang2026_private_internally_probing_steering.pdf`.

## B. Eliciting secrets from internals (mechanism)
9. ★ **Towards Eliciting Latent Knowledge from LLMs with Mechanistic Interpretability (Taboo model)**. Cywiński et al., 2025. arXiv 2505.14352. `2505.14352_cywinski2025_taboo_eliciting_latent_knowledge.pdf`. Logit lens and SAEs recover a secret word from a fine-tuned Gemma-2-9B.
10. ★ **Eliciting Secret Knowledge from Language Models**. Cywiński et al., 2025. arXiv 2510.01070. `2510.01070_cywinski2025_eliciting_secret_knowledge.pdf`. Taboo / user-gender / SSC model organisms; prefill attacks work best; code and models are public.
11. **Activation Oracles**. 2025. arXiv 2512.15674. `2512.15674_2025_activation_oracles.pdf`. LLMs trained to answer questions about activations; evaluated on Taboo.
12. **When Activation Oracles Learn Not to Read**. Bersia & Gaintseva, 2026. arXiv 2607.23379. `2607.23379_2026_activation_oracles_blind_spots.pdf`. Behavioral leakage, decodability and verbalizability come apart.
13. **Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences**. 2025. arXiv 2510.13900. `2510.13900_2025_narrow_finetuning_traces.pdf`.

## C. Steering / ablation / erasure tools
14. **Refusal in LMs Is Mediated by a Single Direction**. Arditi et al., 2024. arXiv 2406.11717. `2406.11717_arditi2024_refusal_single_direction.pdf`. Difference-in-means direction plus directional ablation, the template for "remove the secret concept".
15. **Steering Llama 2 via Contrastive Activation Addition**. Rimsky et al., 2023. arXiv 2312.06681. `2312.06681_rimsky2023_contrastive_activation_addition.pdf`.
16. **Steering LMs with Activation Engineering (ActAdd)**. Turner et al., 2023. arXiv 2308.10248. `2308.10248_turner2023_activation_addition.pdf`.
17. **LEACE: Perfect Linear Concept Erasure in Closed Form**. Belrose et al., 2023. arXiv 2306.03819. `2306.03819_belrose2023_leace_concept_erasure.pdf`.

## D. Planning in LMs (implicit lookahead)
18. ★ **Do Language Models Plan Ahead for Future Tokens?** Wu, Morris, Levine, 2024. arXiv 2404.00859. `2404.00859_wu2024_plan_ahead_future_tokens.pdf`. Pre-caching vs. breadcrumbs.
19. ★ **Emergent Response Planning in LLMs**. Dong et al., 2025. arXiv 2502.06258. `2502.06258_dong2025_emergent_response_planning.pdf`. Prompt activations encode global attributes of the upcoming response, including story character choices.
20. **ParaScopes: What do LM Activations Encode About Future Text?** 2025. arXiv 2511.00180. `2511.00180_2025_parascopes_future_text.pdf`.
21. **What's the plan? Metrics for implicit planning in LLMs**. 2026. arXiv 2601.20164. `2601.20164_2026_whats_the_plan_implicit_planning.pdf`.
22. **Detecting and Characterizing Planning in Language Models**. 2025. arXiv 2508.18098. `2508.18098_2025_detecting_planning_in_lms.pdf`.
23. **Unlocking the Future: Look-Ahead Planning Mechanistic Interpretability**. 2024. arXiv 2406.16033. `2406.16033_2024_lookahead_planning_mech_interp.pdf`.

## E. Story planning / suspense / foreshadowing
24. ★ **Creating Suspenseful Stories: Iterative Planning with LLMs**. Xie & Riedl, 2024. arXiv 2402.17119. `2402.17119_xie2024_suspenseful_stories_planning.pdf`.
25. **Spoiler Alert: Narrative Forecasting as a Metric for Tension in LLM Storytelling**. 2026. arXiv 2604.09854. `2604.09854_2026_spoiler_alert_narrative_forecasting.pdf`. Measures predictability/foreshadowing of story continuations.
26. **Re3: Generating Longer Stories with Recursive Reprompting and Revision**. Yang et al., 2022. arXiv 2210.06774. `2210.06774_yang2022_re3_story_generation.pdf`. Plan → draft pipeline.
27. **DOC: Improving Long Story Coherence with Detailed Outline Control**. Yang et al., 2023. arXiv 2212.10077. `2212.10077_yang2022_doc_outline_control.pdf`. Hierarchical outlines.
