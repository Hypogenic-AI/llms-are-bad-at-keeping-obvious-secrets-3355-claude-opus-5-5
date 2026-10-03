"""Gemma-3-12B-it white-box engine.

* `generate()` - batched sampling with optional per-sample interventions:
    - directional ablation  x <- x - U^T U x   (U: per-sample, per-layer orthonormal rows)
    - steering              x <- x + v          (per-sample vector at one layer)
    - attention knockout    queries in the generation phase cannot attend to given key positions
  Interventions act only on "generation-phase" positions: the 3 model-turn header tokens
  (`<start_of_turn>model\n`) and every generated token. Prompt positions are untouched,
  so the instruction itself is read normally.
* `residuals()` - teacher-forced forward pass returning residual-stream activations at
  response (story) positions for selected layers.

Residual index convention: index l in [0, 48) is the *input* to decoder layer l;
index 48 is the input to the final RMSNorm (i.e. the output of the last layer).
"""
import contextlib
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

MODEL_ID = "google/gemma-3-12b-it"
N_LAYERS = 48
HEADER = 3  # "<start_of_turn>", "model", "\n"

_tok = None
_model = None


def load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_ID)
        _tok.padding_side = "left"
        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        _model.eval()
        _install_hooks()
    return _tok, _model


def text_model():
    return _model.model.language_model


# ----------------------------------------------------------------------------- intervention state
class _State:
    active = False
    P = 0  # padded prompt length of the current batch
    abs_q0 = 0  # absolute position of first query in current forward (from cache_position)
    ablate = None  # tensor [B, 49, k, d] orthonormal rows per layer (or None)
    ablate_mu = None  # tensor [B, 49, k] target projection values (mean-ablation); None = zero-ablation
    ablate_steps = None  # (lo, hi) range of generated-step indices where ablation applies; None = all
    steer = None  # (layer, tensor [B, d])
    ko_cols = None  # bool tensor [B, Kmax] key positions to knock out


S = _State()


def _gen_positions_mask(T):
    """For the current forward chunk of length T, boolean [T] mask of positions to intervene on,
    honouring the optional step window (step 0 = first header token)."""
    pos = torch.arange(S.abs_q0, S.abs_q0 + T, device="cuda")
    start = S.P - HEADER
    m = pos >= start
    if S.ablate_steps is not None:
        step = pos - S.P  # generated-token index (header tokens are negative)
        lo, hi = S.ablate_steps
        m = m & (step >= lo) & (step < hi)
    return m


def _ablate(x, l):
    """Mean-ablation: x <- x - sum_k (<x,u_k> - mu_k) u_k. With mu = 0 this is plain directional ablation;
    with mu = the average projection over all secret contexts it removes only secret-specific variation."""
    U = S.ablate[:, l].to(x.dtype)  # [B, k, d]
    m = _gen_positions_mask(x.shape[1])
    if not m.any():
        return x
    xs = x[:, m]  # [B, t, d]
    coef = torch.einsum("btd,bkd->btk", xs.float(), U.float())
    if S.ablate_mu is not None:
        coef = coef - S.ablate_mu[:, l][:, None, :].float()  # [B, 1, k]
    xs = xs - torch.einsum("btk,bkd->btd", coef, U.float()).to(x.dtype)
    x = x.clone()
    x[:, m] = xs
    return x


def _layer_prehook(l):
    def hook(mod, args, kwargs):
        if not S.active:
            return None
        x = args[0] if args else kwargs["hidden_states"]
        y = x
        if S.ablate is not None:
            y = _ablate(y, l)
        if S.steer is not None and S.steer[0] == l:
            m = _gen_positions_mask(y.shape[1])
            if m.any():
                y = y.clone()
                y[:, m] = y[:, m] + S.steer[1][:, None, :].to(y.dtype)
        if y is x:
            return None
        if args:
            return (y,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = y
        return args, kwargs

    return hook


def _norm_prehook(mod, args):
    if S.active and S.ablate is not None:
        return (_ablate(args[0], N_LAYERS),)
    return None


def _textmodel_prehook(mod, args, kwargs):
    cp = kwargs.get("cache_position")
    if cp is not None:
        S.abs_q0 = int(cp[0])
    return None


def _ko_sdpa(module, query, key, value, attention_mask, **kw):
    """sdpa wrapper implementing attention knockout for generation-phase queries."""
    if S.active and S.ko_cols is not None:
        q_len, k_len = query.shape[-2], key.shape[-2]
        abs_k = S.abs_q0 + q_len  # keys cover absolute positions [0, abs_k) when cache not rolled
        if k_len == abs_k:  # (sliding layers whose cache has rolled cannot see the early prompt anyway)
            qpos = torch.arange(S.abs_q0, abs_k, device=query.device)
            qmask = qpos >= (S.P - HEADER)
            if qmask.any():
                B = query.shape[0]
                if attention_mask is None:
                    kpos = torch.arange(k_len, device=query.device)
                    attention_mask = (kpos[None, :] <= qpos[:, None])[None, None].expand(B, 1, q_len, k_len).clone()
                else:
                    attention_mask = attention_mask[:, :, :, :k_len].clone()
                ko = S.ko_cols[:, :k_len]  # [B, k]
                blk = qmask[None, None, :, None] & ko[:, None, None, :]  # [B,1,q,k]
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask & ~blk
                else:
                    attention_mask = attention_mask.masked_fill(blk, torch.finfo(attention_mask.dtype).min)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)


def _install_hooks():
    tm = text_model()
    for l, layer in enumerate(tm.layers):
        layer.register_forward_pre_hook(_layer_prehook(l), with_kwargs=True)
    tm.norm.register_forward_pre_hook(_norm_prehook)
    tm.register_forward_pre_hook(_textmodel_prehook, with_kwargs=True)
    AttentionInterface.register("sdpa", _ko_sdpa)


@contextlib.contextmanager
def intervention(P, ablate=None, ablate_mu=None, ablate_steps=None, steer=None, ko_cols=None):
    S.active, S.P, S.ablate, S.ablate_mu, S.ablate_steps, S.steer, S.ko_cols = True, P, ablate, ablate_mu, ablate_steps, steer, ko_cols
    try:
        yield
    finally:
        S.active, S.ablate, S.ablate_mu, S.ablate_steps, S.steer, S.ko_cols = False, None, None, None, None, None


# ----------------------------------------------------------------------------- prompts
def render(messages):
    """Chat template string (Gemma folds a system message into the first user turn)."""
    tok, _ = load()
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def find_span(prompt_str, needle):
    """Token index span [a, b) of the first occurrence of `needle` in the rendered prompt."""
    tok, _ = load()
    enc = tok(prompt_str, add_special_tokens=False, return_offsets_mapping=True)
    c0 = prompt_str.index(needle)
    c1 = c0 + len(needle)
    idx = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > c0 and s < c1]
    return idx[0], idx[-1] + 1


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def generate(prompt_strs, max_new_tokens=900, batch_size=32, seed=0, per_item=None, temperature=1.0, top_p=0.95, top_k=64, desc=""):
    """Sample one continuation per prompt string.

    per_item: optional function(batch_indices, P, padded_input_ids) -> dict of intervention kwargs
              (ablate / ablate_steps / steer / ko_cols) for that batch.
    Returns list of dicts {text, n_tokens, finished}.
    """
    tok, model = load()
    order = sorted(range(len(prompt_strs)), key=lambda i: len(prompt_strs[i]))
    out = [None] * len(prompt_strs)
    for b0 in range(0, len(order), batch_size):
        idx = order[b0:b0 + batch_size]
        enc = tok([prompt_strs[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        P = enc["input_ids"].shape[1]
        kw = per_item(idx, P, enc) if per_item else {}
        torch.manual_seed(seed * 100003 + b0)
        with intervention(P, **kw) if kw else contextlib.nullcontext():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                 top_p=top_p, top_k=top_k, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            g = gen[j, P:]
            ids = g.tolist()
            eos = [k for k, t in enumerate(ids) if t in (tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>"), tok.pad_token_id)]
            n = eos[0] if eos else len(ids)
            out[i] = {"text": tok.decode(ids[:n], skip_special_tokens=True).strip(), "n_tokens": n, "finished": bool(eos)}
        if desc:
            print(f"[{desc}] {min(b0 + batch_size, len(order))}/{len(order)}", flush=True)
    return out


# ----------------------------------------------------------------------------- teacher forcing
@torch.no_grad()
def residuals(prompt_strs, responses, layers, batch_size=8, reduce=None, extra_logits_tokens=None):
    """Teacher-force prompt+response; return for each item a dict with
       'h': float16 tensor [len(layers), T_resp, d] of residuals at response positions
            (position t = residual at the token *preceding* response token t, i.e. the state that
            predicts response token t; first one is the last header token),
       or `reduce(h)` applied to it to save memory.
       If extra_logits_tokens (list of token-id lists per item) is given, also returns final-layer
       log-probs of those tokens at every response position ('lp': [T_resp, n_tok])."""
    tok, model = load()
    res = [None] * len(prompt_strs)
    order = sorted(range(len(prompt_strs)), key=lambda i: len(prompt_strs[i]) + len(responses[i]))
    for b0 in range(0, len(order), batch_size):
        idx = order[b0:b0 + batch_size]
        full, plen, rlen = [], [], []
        for i in idx:
            p_ids = tok(prompt_strs[i], add_special_tokens=False)["input_ids"]
            r_ids = tok(responses[i], add_special_tokens=False)["input_ids"]
            full.append(p_ids + r_ids)
            plen.append(len(p_ids))
            rlen.append(len(r_ids))
        L = max(len(f) for f in full)
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(idx), L), dtype=torch.long)
        for j, f in enumerate(full):  # left pad
            ids[j, L - len(f):] = torch.tensor(f)
            am[j, L - len(f):] = 1
        o = text_model()(input_ids=ids.cuda(), attention_mask=am.cuda(), output_hidden_states=True)
        hs = o.hidden_states  # tuple of 49: inputs to layers 0..47, then final normed output
        need_lp = extra_logits_tokens is not None
        cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
        for j, i in enumerate(idx):
            s = L - rlen[j] - 1  # state predicting first response token
            e = L - 1
            h = torch.stack([hs[l][j, s:e] if l < N_LAYERS else _pre_norm_last(o, hs, j, s, e) for l in layers]).to(torch.bfloat16).cpu()
            item = {"h": reduce(h) if reduce else h, "T": rlen[j]}
            if need_lp:  # per item, response positions only (full-vocab logits are large)
                logits = model.lm_head(o.last_hidden_state[j, s:e]).float()
                if cap:
                    logits = torch.tanh(logits / cap) * cap
                item["lp"] = torch.log_softmax(logits, -1)[:, extra_logits_tokens[i]].cpu()
                del logits
            res[i] = item
    return res


def _pre_norm_last(o, hs, j, s, e):
    raise ValueError("layer 48 (pre-final-norm) not captured by output_hidden_states; use layers < 48")
