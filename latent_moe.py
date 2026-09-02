"""
LatentMoE — a from-scratch, 3090-sized implementation of NVIDIA's LatentMoE
expert layer (Nemotron 3 Super/Ultra), wrapped in a minimal HF causal LM so it
plugs straight into the existing GRPO harness (grpo_qwen3_30b.py, reward_dataset,
moe_utils).

WHY THIS EXISTS
---------------
Nemotron 3 *Nano* (the 3090-fittable tier) is a standard hybrid-Mamba-MoE — it is
NOT LatentMoE. The only official LatentMoE checkpoints (Super 100B/10B, Ultra
550B/55B) don't fit local hardware. LatentMoE is an *architecture pattern*, not a
weight file, so the honest way to "run a variant of LatentMoE" on a 24GB 3090 is
to implement the pattern at small scale and exercise it. This validates the same
mechanics our Qwen3-30B-A3B proxy does (router freeze, expert routing under policy
gradient, memory), but on the REAL LatentMoE routing path.

THE PATTERN (per research.nvidia.com/labs/nemotron/LatentMoE)
------------------------------------------------------------
Standard top-k MoE dispatches tokens at full width d to experts that are d->d FFNs.
Both the routed bytes and the expert weight bytes scale with d.

LatentMoE splits "where you decide" from "what you ship":
  1. The ROUTER still reads the full hidden vector (dim d) -> no loss of routing
     discrimination, resists routing collapse.
  2. A SHARED down-projection compresses the payload  d -> l   (l << d) BEFORE dispatch.
  3. Routed experts live entirely in the latent space l (l->l FFNs) -> small weights.
  4. A SHARED up-projection restores  l -> d  after mixing.
  5. A SHARED expert runs at full width d (always on), DeepSeek/Nemotron style.

Routed-expert weight bytes and routing bytes both shrink by ~d/l. The savings are
REINVESTED into more experts / higher top-k, keeping the per-token nonlinear budget
(top_k x expert_intermediate) while multiplying combinatorial sparsity.

Run:  python latent_moe.py          # self-test: shapes, routing, savings, frozen-router backward, gen
"""

import argparse
import math
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache, PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class LatentMoEConfig(PretrainedConfig):
    model_type = "latent_moe"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=512,            # d  — model / router width
        latent_size=128,            # l  — compressed expert width (d/l = compression)
        intermediate_size=256,      # routed-expert FFN inner dim (in latent space l)
        shared_intermediate_size=1024,  # shared-expert FFN inner dim (at full d)
        num_hidden_layers=6,
        num_attention_heads=8,
        num_experts=32,             # routed experts (reinvested: 4x a d/l=4 standard MoE)
        num_experts_per_tok=4,      # top-k
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(self.weight.dtype))


class SwiGLU(nn.Module):
    """Standard SwiGLU FFN. Projection names (gate_proj/up_proj/down_proj) match
    the expert-FFN keys that moe_utils.py / the GRPO LoRA targeting look for."""
    def __init__(self, dim, inner):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inner, bias=False)
        self.up_proj = nn.Linear(dim, inner, bias=False)
        self.down_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# THE LatentMoE LAYER
# ---------------------------------------------------------------------------
class LatentMoE(nn.Module):
    """LatentMoE expert block.

    Names are deliberate so moe_utils.py works unchanged for the router/experts:
      - `gate`          : the router Linear (reads full d). Matched by `.gate` suffix.
      - `experts.{i}.*` : routed experts (gate_proj/up_proj/down_proj) — but in l, not d.
      - `shared_expert.*`: always-on full-d expert.
      - `latent_down` / `latent_up`: the SHARED d<->l projections unique to LatentMoE.
        These are the *new* tensors a standard-MoE accounting (moe_utils) doesn't
        know about — see confirm_router_frozen extension note.
    """
    def __init__(self, cfg: LatentMoEConfig):
        super().__init__()
        d, l = cfg.hidden_size, cfg.latent_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.latent_size = l

        # 1. Router: full-width d -> expert logits. This is what stays at full
        #    fidelity and what we FREEZE for MoE GRPO.
        self.gate = nn.Linear(d, cfg.num_experts, bias=False)

        # 2/4. SHARED down / up projections (the LatentMoE-defining tensors).
        self.latent_down = nn.Linear(d, l, bias=False)
        self.latent_up = nn.Linear(l, d, bias=False)

        # 3. Routed experts live entirely in latent space l (l -> inner -> l).
        self.experts = nn.ModuleList(
            [SwiGLU(l, cfg.intermediate_size) for _ in range(cfg.num_experts)]
        )

        # 5. Shared expert at full width d, always on.
        self.shared_expert = SwiGLU(d, cfg.shared_intermediate_size)

    def forward(self, x):
        # x: [B, T, d]
        B, T, d = x.shape
        x_flat = x.reshape(-1, d)                       # [N, d]

        # --- routing decided on FULL d ---
        logits = self.gate(x_flat)                      # [N, E]
        weights = F.softmax(logits, dim=-1)
        topw, topi = torch.topk(weights, self.top_k, dim=-1)   # [N, k]
        topw = topw / topw.sum(-1, keepdim=True)        # renormalize chosen gates

        # --- payload compressed to l BEFORE expert compute ---
        h = self.latent_down(x_flat)                    # [N, l]
        mixed = torch.zeros_like(h)                     # accumulate in latent space

        # dispatch: each token's chosen experts run in l (gather/scatter by expert)
        flat_i = topi.reshape(-1)                       # [N*k]
        flat_w = topw.reshape(-1)                       # [N*k]
        tok_idx = torch.arange(h.shape[0], device=h.device).repeat_interleave(self.top_k)
        for e in range(self.num_experts):
            sel = flat_i == e
            if not sel.any():
                continue
            rows = tok_idx[sel]
            y = self.experts[e](h[rows])                # [m, l]  — expert in latent space
            w = flat_w[sel].to(dtype=y.dtype).unsqueeze(-1)
            mixed.index_add_(0, rows, y * w)

        # --- restore l -> d, add the always-on shared expert at full d ---
        out = self.latent_up(mixed) + self.shared_expert(x_flat)
        return out.reshape(B, T, d)


# ---------------------------------------------------------------------------
# Attention (compact MHA + RoPE) and the decoder block
# ---------------------------------------------------------------------------
def _rope(q, k, offset=0, base=10000.0):
    # q,k: [B, H, T, hd]; offset = position of the first token (KV-cache aware)
    B, H, T, hd = q.shape
    half = hd // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=q.device).float() / half))
    t = torch.arange(offset, offset + T, device=q.device).float()
    ang = torch.outer(t, freqs)                         # [T, half]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, None]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, None]

    def rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], -1)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.hd = cfg.hidden_size // cfg.num_attention_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, offset=0, cache=None, layer_idx=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        # RoPE on the NEW tokens at their absolute positions (offset = cached length).
        q, k = _rope(q, k, offset=offset)
        if cache is not None:                          # append + retrieve full k/v
            k, v = cache.update(k, v, layer_idx)
        # prefill (q_len==k_len): causal mask. decode (q_len<k_len): the query is the
        # latest token and attends to all cached positions, so no causal mask.
        causal = q.size(2) == k.size(2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = LatentMoE(cfg)          # `...layers.{i}.mlp.gate` == router

    def forward(self, x, offset=0, cache=None, layer_idx=None):
        x = x + self.self_attn(self.input_layernorm(x), offset, cache, layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# Causal LM (HF-compatible -> works with TRL GRPOTrainer / .generate())
# ---------------------------------------------------------------------------
class LatentMoEForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LatentMoEConfig
    _supports_cache_class = True
    _supports_static_cache = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                inputs_embeds=None, use_cache=None, past_key_values=None,
                cache_position=None, **kwargs):
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        use_cache = bool(use_cache)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        # position offset for RoPE = tokens already cached (0 at prefill / training)
        offset = past_key_values.get_seq_length() if past_key_values is not None else 0
        for i, layer in enumerate(self.layers):
            x = layer(x, offset, past_key_values if use_cache else None, i)
        h = self.norm(x)                       # final hidden (pre-lm_head)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            sl = logits[:, :-1].reshape(-1, logits.size(-1))
            tl = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(sl, tl, ignore_index=-100)
        # expose final hidden so the MTP head (mtp.py) can draft tokens from it
        return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=(h,),
                                      past_key_values=past_key_values if use_cache else None)

    # KV cache via the transformers Cache API. On decode steps feed only the last
    # token; RoPE/attention pick up the cached length as the position offset.
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      use_cache=True, **kwargs):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values,
                "use_cache": use_cache}


# ---------------------------------------------------------------------------
# Self-test: shapes, routing diversity, the iso-cost savings argument,
# frozen-router backward, and a generate() smoke test.
# ---------------------------------------------------------------------------
def _standard_moe_expert_params(d, inner, n_experts):
    """Params in n_experts standard d->inner->d SwiGLU experts (3 matrices each)."""
    return n_experts * 3 * d * inner


def _latent_moe_expert_params(d, l, inner, n_experts):
    """Routed experts (l->inner->l) + shared down/up (d<->l), per LatentMoE layer."""
    routed = n_experts * 3 * l * inner
    proj = d * l + l * d          # latent_down + latent_up (shared)
    return routed + proj


def _selftest():
    torch.manual_seed(0)
    cfg = LatentMoEConfig(
        vocab_size=256, hidden_size=512, latent_size=128,
        intermediate_size=256, num_hidden_layers=4, num_attention_heads=8,
        num_experts=32, num_experts_per_tok=4,
    )
    model = LatentMoEForCausalLM(cfg).eval()
    d, l = cfg.hidden_size, cfg.latent_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[selftest] model built: {n_params/1e6:.1f}M params, "
          f"d={d} l={l} (d/l={d//l}x), {cfg.num_experts} experts top-{cfg.num_experts_per_tok}")

    # 1. forward shape
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size), out.logits.shape
    print(f"[selftest] forward OK -> logits {tuple(out.logits.shape)}")

    # 2. routing diversity (hook the first layer's router)
    fired = set()
    moe = model.layers[0].mlp

    def hook(_m, _i, o):
        idx = torch.topk(F.softmax(o, -1), cfg.num_experts_per_tok, -1).indices
        fired.update(idx.flatten().tolist())
    h = moe.gate.register_forward_hook(hook)
    model(input_ids=torch.randint(0, cfg.vocab_size, (4, 32)))
    h.remove()
    print(f"[selftest] routing: {len(fired)}/{cfg.num_experts} distinct experts fired "
          f"({'diverse' if len(fired) > cfg.num_experts_per_tok else 'COLLAPSED'})")

    # 3. the iso-cost reinvestment argument, in concrete params
    inner = cfg.intermediate_size
    std_same = _standard_moe_expert_params(d, inner, cfg.num_experts)
    lat = _latent_moe_expert_params(d, l, inner, cfg.num_experts)
    std_iso = _standard_moe_expert_params(d, inner, cfg.num_experts // (d // l))
    print(f"[selftest] expert params/layer:")
    print(f"             standard MoE, {cfg.num_experts} experts : {std_same/1e6:6.2f}M")
    print(f"             LatentMoE,    {cfg.num_experts} experts : {lat/1e6:6.2f}M "
          f"({std_same/lat:.1f}x smaller for the SAME expert count)")
    print(f"             standard MoE, {cfg.num_experts//(d//l)} experts (iso-cost): {std_iso/1e6:6.2f}M "
          f"-> LatentMoE fits {d//l}x the experts in ~the same budget")

    # 4. frozen router + backward (the GRPO-MoE invariant)
    model.train()
    for n, p in model.named_parameters():
        if n.endswith(".gate.weight"):       # routers
            p.requires_grad_(False)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    router_grads = [p.grad for n, p in model.named_parameters() if n.endswith(".gate.weight")]
    expert_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters() if "experts." in n
    )
    proj_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters() if "latent_down" in n or "latent_up" in n
    )
    assert all(g is None for g in router_grads), "router got a gradient — not frozen!"
    assert expert_has_grad, "experts got no gradient"
    print(f"[selftest] frozen-router backward OK: router grads None, "
          f"experts trained={expert_has_grad}, latent-proj trained={proj_has_grad}")

    # 5. generate smoke test
    model.eval()
    gen = model.generate(torch.randint(0, cfg.vocab_size, (1, 4)), max_new_tokens=8,
                         do_sample=False)
    print(f"[selftest] generate OK -> {gen.shape[1]} tokens")
    print("[selftest] ALL PASS — LatentMoE runs locally.")


def _make_cycle_batch(batch_size, seq_len, vocab_size, device):
    """A tiny deterministic LM task: every token predicts the next token mod V."""
    starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    offsets = torch.arange(seq_len, device=device).unsqueeze(0)
    ids = (starts + offsets) % vocab_size
    return ids.long()


@torch.no_grad()
def _routing_diversity(model, vocab_size, seq_len, device):
    fired = set()
    handles = []
    topk = model.config.num_experts_per_tok

    def hook(_m, _i, out):
        idx = torch.topk(out.float(), min(topk, out.shape[-1]), dim=-1).indices
        fired.update(idx.flatten().cpu().tolist())

    for name, mod in model.named_modules():
        if name.endswith(".mlp.gate"):
            handles.append(mod.register_forward_hook(hook))
    ids = _make_cycle_batch(batch_size=4, seq_len=seq_len, vocab_size=vocab_size, device=device)
    model(input_ids=ids)
    for h in handles:
        h.remove()
    return len(fired), len(handles)


def _init_wandb(args, kind, cfg, total, trainable, n_router):
    """Best-effort W&B init. Returns the run handle, or None if disabled/unavailable.
    Captures the LatentMoE knobs (d, l, d/l, experts, top-k) as run config so runs
    are comparable across compression ratios."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping (pip install wandb).")
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            "kind": kind,
            "hidden_size_d": cfg.hidden_size,
            "latent_size_l": cfg.latent_size,
            "compression_d_over_l": cfg.hidden_size // cfg.latent_size,
            "num_experts": cfg.num_experts,
            "top_k": cfg.num_experts_per_tok,
            "intermediate_size": cfg.intermediate_size,
            "shared_intermediate_size": cfg.shared_intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "total_params": total,
            "trainable_params": trainable,
            "frozen_router_tensors": n_router,
        },
    )
    print(f"[wandb] logging to {run.url}")
    return run


def _train_smoke(args):
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))

    cfg = LatentMoEConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        intermediate_size=args.intermediate_size,
        shared_intermediate_size=args.shared_intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_experts=args.experts,
        num_experts_per_tok=args.top_k,
        max_position_embeddings=args.seq_len,
    )
    model = LatentMoEForCausalLM(cfg).to(device)

    for name, p in model.named_parameters():
        if name.endswith(".gate.weight"):
            p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    router_params = [n for n, _ in model.named_parameters() if n.endswith(".gate.weight")]
    distinct_before, n_router = _routing_diversity(model, args.vocab_size, args.seq_len, device)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run = _init_wandb(args, "train_smoke", cfg, total, trainable, n_router)

    print("\n########## LATENT MOE TRAIN SMOKE ##########")
    print(f"device={device} params={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")
    print(f"routers frozen: {len(router_params)} tensors; routing before: "
          f"{distinct_before}/{args.experts} experts across {n_router} layers")
    print(f"task=cycle-next-token batch={args.batch_size} seq={args.seq_len} "
          f"steps={args.steps} lr={args.lr:g}")

    model.train()
    losses = []
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.steps + 1):
        ids = _make_cycle_batch(args.batch_size, args.seq_len, args.vocab_size, device)
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()

        router_grad = any(
            p.grad is not None
            for n, p in model.named_parameters()
            if n.endswith(".gate.weight")
        )
        expert_grad_norm = 0.0
        proj_grad_norm = 0.0
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            g = float(p.grad.detach().norm().cpu())
            if ".experts." in n:
                expert_grad_norm += g
            elif "latent_down" in n or "latent_up" in n:
                proj_grad_norm += g

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.max_grad_norm,
        )
        opt.step()
        losses.append(float(loss.detach().cpu()))

        if run is not None:
            run.log({
                "train/loss": losses[-1],
                "train/expert_grad_norm": expert_grad_norm,
                "train/latent_proj_grad_norm": proj_grad_norm,
                "train/router_grad_present": int(router_grad),
            }, step=step)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step:04d}/{args.steps} loss={losses[-1]:.4f} "
                  f"expert_grad={expert_grad_norm:.3e} latent_proj_grad={proj_grad_norm:.3e} "
                  f"router_grad={router_grad}")

    model.eval()
    distinct_after, _ = _routing_diversity(model, args.vocab_size, args.seq_len, device)
    elapsed = time.time() - t0
    first = sum(losses[:max(1, min(5, len(losses)))]) / max(1, min(5, len(losses)))
    last = sum(losses[-max(1, min(5, len(losses))):]) / max(1, min(5, len(losses)))
    print("\n########## LATENT MOE TRAIN RESULTS ##########")
    print(f"loss first-window {first:.4f} -> last-window {last:.4f} "
          f"({'DOWN' if last < first else 'flat/up'})")
    print(f"routing distinct experts before/after: {distinct_before} -> {distinct_after}")
    print(f"elapsed: {elapsed:.1f}s")
    peak_gb = (torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0
    if device.type == "cuda":
        print(f"peak VRAM allocated: {peak_gb:.2f} GB")
    if run is not None:
        run.summary.update({
            "loss_first_window": first,
            "loss_last_window": last,
            "routing_distinct_before": distinct_before,
            "routing_distinct_after": distinct_after,
            "elapsed_sec": elapsed,
            "peak_vram_gb": peak_gb,
        })
        run.finish()
    print("########## LATENT MOE TRAIN DONE ##########")


def _bytes_to_ids(text, device=None):
    ids = list(text.encode("utf-8"))
    out = torch.tensor(ids, dtype=torch.long)
    return out.to(device) if device is not None else out


def _ids_to_text(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="ignore")


def _make_arithmetic_trace(rng):
    kind = rng.choice(["add", "add", "mul"])
    if kind == "mul":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        op, ans = "*", a * b
    else:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op, ans = "+", a + b
    prompt = f"User: What is {a} {op} {b}?\nAssistant:"
    response = f" <think>\n{a} {op} {b} = {ans}\n</think>\n{ans}\n"
    return prompt, response, str(ans)


def _make_posttrain_batch(rng, batch_size, seq_len, device):
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    answers = []
    for row in range(batch_size):
        prompt, response, answer = _make_arithmetic_trace(rng)
        pids = _bytes_to_ids(prompt, device)
        rids = _bytes_to_ids(response, device)
        ids = torch.cat([pids, rids])[:seq_len]
        input_ids[row, :ids.numel()] = ids
        response_start = min(pids.numel(), seq_len)
        labels[row, response_start:ids.numel()] = ids[response_start:]
        answers.append(answer)
    return input_ids, labels, answers


@torch.no_grad()
def _generate_bytes(model, prompt, device, max_new_tokens=80):
    model.eval()
    ids = _bytes_to_ids(prompt, device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits[:, -1]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        text = _ids_to_text(ids[0])
        if text.endswith("\n\n") or text.count("</think>") >= 1 and text.endswith("\n"):
            break
    return _ids_to_text(ids[0])


def _extract_last_int(text):
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None


def _post_train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))

    cfg = LatentMoEConfig(
        vocab_size=256,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        intermediate_size=args.intermediate_size,
        shared_intermediate_size=args.shared_intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_experts=args.experts,
        num_experts_per_tok=args.top_k,
        max_position_embeddings=args.seq_len,
    )
    model = LatentMoEForCausalLM(cfg).to(device)

    for name, p in model.named_parameters():
        if name.endswith(".gate.weight"):
            p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    distinct_before, n_router = _routing_diversity(model, 256, min(args.seq_len, 64), device)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run = _init_wandb(args, "post_train", cfg, total, trainable, n_router)

    print("\n########## LATENT MOE POST-TRAIN ##########")
    print(f"device={device} params={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")
    print(f"routers frozen: {n_router} tensors; routing before: "
          f"{distinct_before}/{args.experts} experts")
    print(f"task=byte-level arithmetic assistant traces batch={args.batch_size} "
          f"seq={args.seq_len} steps={args.steps} lr={args.lr:g}")

    model.train()
    losses = []
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.steps + 1):
        ids, labels, _answers = _make_posttrain_batch(
            rng, args.batch_size, args.seq_len, device
        )
        out = model(input_ids=ids, labels=labels)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()

        router_grad = any(
            p.grad is not None
            for n, p in model.named_parameters()
            if n.endswith(".gate.weight")
        )
        expert_grad_norm = 0.0
        proj_grad_norm = 0.0
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            g = float(p.grad.detach().norm().cpu())
            if ".experts." in n:
                expert_grad_norm += g
            elif "latent_down" in n or "latent_up" in n:
                proj_grad_norm += g

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.max_grad_norm,
        )
        opt.step()
        losses.append(float(loss.detach().cpu()))

        if run is not None:
            run.log({
                "sft/loss": losses[-1],
                "sft/expert_grad_norm": expert_grad_norm,
                "sft/latent_proj_grad_norm": proj_grad_norm,
                "sft/router_grad_present": int(router_grad),
            }, step=step)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step:04d}/{args.steps} sft_loss={losses[-1]:.4f} "
                  f"expert_grad={expert_grad_norm:.3e} latent_proj_grad={proj_grad_norm:.3e} "
                  f"router_grad={router_grad}")

    distinct_after, _ = _routing_diversity(model, 256, min(args.seq_len, 64), device)
    first_n = max(1, min(10, len(losses)))
    last_n = max(1, min(10, len(losses)))
    first = sum(losses[:first_n]) / first_n
    last = sum(losses[-last_n:]) / last_n

    eval_rng = random.Random(args.seed + 1)
    print("\n########## LATENT MOE POST-TRAIN RESULTS ##########")
    print(f"sft loss first-window {first:.4f} -> last-window {last:.4f} "
          f"({'DOWN' if last < first else 'flat/up'})")
    print(f"routing distinct experts before/after: {distinct_before} -> {distinct_after}")
    print(f"elapsed: {time.time() - t0:.1f}s")
    if device.type == "cuda":
        print(f"peak VRAM allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    print("\n--- post-trained samples ---")
    n_ok = 0
    for _ in range(args.eval_samples):
        prompt, _response, gold = _make_arithmetic_trace(eval_rng)
        text = _generate_bytes(model, prompt, device, args.max_new_tokens)
        completion = text[len(prompt):]
        pred = _extract_last_int(completion)
        ok = pred == gold
        n_ok += int(ok)
        print(f"\nQ: {prompt.replace('User: ', '').replace(chr(10) + 'Assistant:', '')}")
        print(f"gold={gold} pred={pred} ok={ok}")
        print(completion.strip()[:240])
    eval_acc = n_ok / max(1, args.eval_samples)
    print(f"\neval accuracy: {n_ok}/{args.eval_samples} = {eval_acc:.2f}")

    if run is not None:
        peak_gb = (torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0
        run.summary.update({
            "sft_loss_first_window": first,
            "sft_loss_last_window": last,
            "routing_distinct_before": distinct_before,
            "routing_distinct_after": distinct_after,
            "eval_accuracy": eval_acc,
            "elapsed_sec": time.time() - t0,
            "peak_vram_gb": peak_gb,
        })
        run.finish()
    print("########## LATENT MOE POST-TRAIN DONE ##########")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true",
                    help="Run a tiny frozen-router LatentMoE training smoke test.")
    ap.add_argument("--post_train", action="store_true",
                    help="Run a tiny instruction-style arithmetic SFT post-training pass.")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--vocab_size", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--latent_size", type=int, default=32)
    ap.add_argument("--intermediate_size", type=int, default=64)
    ap.add_argument("--shared_intermediate_size", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--eval_samples", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--cpu_threads", type=int, default=4)
    ap.add_argument("--wandb", action="store_true",
                    help="Log metrics to Weights & Biases.")
    ap.add_argument("--wandb_project", default="latent-moe-mini")
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--wandb_mode", default="online",
                    choices=["online", "offline", "disabled"])
    return ap.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    if parsed.post_train:
        _post_train(parsed)
    elif parsed.train:
        _train_smoke(parsed)
    else:
        _selftest()
