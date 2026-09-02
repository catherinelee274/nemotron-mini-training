"""
MoE-specific helpers shared by both GRPO scripts.

Two things we care about for the Nemotron-proxy validation:

  1. confirm_router_frozen(model): verify the MoE *router/gate* layers are NOT
     trainable and carry no LoRA adapter. Unsloth disables router fine-tuning
     for MoE by default; this asserts and logs it so we have proof in the run.

  2. probe_expert_routing(model, tokenizer): run one forward pass with hooks on
     the router linears and report how many distinct experts fire — i.e. that
     routing is live and not collapsed onto a single expert.

Router naming: for Qwen3-MoE and OLMoE the router is the per-layer Linear named
`...mlp.gate` (maps hidden -> num_experts). The expert FFN projections are
`...experts.{i}.{gate_proj,up_proj,down_proj}`. We must LoRA the latter, never
the former. Note that the substring "gate" appears in both `mlp.gate` (router)
and `gate_proj` (expert); we match the router precisely on a `.gate` suffix.
"""

import torch
import torch._dynamo


def _is_router_param(name):
    # router weight/bias live at `...mlp.gate.weight` / `...mlp.gate.bias`
    # (NOT `gate_proj`, which is an expert projection).
    return ".gate.weight" in name or ".gate.bias" in name


def _is_router_module_name(name):
    return name.endswith("mlp.gate") or name.endswith(".gate")


def confirm_router_frozen(model, verbose=True):
    """Assert MoE routers are frozen and adapter-free. Returns a summary dict."""
    router_params = []
    router_trainable = []
    lora_on_router = []
    for name, p in model.named_parameters():
        if _is_router_param(name):
            router_params.append(name)
            if p.requires_grad:
                router_trainable.append(name)
        # any LoRA adapter param attached to a router module is a red flag
        if "lora_" in name and ".gate." in name and "gate_proj" not in name:
            lora_on_router.append(name)

    # count trainable LoRA params by location: attention vs expert FFN.
    # On the vLLM-backed MoE recipe only attention is LoRA'd (experts frozen);
    # either way the router itself must carry no trainable params / adapter.
    attn_keys = ("q_proj", "k_proj", "v_proj", "o_proj")
    expert_keys = ("gate_proj", "up_proj", "down_proj")
    attn_lora = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and "lora_" in n and any(k in n for k in attn_keys))
    expert_lora = sum(p.numel() for n, p in model.named_parameters()
                      if p.requires_grad and "lora_" in n and any(k in n for k in expert_keys))

    summary = {
        "num_router_param_tensors": len(router_params),
        "router_trainable_tensors": len(router_trainable),
        "lora_adapters_on_router": len(lora_on_router),
        "trainable_attn_lora_params": int(attn_lora),
        "trainable_expert_lora_params": int(expert_lora),
    }
    if verbose:
        print("[router-freeze] router param tensors found:",
              summary["num_router_param_tensors"])
        print("[router-freeze] router tensors with requires_grad=True:",
              summary["router_trainable_tensors"], "(expect 0)")
        print("[router-freeze] LoRA adapters attached to router:",
              summary["lora_adapters_on_router"], "(expect 0)")
        print("[router-freeze] trainable LoRA params -> attention:",
              f"{summary['trainable_attn_lora_params']:,}",
              "| expert-FFN:", f"{summary['trainable_expert_lora_params']:,}")
        if router_params[:2]:
            print("[router-freeze] sample router tensors:", router_params[:2])

    assert summary["router_trainable_tensors"] == 0, \
        "Router is trainable! It must be frozen for MoE GRPO."
    assert summary["lora_adapters_on_router"] == 0, \
        "A LoRA adapter is attached to the router! Remove it."
    assert summary["num_router_param_tensors"] > 0, \
        "No router tensors found - naming assumption is wrong for this model."
    assert (summary["trainable_attn_lora_params"]
            + summary["trainable_expert_lora_params"]) > 0, \
        "No trainable LoRA params at all - nothing would learn."
    print("[router-freeze] CONFIRMED: routers frozen, no router adapters, "
          "trainable LoRA present (attention"
          + ("+experts)." if summary["trainable_expert_lora_params"] else " only)."))
    return summary


@torch.no_grad()
def probe_expert_routing(model, tokenizer, prompt="What is 17 + 25?", topk_guess=None):
    """Run one forward pass; report distinct experts selected by the routers.

    Hooks every router Linear and reads its output logits, then takes the top-k
    argmax to see which experts would be dispatched. Confirms routing is active
    and diverse (not collapsed to one expert).
    """
    selected = {"per_layer": [], "all_experts": set()}
    handles = []

    # infer experts-per-token (top-k) from config if available
    cfg = model.config
    topk = topk_guess or getattr(cfg, "num_experts_per_tok",
                                 getattr(cfg, "moe_topk", 2))

    def make_hook(layer_name):
        def hook(_module, _inp, out):
            logits = out[0] if isinstance(out, tuple) else out
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            k = min(topk, logits.shape[-1])
            idx = torch.topk(logits.float(), k, dim=-1).indices  # [tokens, k]
            experts = set(idx.flatten().tolist())
            selected["per_layer"].append((layer_name, len(experts)))
            selected["all_experts"].update(experts)
        return hook

    n_router = 0
    for name, mod in model.named_modules():
        if _is_router_module_name(name) and isinstance(mod, torch.nn.Linear):
            handles.append(mod.register_forward_hook(make_hook(name)))
            n_router += 1

    total_experts = getattr(cfg, "num_experts",
                            getattr(cfg, "num_local_experts", "?"))

    # Unsloth torch.compiles the Qwen3-MoE router; a forward hook doing a
    # data-dependent .tolist() inside the compiled graph raises a dynamo error.
    # Disable dynamo for the probe forward so the router runs eagerly and the
    # hook works. The probe is best-effort: never let it crash the run.
    prev_disable = getattr(torch._dynamo.config, "disable", False)
    n_distinct = None
    try:
        torch._dynamo.config.disable = True
        msgs = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        model(inputs)
        n_distinct = len(selected["all_experts"])
    except Exception as e:
        print(f"[routing-probe] probe forward failed ({type(e).__name__}: "
              f"{str(e)[:120]}); skipping routing stats (non-fatal).")
    finally:
        for h in handles:
            h.remove()
        torch._dynamo.config.disable = prev_disable

    if n_distinct is not None:
        print(f"[routing-probe] {n_router} router layers hooked, top-{topk} per token")
        print(f"[routing-probe] distinct experts fired across all layers: "
              f"{n_distinct} (model has {total_experts} experts total)")
        print(f"[routing-probe] routing is "
              f"{'ACTIVE & diverse' if n_distinct > topk else 'SUSPICIOUSLY narrow'}")
    return {"n_router_layers": n_router,
            "distinct_experts": n_distinct if n_distinct is not None else -1,
            "total_experts": total_experts,
            "topk": topk}
