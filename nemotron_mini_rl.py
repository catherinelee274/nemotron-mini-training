"""
Nemotron-3-style RL post-training, modeled at 3090 scale on our LatentMoE.

Goal: reproduce the *recipe* NVIDIA used to RL-post-train Nemotron 3 Ultra, but on a
small LatentMoE we can actually run. Per the Nemotron 3 white paper (arXiv 2512.20856)
the RL recipe is:
  - GRPO with MASKED IMPORTANCE SAMPLING (corrects the train/rollout policy gap)
  - verifiable-reward environments, multiple domains trained SIMULTANEOUSLY
We mirror that with TRL's GRPO at token-level importance sampling + num_iterations>1
(so the IS correction is actually live), and a simultaneous blend of two verifiable
rewards: MATH (exact-match) + REASONING FORMAT (<think>...</think> then the answer).

Why two stages? RL sharpens existing competence; it cannot create it from random
weights (we proved this: random init => every rollout wrong => zero reward variance
=> zero advantage => no gradient). NVIDIA RL-trains a model that is already pretrained
+ SFT'd. Since no small LatentMoE checkpoint exists to download, we manufacture the
competence ourselves with a short SFT WARM-UP, then run the RL recipe on top — this
is the "architecture fidelity" path (keep LatentMoE, pay one training stage).

  STAGE 1  SFT warm-up : teach the LatentMoE the format + easy arithmetic (router
                          frozen; embeddings/attn/experts/head train).
  STAGE 2  GRPO RL     : masked-IS GRPO on math+format rewards; LoRA on attention,
                          experts + router FROZEN (the Nemotron MoE recipe).

Run:  python nemotron_mini_rl.py --wandb                 # both stages
      python nemotron_mini_rl.py --wandb --stage sft     # warm-up only
      python nemotron_mini_rl.py --wandb --stage rl --resume outputs/nemotron_mini/sft.pt
"""

import argparse
import os
import random
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer

from latent_moe import LatentMoEConfig, LatentMoEForCausalLM
from reward_dataset import REWARD_FUNCS, SYSTEM_PROMPT, extract_answer, _completion_text
from moe_utils import confirm_router_frozen, probe_expert_routing
import rl_environments as ENV

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
SFT_CKPT = "outputs/nemotron_mini/sft.pt"


# ---------------------------------------------------------------------------
# Easy verifiable-math problems (tuned to be LEARNABLE by a ~70M model, unlike
# reward_dataset's 30B-tuned 3x2 multiplication). Reward funcs are reused as-is.
# ---------------------------------------------------------------------------
def gen_problem(rng):
    # Number range tuned to ~70M-model capacity: small enough that warm-up SFT
    # reaches a MIDDLE accuracy band (partial), leaving headroom + reward variance
    # for RL to climb. Too-hard ranges => 0% acc => no GRPO signal.
    kind = rng.choice(["add", "add", "sub", "mul1"])
    if kind == "add":
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        op, ans = "+", a + b
    elif kind == "sub":
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        a, b = max(a, b), min(a, b)
        op, ans = "-", a - b
    else:
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        op, ans = "*", a * b
    q = f"What is {a} {op} {b}?"
    trace = f"{a} {op} {b} = {ans}"
    return q, trace, str(ans)


def chat_prompt(tok, q):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def build_rl_dataset(n, seed):
    from datasets import Dataset
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n:
        q, _trace, ans = gen_problem(rng)
        if q in seen:
            continue
        seen.add(q)
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": q}],
            "answer": ans,
        })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(tok, args):
    cfg = LatentMoEConfig(
        vocab_size=len(tok),
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        intermediate_size=args.intermediate_size,
        shared_intermediate_size=args.shared_intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_experts=args.experts,
        num_experts_per_tok=args.top_k,
        max_position_embeddings=args.max_prompt_length + args.max_completion_length + 8,
    )
    torch.manual_seed(args.seed)
    model = LatentMoEForCausalLM(cfg)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.bos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    return model


def freeze_router(model):
    for n, p in model.named_parameters():
        if ".gate.weight" in n:
            p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Eval: greedy-generate, score correctness + format rate
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, tok, device, n=24, seed=999, max_new=80):
    model.eval()
    rng = random.Random(seed)
    correct = fmt = 0
    for _ in range(n):
        q, _t, gold = gen_problem(rng)
        text = chat_prompt(tok, q)
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        comp = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(comp)
        if pred is not None and pred.lstrip("-").isdigit() and int(pred) == int(gold):
            correct += 1
        if "<think>" in comp and "</think>" in comp:
            fmt += 1
    return correct / n, fmt / n


# ---------------------------------------------------------------------------
# STAGE 1 — SFT warm-up
# ---------------------------------------------------------------------------
def make_sft_batch(tok, rng, bs, max_len, device):
    input_ids, labels = [], []
    for _ in range(bs):
        q, trace, ans = gen_problem(rng)
        prompt = chat_prompt(tok, q)
        resp = f"<think>\n{trace}\n</think>\n{ans}"
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        r_ids = tok(resp, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (p_ids + r_ids)[:max_len]
        lab = ([-100] * len(p_ids) + r_ids)[:max_len]
        pad = max_len - len(ids)
        input_ids.append(ids + [tok.pad_token_id] * pad)
        labels.append(lab + [-100] * pad)
    return (torch.tensor(input_ids, device=device),
            torch.tensor(labels, device=device))


def make_generate_fn(model, tok, device, max_new=64):
    """Greedy generate_fn(messages)->completion_text, for reward profiling/eval."""
    @torch.no_grad()
    def gen(messages):
        model.eval()
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return gen


def make_sft_batch_multi(tok, rng, bs, max_len, device):
    """SFT batch drawn from ALL verifiable environments (multi-domain prep)."""
    exs = ENV.build_sft_examples(rng, bs)
    input_ids, labels = [], []
    for messages, resp in exs:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        r_ids = tok(resp, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (p_ids + r_ids)[:max_len]
        lab = ([-100] * len(p_ids) + r_ids)[:max_len]
        pad = max_len - len(ids)
        input_ids.append(ids + [tok.pad_token_id] * pad)
        labels.append(lab + [-100] * pad)
    return (torch.tensor(input_ids, device=device),
            torch.tensor(labels, device=device))


def eval_multi(model, tok, device, n_per_env=12):
    """Per-environment pass-rates for the multi-domain replica."""
    gen = make_generate_fn(model, tok, device)
    return ENV.profile_rewards(gen, n_per_env=n_per_env)


def stage_sft(model, tok, args, device):
    print("\n########## STAGE 1: SFT WARM-UP (LatentMoE, router frozen) ##########")
    freeze_router(model)
    model.to(device)
    rng = random.Random(args.seed)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.sft_lr)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name="nemotron-mini-sft",
                         config={"stage": "sft", "steps": args.sft_steps,
                                 "hidden": args.hidden_size, "experts": args.experts},
                         reinit=True)

    sft_batch_fn = make_sft_batch_multi if args.multi_env else make_sft_batch
    if args.multi_env:
        print("[sft] pre-warmup per-env:", {k: round(v, 2) for k, v in eval_multi(model, tok, device).items()})
    else:
        acc0, fmt0 = evaluate(model, tok, device, n=24)
        print(f"[sft] pre-warmup eval: acc={acc0:.2f} format={fmt0:.2f}")
    model.train()
    max_len = args.max_prompt_length + args.max_completion_length
    t0 = time.time()
    for step in range(1, args.sft_steps + 1):
        ids, labels = sft_batch_fn(tok, rng, args.sft_batch, max_len, device)
        loss = model(input_ids=ids, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if run:
            run.log({"sft/loss": float(loss.detach())}, step=step)
        if step == 1 or step % args.sft_eval_every == 0 or step == args.sft_steps:
            model.train()
            if args.multi_env:
                rates = eval_multi(model, tok, device)
                mean = sum(rates.values()) / len(rates)
                print(f"[sft] step {step:04d}/{args.sft_steps} loss={float(loss):.3f} "
                      f"mean_pass={mean:.2f} " + " ".join(f"{k}={v:.2f}" for k, v in rates.items()))
                if run:
                    run.log({"sft/mean_pass": mean,
                             **{f"sft/pass_{k}": v for k, v in rates.items()}}, step=step)
            else:
                acc, fmt = evaluate(model, tok, device, n=24)
                print(f"[sft] step {step:04d}/{args.sft_steps} loss={float(loss):.3f} "
                      f"acc={acc:.2f} format={fmt:.2f}")
                if run:
                    run.log({"sft/eval_accuracy": acc, "sft/eval_format": fmt}, step=step)

    if args.multi_env:
        rates = eval_multi(model, tok, device, n_per_env=20)
        acc1 = sum(rates.values()) / len(rates)
        print(f"[sft] post-warmup per-env: " + " ".join(f"{k}={v:.2f}" for k, v in rates.items())
              + f"  mean={acc1:.2f}  ({time.time()-t0:.0f}s)")
        if run:
            run.summary.update({"final_mean_pass": acc1, **{f"final_pass_{k}": v for k, v in rates.items()}})
            run.finish()
        fmt1 = None
    else:
        acc1, fmt1 = evaluate(model, tok, device, n=48)
        print(f"[sft] post-warmup eval: acc={acc1:.2f} format={fmt1:.2f}  "
              f"({time.time()-t0:.0f}s)")
        if run:
            run.summary.update({"final_acc": acc1, "final_format": fmt1})
            run.finish()
    os.makedirs(os.path.dirname(SFT_CKPT), exist_ok=True)
    torch.save(model.state_dict(), SFT_CKPT)
    print(f"[sft] warm-up checkpoint -> {SFT_CKPT}")
    return acc1, fmt1


# ---------------------------------------------------------------------------
# STAGE 2 — GRPO RL with masked importance sampling
# ---------------------------------------------------------------------------
def stage_rl(model, tok, args, device):
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig, get_peft_model

    print("\n########## STAGE 2: GRPO RL (masked importance sampling) ##########")
    # MoE recipe: experts + router frozen, LoRA on attention only.
    freeze_router(model)
    for n, p in model.named_parameters():
        if ".experts." in n:
            p.requires_grad_(False)
    # LoRA scope: attention always; optionally also the expert FFNs + the
    # LatentMoE shared projections so RL can reshape the experts (more plasticity).
    targets = list(ATTN)
    if args.train_experts:
        targets += ["gate_proj", "up_proj", "down_proj", "latent_down", "latent_up"]
        print("[rl] train_experts=ON -> LoRA on experts + latent projections too")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=targets, task_type="CAUSAL_LM", bias="none"))
    confirm_router_frozen(model)
    print()
    probe_before = probe_expert_routing(model, tok)

    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # --- Build the RL data + reward functions ---
    if args.multi_env:
        # Reward profiling -> Gaussian difficulty curriculum (Ultra's data mixture).
        gen = make_generate_fn(model, tok, device)
        rates = ENV.profile_rewards(gen, n_per_env=12)
        weights = ENV.gaussian_curriculum(rates)
        print("[rl] reward profiling (pass-rate):",
              " ".join(f"{k}={v:.2f}" for k, v in rates.items()))
        print("[rl] gaussian curriculum weights:",
              " ".join(f"{k}={v:.2f}" for k, v in weights.items()))
        dataset = ENV.build_unified_dataset(args.n_problems, args.seed, weights=weights)
        reward_funcs = ENV.make_reward_funcs()
        acc_pre = sum(rates.values()) / len(rates)
        reward_label = f"{len(ENV.ENVS)} verifiable envs + format"
    else:
        dataset = build_rl_dataset(args.n_problems, args.seed)
        reward_funcs = REWARD_FUNCS
        acc_pre, _ = evaluate(model, tok, device, n=48)
        reward_label = "correctness + format"
    print(f"[rl] pre-RL competence (mean): {acc_pre:.2f}")

    cfg = GRPOConfig(
        output_dir="outputs/nemotron_mini/rl",
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.rl_steps,
        learning_rate=args.rl_lr,
        temperature=1.0,
        # --- Nemotron-3 fingerprint: masked importance sampling ---
        importance_sampling_level="token",   # token-level masked IS
        num_iterations=args.num_iterations,   # >1 => off-policy => IS correction is live
        epsilon=0.2, epsilon_high=0.28,       # asymmetric clipping (DAPO-style)
        mask_truncated_completions=True,
        beta=args.beta,                       # 0 => no KL ref model (lean local run)
        scale_rewards="group",
        loss_type="dr_grpo",
        logging_steps=1, save_steps=10_000,
        use_vllm=False, report_to="wandb" if args.wandb else "none",
        run_name="nemotron-mini-rl", gradient_checkpointing=False, seed=args.seed,
    )
    trainer = GRPOTrainer(model=model, processing_class=tok,
                          reward_funcs=reward_funcs, args=cfg, train_dataset=dataset)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"[rl] GRPO: num_generations={args.num_generations} "
          f"importance_sampling=token num_iterations={args.num_iterations} "
          f"rewards=[{reward_label}] steps={args.rl_steps}")
    trainer.train()

    if args.multi_env:
        rates_post = eval_multi(model, tok, device, n_per_env=20)
        acc_post = sum(rates_post.values()) / len(rates_post)
    else:
        acc_post, _ = evaluate(model, tok, device, n=48)
    probe_after = probe_expert_routing(model, tok)
    print("\n########## NEMOTRON-MINI RL RESULTS ##########")
    print(f"competence (mean) pre-RL {acc_pre:.2f} -> post-RL {acc_post:.2f}")
    if args.multi_env:
        print("[rl] post-RL per-env: " + " ".join(f"{k}={v:.2f}" for k, v in rates_post.items()))
    print(f"routing distinct experts {probe_before['distinct_experts']} -> "
          f"{probe_after['distinct_experts']}")
    if device == "cuda":
        print(f"peak VRAM allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print("Reward / KL / clip-ratio curves logged to wandb project "
          f"'{args.wandb_project}' (run nemotron-mini-rl).")
    # Save the merged RLVR student so MOPD (Phase 3) distills from the POST-RLVR
    # model, preserving the SFT->RLVR->MOPD ordering.
    try:
        rlvr_ckpt = "outputs/nemotron_mini/rlvr.pt"
        merged = model.merge_and_unload()
        torch.save(merged.state_dict(), rlvr_ckpt)
        print(f"[rl] saved merged RLVR checkpoint -> {rlvr_ckpt}")
    except Exception as e:
        print(f"[rl] could not save merged RLVR checkpoint ({e}); MOPD can use sft.pt")
    print("########## DONE ##########")


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "sft", "rl"], default="all")
    ap.add_argument("--resume", default=None, help="SFT checkpoint to load for --stage rl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--multi_env", action="store_true",
                    help="Unified RLVR over ALL verifiable environments (Ultra-style) "
                         "with reward profiling + Gaussian curriculum. Off = math only.")
    ap.add_argument("--train_experts", action="store_true",
                    help="Also LoRA the experts + latent projections during RL "
                         "(more plasticity; router stays frozen).")
    # model (bigger transformer than the smoke runs so warm-up can actually learn)
    ap.add_argument("--hidden_size", type=int, default=384)
    ap.add_argument("--latent_size", type=int, default=96)       # d/l = 4x
    ap.add_argument("--intermediate_size", type=int, default=256)
    ap.add_argument("--shared_intermediate_size", type=int, default=768)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    # SFT
    ap.add_argument("--sft_steps", type=int, default=400)
    ap.add_argument("--sft_batch", type=int, default=32)
    ap.add_argument("--sft_lr", type=float, default=3e-4)
    ap.add_argument("--sft_eval_every", type=int, default=50)
    # RL
    ap.add_argument("--rl_steps", type=int, default=60)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--num_iterations", type=int, default=2)
    ap.add_argument("--rl_lr", type=float, default=1e-5)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.03,
                    help="KL coefficient anchoring the policy to the SFT model "
                         "(prevents drift when train_experts is on). With LoRA the "
                         "frozen base IS the reference, so this is cheap. 0 disables.")
    ap.add_argument("--n_problems", type=int, default=64)
    ap.add_argument("--max_prompt_length", type=int, default=96)
    ap.add_argument("--max_completion_length", type=int, default=96)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="latent-moe-mini")
    return ap.parse_args()


def main():
    args = parse_args()
    device = (args.device if args.device != "auto"
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"\n########## NEMOTRON-MINI RL POST-TRAINING (LatentMoE @ 3090) ##########")
    print(f"[tok] {args.tokenizer} vocab={len(tok)} device={device}")

    model = build_model(tok, args)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] LatentMoE {total/1e6:.1f}M params d={args.hidden_size} "
          f"l={args.latent_size} (d/l={args.hidden_size//args.latent_size}x) "
          f"{args.experts} experts top-{args.top_k}")

    if args.stage in ("all", "sft"):
        stage_sft(model, tok, args, device)
    if args.stage == "rl":
        ckpt = args.resume or SFT_CKPT
        print(f"[rl] loading warm-up checkpoint: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
    if args.stage in ("all", "rl"):
        stage_rl(model, tok, args, device)


if __name__ == "__main__":
    main()
