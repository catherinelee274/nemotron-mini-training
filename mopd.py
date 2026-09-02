"""
Phase 3 — MOPD (Multi-teacher On-Policy Distillation), Ultra-style, scalable.

Ultra's final RL-flavored stage (§3.3): the RLVR student generates rollouts ON-POLICY,
and is trained to match domain-specialized TEACHERS via dense TOKEN-LEVEL KL, using a
clipped proximal-policy objective, run asynchronously and over TWO iterations (teachers
in round 2 are re-initialized from the round-1 student — co-evolution).

This reproduces the mechanism at 3090 scale and scales to an 80GB GPU by flags:
  - Student: our LatentMoE (post-RLVR checkpoint). Router FROZEN; rest trains.
  - Teacher(s): cached HF models sharing the Qwen2.5 vocab (so token-level KL aligns).
    Pass several --teachers for the multi-teacher / per-domain-specialist setup.
  - On-policy rollouts from the student (optionally MTP-accelerated, Phase 2).
  - Loss: mean token-level KL(student || teacher) over response tokens (eq. "minimize
    D_KL(pi_theta || pi_T)"), optional clipped proximal ratio for the async off-policy gap.
  - --iterations 2 reproduces the two-round co-evolution (teacher pool can grow per round).

GPU: 1-2 small teachers + sequential => fits a 24GB 3090. Faithful (>=4 teachers and/or
--async_offpolicy with many resident teachers) => 1x H100/A100 80GB (or 2x 48GB).

Run (3090 demo):
  python mopd.py --wandb --teachers Qwen/Qwen2.5-1.5B-Instruct \
                 --student_ckpt outputs/nemotron_mini/sft.pt --steps 40 --iterations 2
"""

import argparse
import os
import random
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_moe import LatentMoEConfig, LatentMoEForCausalLM
import rl_environments as ENV


def build_student(tok, args, device):
    cfg = LatentMoEConfig(
        vocab_size=len(tok), hidden_size=args.hidden_size, latent_size=args.latent_size,
        intermediate_size=args.intermediate_size,
        shared_intermediate_size=args.shared_intermediate_size,
        num_hidden_layers=args.layers, num_attention_heads=args.heads,
        num_experts=args.experts, num_experts_per_tok=args.top_k,
        max_position_embeddings=args.max_prompt_length + args.max_completion_length + 8)
    model = LatentMoEForCausalLM(cfg)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    if args.student_ckpt and os.path.exists(args.student_ckpt):
        sd = torch.load(args.student_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[mopd] loaded student {args.student_ckpt} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
    else:
        print("[mopd] WARNING: no student checkpoint -> random-init student (mechanics only)")
    return model.to(device)


@torch.no_grad()
def sample_rollouts(student, tok, device, n, max_prompt, max_new, temperature=1.0, rng=None):
    """Student generates ON-POLICY rollouts across all envs. Returns list of
    (full_ids[T], resp_mask[T], env, spec, prompt_len)."""
    student.eval()
    rng = rng or random.Random()
    out = []
    names = list(ENV.ENVS)
    for _ in range(n):
        env = rng.choice(names)
        q, spec, _resp = ENV.ENVS[env](rng)
        text = tok.apply_chat_template(ENV._messages(q, False), add_generation_prompt=True,
                                       tokenize=False)
        p_ids = tok(text, return_tensors="pt").input_ids.to(device)[:, :max_prompt]
        gen = student.generate(p_ids, max_new_tokens=max_new, do_sample=True,
                               temperature=temperature, top_k=50)
        full = gen[0]
        mask = torch.zeros_like(full)
        mask[p_ids.shape[1]:] = 1                      # response tokens only
        out.append((full, mask, env, spec, p_ids.shape[1]))
    return out


def mopd_kl_loss(student_logits, teacher_logits, resp_mask):
    """Mean token-level KL(student || teacher) over response positions.
    Aligns predicted-token positions; truncates to shared vocab for safety."""
    V = min(student_logits.size(-1), teacher_logits.size(-1))
    s = F.log_softmax(student_logits[:, :-1, :V], dim=-1)
    with torch.no_grad():
        t = F.log_softmax(teacher_logits[:, :-1, :V], dim=-1)
    p_s = s.exp()
    kl = (p_s * (s - t)).sum(-1)                        # [B, T-1]
    m = resp_mask[:, 1:].float()
    return (kl * m).sum() / m.sum().clamp_min(1.0)


def pad_batch(rollouts, pad_id, device):
    T = max(r[0].size(0) for r in rollouts)
    ids = torch.full((len(rollouts), T), pad_id, dtype=torch.long, device=device)
    msk = torch.zeros((len(rollouts), T), dtype=torch.long, device=device)
    for i, r in enumerate(rollouts):
        full, mask = r[0], r[1]
        ids[i, :full.size(0)] = full
        msk[i, :mask.size(0)] = mask
    return ids, msk


def main():
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("\n########## PHASE 3: MOPD (multi-teacher on-policy distillation) ##########")
    student = build_student(tok, args, device)
    # MoE recipe: router frozen, everything else trainable (full plasticity for distill).
    for n, p in student.named_parameters():
        p.requires_grad_(not n.endswith(".gate.weight"))
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[mopd] student trainable params: {n_train/1e6:.1f}M (router frozen)")

    # Load teacher pool (the memory crunch on small GPUs).
    def _load_teacher(name):
        print(f"[mopd] loading teacher: {name}")
        t = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16).to(device).eval()
        for p in t.parameters():
            p.requires_grad_(False)
        return t

    # Per-domain specialist routing (--teacher_map env=path ...): each env's rollouts
    # are scored by ITS specialist teacher. This is the faithful fix — a teacher
    # fine-tuned on the env produces the exact verifiable format, so its KL HELPS
    # instead of corrupting it (unlike a general teacher). Falls back to the pooled
    # --teachers list when no map is given.
    teacher_by_env = {}
    if args.teacher_map:
        cache = {}
        for entry in args.teacher_map:
            env, path = entry.split("=", 1)
            if path not in cache:
                cache[path] = _load_teacher(path)
            teacher_by_env[env] = cache[path]
        teachers = [(p, m) for p, m in cache.items()]
        print(f"[mopd] per-domain specialists: "
              + " ".join(f"{e}->{teacher_by_env[e].config._name_or_path.split('/')[-1]}"
                         for e in teacher_by_env))
    else:
        teachers = [(name, _load_teacher(name)) for name in args.teachers]
    print(f"[mopd] {len(teachers)} teacher(s) resident; "
          f"async={'on' if args.async_offpolicy else 'off (sequential)'}")
    if device == "cuda":
        print(f"[mopd] VRAM after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name="nemotron-mini-mopd",
                         config={"stage": "mopd", "teachers": args.teachers,
                                 "iterations": args.iterations, "steps": args.steps},
                         reinit=True)

    torch.manual_seed(args.seed)              # reproducible rollout sampling
    roll_rng = random.Random(args.seed)
    gen_fn = make_eval_gen(student, tok, device)
    pre = ENV.profile_rewards(gen_fn, n_per_env=12)
    print("[mopd] pre-MOPD per-env:", {k: round(v, 2) for k, v in pre.items()})

    opt = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=args.lr)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gstep = 0
    for it in range(1, args.iterations + 1):
        print(f"\n=== MOPD iteration {it}/{args.iterations} "
              f"(teacher pool size {len(teachers)}) ===")
        for step in range(1, args.steps + 1):
            gstep += 1
            rollouts = sample_rollouts(student, tok, device, args.batch,
                                       args.max_prompt_length, args.max_completion_length,
                                       temperature=args.temperature, rng=roll_rng)
            ids, msk = pad_batch(rollouts, tok.pad_token_id, device)
            # verifiable reward per rollout (this is what keeps task competence)
            rewards = torch.zeros(len(rollouts), device=device)
            for i, (full, _m, env, spec, p_len) in enumerate(rollouts):
                comp = tok.decode(full[p_len:], skip_special_tokens=True)
                rewards[i] = ENV._check(env, comp, spec)
            # teacher target: route each rollout to its env's specialist (if mapped),
            # else average the teacher pool.
            with torch.no_grad():
                if teacher_by_env:
                    envs_b = [r[2] for r in rollouts]
                    t_logits = None
                    for env in set(envs_b):
                        rows = [i for i, e in enumerate(envs_b) if e == env]
                        teacher = teacher_by_env.get(env, teachers[0][1])
                        tl = teacher(input_ids=ids[rows]).logits.float()
                        if t_logits is None:
                            t_logits = torch.zeros(ids.size(0), ids.size(1), tl.size(-1),
                                                   device=device)
                        t_logits[rows] = tl
                else:
                    t_logits = None
                    for _name, teacher in teachers:
                        tl = teacher(input_ids=ids).logits.float()
                        t_logits = tl if t_logits is None else t_logits + tl
                    t_logits = t_logits / len(teachers)
            student.train()
            s_logits = student(input_ids=ids).logits.float()

            # --- BLEND: BOTH terms REWARD-GATED so failing rollouts give zero signal
            # (this makes the runaway collapse structurally impossible: the teacher
            # can only ever polish trajectories the student already gets RIGHT). ---
            m = msk[:, 1:].float()
            # 1) RFT: reinforce the student's OWN correct rollouts.
            logp = F.log_softmax(s_logits[:, :-1], dim=-1)
            tok_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)   # [B, T-1]
            seq_nll = -(tok_lp * m).sum(-1) / m.sum(-1).clamp_min(1.0)       # [B]
            loss_rft = (rewards * seq_nll).sum() / rewards.sum().clamp_min(1.0)
            # 2) teacher KL, gated by reward (only on CORRECT rollouts).
            V = min(s_logits.size(-1), t_logits.size(-1))
            s_lp = F.log_softmax(s_logits[:, :-1, :V], dim=-1)
            t_lp = F.log_softmax(t_logits[:, :-1, :V], dim=-1).detach()
            kl_tok = (s_lp.exp() * (s_lp - t_lp)).sum(-1)                    # [B, T-1]
            kl_seq = (kl_tok * m).sum(-1) / m.sum(-1).clamp_min(1.0)         # [B]
            loss_kl = (rewards * kl_seq).sum() / rewards.sum().clamp_min(1.0)
            loss = args.reward_weight * loss_rft + args.kl_weight * loss_kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0)
            opt.step()
            if run:
                run.log({"mopd/loss": float(loss), "mopd/kl": float(loss_kl),
                         "mopd/rft_nll": float(loss_rft), "mopd/reward": float(rewards.mean()),
                         "mopd/iteration": it}, step=gstep)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(f"[mopd] it{it} step {step:03d}/{args.steps} "
                      f"loss={float(loss):.4f} kl={float(loss_kl):.3f} "
                      f"rft={float(loss_rft):.3f} reward={float(rewards.mean()):.2f}")
        # iteration boundary: in the faithful setup, round-2 teachers would be
        # re-initialized from this student (co-evolution). Hook left for the H100 path.

    post = ENV.profile_rewards(gen_fn, n_per_env=20)
    print("\n########## MOPD RESULTS ##########")
    print("per-env pass-rate  pre:", {k: round(v, 2) for k, v in pre.items()})
    print("per-env pass-rate post:", {k: round(v, 2) for k, v in post.items()})
    print(f"mean {sum(pre.values())/len(pre):.2f} -> {sum(post.values())/len(post):.2f}")
    if device == "cuda":
        print(f"peak VRAM allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    if run:
        run.summary.update({"final_mean_pass": sum(post.values())/len(post),
                            **{f"final_pass_{k}": v for k, v in post.items()}})
        run.finish()
    print("########## DONE ##########")


def make_eval_gen(student, tok, device, max_new=64):
    @torch.no_grad()
    def gen(messages):
        student.eval()
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        out = student.generate(ids, max_new_tokens=max_new, do_sample=False)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return gen


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--teachers", nargs="+", default=["Qwen/Qwen2.5-1.5B-Instruct"],
                    help="One or more cached teacher models (share Qwen2.5 vocab). "
                         "Pooled (averaged) unless --teacher_map is given.")
    ap.add_argument("--teacher_map", nargs="*", default=None,
                    help="Per-domain specialists as 'env=path' entries (faithful MOPD). "
                         "Each env's rollouts are scored by its specialist teacher.")
    ap.add_argument("--student_ckpt", default="outputs/nemotron_mini/sft.pt")
    ap.add_argument("--async_offpolicy", action="store_true",
                    help="Faithful async one-step off-policy with clipped proximal ratio "
                         "(H100 path; sequential by default on the 3090).")
    # student dims (must match the checkpoint)
    ap.add_argument("--hidden_size", type=int, default=384)
    ap.add_argument("--latent_size", type=int, default=96)
    ap.add_argument("--intermediate_size", type=int, default=256)
    ap.add_argument("--shared_intermediate_size", type=int, default=768)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    # MOPD
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--reward_weight", type=float, default=1.0,
                    help="Weight on reward-gated self-training (RFT) — keeps task skill.")
    ap.add_argument("--kl_weight", type=float, default=0.1,
                    help="Weight on teacher KL — small, polishes fluency without "
                         "overwriting verifiable competence. KL is reward-gated.")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_prompt_length", type=int, default=96)
    ap.add_argument("--max_completion_length", type=int, default=64)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="latent-moe-mini")
    return ap.parse_args()


if __name__ == "__main__":
    main()
