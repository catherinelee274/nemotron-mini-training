"""
H100 FAITHFUL PATH — train per-domain SPECIALIST teachers for MOPD.

Our 3090 experiments proved a GENERAL teacher (Qwen2.5-1.5B) is a BAD MOPD target:
its KL corrupts the verifiable-format envs (json/count/keyword -> 0). Ultra avoids
this by distilling from >10 DOMAIN-SPECIALIZED teachers. This script manufactures
that: it fine-tunes a small base (Qwen2.5-0.5B-Instruct) on EACH environment's ideal
responses, producing one specialist per domain that emits the exact verifiable format.
MOPD then routes each rollout to its env's specialist (mopd.py --teacher_map), so the
teacher KL HELPS instead of hurting.

Lives in h100_faithful/ to keep the rentable H100 run separate from the local 3090
scripts. Reuses the parent repo's rl_environments for data.

Run (on the rented H100):
  python train_specialist_teachers.py --base Qwen/Qwen2.5-0.5B-Instruct \
         --steps 600 --out teachers
Outputs one merged HF model per env under  h100_faithful/teachers/<env>/  and prints
the --teacher_map string to paste into the MOPD launch.
"""

import argparse
import os
import random
import sys

# import the parent repo's environment suite
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import rl_environments as ENV


def sft_batch(tok, env, rng, bs, max_len, device):
    """(prompt -> ideal_response) supervised batch for ONE environment."""
    input_ids, labels = [], []
    for _ in range(bs):
        q, _spec, resp = ENV.ENVS[env](rng)
        prompt = tok.apply_chat_template(ENV._messages(q, False),
                                         add_generation_prompt=True, tokenize=False)
        p = tok(prompt, add_special_tokens=False).input_ids
        r = tok(resp, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (p + r)[:max_len]
        lab = ([-100] * len(p) + r)[:max_len]
        pad = max_len - len(ids)
        input_ids.append(ids + [tok.pad_token_id] * pad)
        labels.append(lab + [-100] * pad)
    return (torch.tensor(input_ids, device=device),
            torch.tensor(labels, device=device))


def train_one(env, base, args, device):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    rng = random.Random(args.seed)
    for step in range(1, args.steps + 1):
        ids, labels = sft_batch(tok, env, rng, args.batch, args.max_len, device)
        loss = model(input_ids=ids, labels=labels).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"  [{env}] step {step:04d}/{args.steps} loss={float(loss):.3f}")
    out_dir = os.path.join(args.out, env)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"  [{env}] saved specialist -> {out_dir}")
    del model, merged
    if device == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--envs", nargs="*", default=None,
                    help="Subset of envs (default: all in rl_environments).")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "teachers"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if (torch.cuda.is_available() and args.device != "cpu") else "cpu"
    envs = args.envs or list(ENV.ENVS)
    os.makedirs(args.out, exist_ok=True)
    print(f"Training {len(envs)} specialist teachers from {args.base} on {device}")
    mapping = []
    for env in envs:
        print(f"\n=== specialist: {env} ===")
        path = train_one(env, args.base, args, device)
        mapping.append(f"{env}={path}")
    print("\n########## DONE ##########")
    print("Paste this into the MOPD launch:")
    print("  --teacher_map " + " ".join(mapping))


if __name__ == "__main__":
    main()
