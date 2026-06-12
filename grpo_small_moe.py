"""
STAGE 1 - GRPO + MoE smoke test on a SMALL transformer MoE.

Purpose: validate the GRPO-meets-MoE plumbing cheaply, with comfortable 24GB
headroom, before risking the tight Qwen3-30B-A3B run. Same reward/dataset and
router-freeze handling as Stage 2.

Model:  Qwen/Qwen1.5-MoE-A2.7B-Chat  (14.3B total / 2.7B active, 60 experts +
        shared expert, 4 active/token) loaded as 4-bit QLoRA. Per the MoE gotcha
        we load the 16-bit repo with load_in_4bit=True (on-the-fly convert)
        rather than a pre-quantized 4-bit BnB MoE repo (those are broken to
        import).

        NB: We originally tried allenai/OLMoE-1B-7B-0924-Instruct but vLLM
        0.11.0 raises "OlmoeForCausalLM does not support LoRA yet" — vLLM has no
        LoRA path for OLMoE, so it can't serve GRPO rollouts of a LoRA policy.
        Qwen2MoeForCausalLM (this model) and Qwen3MoeForCausalLM (the Stage-2
        target) both declare SupportsLoRA in vLLM, so this is also the closest
        small architecture to the real 30B target.

Run:   python grpo_small_moe.py
"""

import os
import sys

# Must be set BEFORE importing unsloth: enables vLLM "standby" mode so the vLLM
# KV-cache memory is released back during the optimizer step (key for fitting
# colocated generation + training on one GPU). Skip it on the --no_vllm path:
# standby pre-reserves a GPU memory pool that is wasted (and harmful on tight
# models) when generation isn't using vLLM.
if "--no_vllm" not in sys.argv:
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import argparse
import torch
import unsloth  # noqa: F401  (import first so it can patch TRL/transformers)
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

from reward_dataset import REWARD_FUNCS, build_dataset
from moe_utils import confirm_router_frozen, probe_expert_routing


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=640)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--n_problems", type=int, default=50)
    ap.add_argument("--lora_scope", choices=["attn", "attn_mlp"], default="attn",
                    help="attn = LoRA on attention only (REQUIRED for MoE + vLLM "
                         "rollouts: vLLM has no fused-MoE LoRA kernel, so adapting "
                         "the expert FFNs crashes the bnb fused-MoE path). "
                         "attn_mlp also targets expert FFNs (ok for HF generation).")
    ap.add_argument("--no_vllm", action="store_true",
                    help="Use HF generation instead of vLLM. REQUIRED on this stack: "
                         "vLLM 0.11.0's bitsandbytes 4-bit fused-MoE Triton kernel "
                         "hits an illegal memory access at engine init on Ampere "
                         "(no tuned MoE config for sm_86). HF generation bypasses it "
                         "but forgoes vLLM standby-memory behaviour.")
    return ap.parse_args()


# MoE-specific: vLLM 0.11.0 cannot serve LoRA on the fused MoE expert weights
# ("does not support fused MoE LoRA inference"). Targeting gate_proj/up_proj/
# down_proj (the experts) crashes vLLM's bitsandbytes fused-MoE Triton kernel
# during profiling. So for vLLM-backed GRPO on an MoE we LoRA attention only;
# the experts and router stay frozen (experts still *route & run*, just aren't
# adapted). This is the key recipe difference vs a dense model.
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTN_MLP = ATTN + ["gate_proj", "up_proj", "down_proj"]


class RewardCurveCB(TrainerCallback):
    """Collect the mean reward from each logging step to show the trend."""
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "reward" in logs:
            self.history.append((state.global_step, logs["reward"]))


def main():
    args = parse_args()
    max_prompt_length = args.max_seq_length - args.max_completion_length
    assert max_prompt_length > 0

    print(f"\n########## STAGE 1: GRPO+MoE smoke test on {args.model} ##########\n")

    # ---- load 4-bit QLoRA (+ vLLM fast generation unless --no_vllm) -------
    load_kwargs = dict(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,           # on-the-fly 16bit->4bit (MoE-safe path)
        max_lora_rank=args.lora_rank,
    )
    if not args.no_vllm:
        load_kwargs.update(fast_inference=True,            # vLLM rollouts
                           gpu_memory_utilization=args.gpu_mem_util)
    else:
        print("[gen] HF generation path (vLLM disabled): bypasses the broken "
              "vLLM bnb fused-MoE kernel; standby-memory behaviour NOT exercised.")
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    # ---- LoRA on attention (+ expert MLP projections if attn_mlp); NOT router
    target_modules = ATTN if args.lora_scope == "attn" else ATTN_MLP
    print(f"[lora] scope={args.lora_scope} target_modules={target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=target_modules,
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ---- MoE-specific assertions: router frozen, routing live ------------
    confirm_router_frozen(model)
    print()
    FastLanguageModel.for_inference(model)
    probe_before = probe_expert_routing(model, tokenizer)
    FastLanguageModel.for_training(model)
    print()

    # ---- dataset + GRPO config -------------------------------------------
    dataset = build_dataset(n=args.n_problems)

    cfg_kwargs = dict(
        output_dir="outputs/grpo_small_moe",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_generations,  # -> gen_batch divisible by G
        num_generations=args.num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        temperature=1.0,
        logging_steps=1,
        save_steps=10_000,            # effectively don't checkpoint in a smoke test
        report_to="none",
        gradient_checkpointing=False,  # handled by unsloth in get_peft_model
    )
    if not args.no_vllm:
        # vLLM colocated generation (single GPU, shares weights with training)
        cfg_kwargs.update(use_vllm=True, vllm_mode="colocate",
                          vllm_gpu_memory_utilization=args.gpu_mem_util)
    else:
        cfg_kwargs.update(use_vllm=False)
    config = GRPOConfig(**cfg_kwargs)

    reward_cb = RewardCurveCB()
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=config,
        train_dataset=dataset,
    )
    trainer.add_callback(reward_cb)

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    # ---- report ----------------------------------------------------------
    print("\n########## STAGE 1 RESULTS ##########")
    print(f"peak VRAM allocated: {peak_gb:.2f} GB / 24 GB")
    if reward_cb.history:
        steps = [s for s, _ in reward_cb.history]
        vals = [r for _, r in reward_cb.history]
        first = sum(vals[:5]) / len(vals[:5])
        last = sum(vals[-5:]) / len(vals[-5:])
        print(f"reward curve: {len(vals)} points; "
              f"first-5 mean {first:.3f} -> last-5 mean {last:.3f} "
              f"({'UP' if last > first else 'flat/down'})")
        print("reward trace:", [f"{v:.2f}" for v in vals])

    # post-training routing probe (still diverse => no expert collapse)
    print()
    FastLanguageModel.for_inference(model)
    probe_after = probe_expert_routing(model, tokenizer)

    # sample generations
    print("\n--- sample generations ---")
    for q in ["What is 23 + 47?", "What is 9 * 8?"]:
        msgs = [{"role": "user", "content": q}]
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=256, temperature=0.7,
                             do_sample=True)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"\nQ: {q}\nA: {text.strip()[:400]}")

    print("\nrouting before/after:", probe_before["distinct_experts"], "->",
          probe_after["distinct_experts"], "distinct experts")
    print("########## STAGE 1 DONE ##########")


if __name__ == "__main__":
    main()
