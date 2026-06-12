"""
STAGE 2 - GRPO + MoE scaled up to the REAL target: Qwen3-30B-A3B.

Same GRPO recipe and shared reward/dataset as Stage 1, swapped onto
unsloth/Qwen3-30B-A3B (30B total / 3B active) as 4-bit QLoRA. This is the
same *shape* as NVIDIA's Nemotron Nano 30B-A3B, so it exercises the MoE GRPO
mechanics (router freeze, expert routing under policy-gradient updates, vLLM
standby memory behaviour) that transfer to the Nemotron RL recipe.

This is TIGHT on 24GB. QLoRA of the 30B is ~17.5GB on its own and vLLM rollout
KV-cache stacks on top. We start conservative and, on OOM, step DOWN
systematically (context, then num_generations) and report the smallest config
that runs — or the exact memory wall.

MoE gotcha: Qwen3-30B-A3B QLoRA needs the full 16-bit weights (~60GB) converted
to 4-bit on the fly, because importing pre-quantized 4-bit BnB MoE repos is
broken. This needs ~70GB free disk and meaningful RAM for the conversion --
see the preflight gate below, which aborts early if resources are short.

Run:   python grpo_qwen3_30b.py                 # conservative default
       python grpo_qwen3_30b.py --max_seq_length 768 --num_generations 4
       python grpo_qwen3_30b.py --auto_stepdown  # try configs until one fits
"""

import os

import sys as _sys
# Standby mode pre-reserves a GPU memory pool (via torch_memory_saver) to hold
# the vLLM KV-cache across the optimizer step. That reservation is pointless on
# the HF-generation (--no_vllm) path and, on the tight 30B, it pushes the 4-bit
# load over 24GB -> OOM. So only enable standby when actually using vLLM.
if "--no_vllm" not in _sys.argv:
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1"  # release vLLM KV-cache during optim step

# IMPORTANT: import unsloth FIRST, before transformers/torch get imported, so it
# can patch them. Importing it lazily after `from transformers import ...` leaves
# Unsloth's optimized 4-bit loading disabled and OOMs the 30B load on 24GB.
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import argparse
import shutil
import sys

import torch

WORKDIR = os.path.dirname(os.path.abspath(__file__))
MOE_DISK_REQUIRED_GB = 70


def preflight_or_abort(local_model=False):
    """Abort BEFORE downloading 60GB if disk/RAM are clearly insufficient.

    `local_model` True means --model is an already-present local directory (e.g.
    a pre-quantized 4-bit snapshot): no 60GB download happens, so the disk gate
    is irrelevant and skipped.
    """
    du = shutil.disk_usage(WORKDIR)
    disk_free_gb = du.free / 1024**3
    if local_model:
        print(f"[preflight] local model dir given -> skipping 16-bit download "
              f"disk gate (disk free: {disk_free_gb:.0f}GB).")
    else:
        print(f"[preflight] disk free: {disk_free_gb:.0f}GB (need ~{MOE_DISK_REQUIRED_GB}GB)")
        if disk_free_gb < MOE_DISK_REQUIRED_GB:
            sys.exit(f"[preflight] ABORT: only {disk_free_gb:.0f}GB free, need "
                     f"~{MOE_DISK_REQUIRED_GB}GB for the 16-bit download + 4-bit convert.")
    try:
        import psutil
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        ram = (vm.available + sm.free) / 1024**3
        print(f"[preflight] RAM+swap available: ~{ram:.0f}GB")
        if ram < 16:
            sys.exit(f"[preflight] ABORT: only ~{ram:.0f}GB RAM+swap; the on-the-fly "
                     "4-bit MoE conversion will likely OOM the host. Close apps "
                     "or add swap, then retry.")
        if ram < 28:
            print("[preflight] WARN: low RAM headroom for the 60GB->4bit convert; "
                  "expect heavy swapping. Close other apps if it stalls.")
    except Exception as e:
        print("[preflight] psutil check skipped:", e)
    print("[preflight] OK -> proceeding to load Qwen3-30B-A3B.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-30B-A3B")
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_completion_length", type=int, default=640)
    ap.add_argument("--max_steps", type=int, default=25)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--gpu_mem_util", type=float, default=0.9)
    ap.add_argument("--n_problems", type=int, default=50)
    ap.add_argument("--lora_scope", choices=["attn", "attn_mlp"], default="attn",
                    help="attn = LoRA attention only (REQUIRED for MoE + vLLM: no "
                         "fused-MoE LoRA kernel). attn_mlp also LoRAs expert FFNs "
                         "(crashes vLLM's bnb fused-MoE path).")
    ap.add_argument("--auto_stepdown", action="store_true",
                    help="On CUDA OOM, retry progressively smaller configs.")
    ap.add_argument("--no_vllm", action="store_true",
                    help="Use HF generation instead of vLLM (REQUIRED on this stack: "
                         "vLLM 0.11.0 bnb 4-bit fused-MoE kernel crashes at init on "
                         "Ampere). Forgoes vLLM standby-memory behaviour.")
    return ap.parse_args()


# See grpo_small_moe.py: vLLM cannot serve LoRA on fused MoE experts, so the
# vLLM-backed GRPO recipe LoRAs attention only; experts + router stay frozen.
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTN_MLP = ATTN + ["gate_proj", "up_proj", "down_proj"]


# (max_seq_length, max_completion_length, num_generations) ladder, large->small
STEPDOWN_LADDER = [
    (1024, 640, 4),
    (768, 512, 4),
    (512, 384, 4),
    (512, 384, 2),
]


from transformers import TrainerCallback


class RewardCurveCB(TrainerCallback):
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "reward" in logs:
            self.history.append((state.global_step, logs["reward"]))


def run_once(args, max_seq_length, max_completion_length, num_generations):
    """One full attempt. Returns peak VRAM (GB) on success, raises on OOM."""
    from trl import GRPOConfig, GRPOTrainer
    from reward_dataset import REWARD_FUNCS, build_dataset
    from moe_utils import confirm_router_frozen, probe_expert_routing

    max_prompt_length = max_seq_length - max_completion_length
    assert max_prompt_length > 0
    print(f"\n>>> attempt: seq={max_seq_length} compl={max_completion_length} "
          f"G={num_generations}")

    load_kwargs = dict(
        model_name=args.model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,            # on-the-fly 16bit->4bit (MoE-safe path)
        max_lora_rank=args.lora_rank,
    )
    if not args.no_vllm:
        load_kwargs.update(fast_inference=True,           # vLLM rollouts
                           gpu_memory_utilization=args.gpu_mem_util)
    else:
        print("[gen] HF generation path (vLLM disabled): bypasses the broken "
              "vLLM bnb fused-MoE kernel; standby-memory behaviour NOT exercised.")
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

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

    confirm_router_frozen(model)
    print()
    FastLanguageModel.for_inference(model)
    probe_before = probe_expert_routing(model, tokenizer)
    FastLanguageModel.for_training(model)

    dataset = build_dataset(n=args.n_problems)
    cfg_kwargs = dict(
        output_dir="outputs/grpo_qwen3_30b",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=num_generations,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=args.max_steps,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        temperature=1.0,
        logging_steps=1,
        save_steps=10_000,
        report_to="none",
        gradient_checkpointing=False,
    )
    if not args.no_vllm:
        cfg_kwargs.update(use_vllm=True, vllm_mode="colocate",
                          vllm_gpu_memory_utilization=args.gpu_mem_util)
    else:
        cfg_kwargs.update(use_vllm=False)
    config = GRPOConfig(**cfg_kwargs)

    reward_cb = RewardCurveCB()
    import time
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=config,
        train_dataset=dataset,
    )
    trainer.add_callback(reward_cb)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    print("\n########## STAGE 2 RESULTS ##########")
    print(f"FIT config: seq={max_seq_length} compl={max_completion_length} "
          f"G={num_generations}")
    print(f"peak VRAM allocated: {peak_gb:.2f} GB / 24 GB")
    print(f"steps/sec: {args.max_steps / dt:.3f}  ({args.max_steps} steps in {dt:.0f}s)")
    if reward_cb.history:
        vals = [r for _, r in reward_cb.history]
        first = sum(vals[:5]) / len(vals[:5])
        last = sum(vals[-5:]) / len(vals[-5:])
        print(f"reward: first-5 {first:.3f} -> last-5 {last:.3f} "
              f"({'UP' if last > first else 'flat/down'})")
        print("reward trace:", [f"{v:.2f}" for v in vals])
    FastLanguageModel.for_inference(model)
    probe_after = probe_expert_routing(model, tokenizer)
    print("routing distinct experts before/after:",
          probe_before["distinct_experts"], "->", probe_after["distinct_experts"])
    return peak_gb


def main():
    args = parse_args()
    print("\n########## STAGE 2: GRPO on Qwen3-30B-A3B (the real target) ##########")
    preflight_or_abort(local_model=os.path.isdir(args.model))

    configs = STEPDOWN_LADDER if args.auto_stepdown else [
        (args.max_seq_length, args.max_completion_length, args.num_generations)]

    for i, (seq, compl, g) in enumerate(configs):
        try:
            run_once(args, seq, compl, g)
            print(f"\nSUCCESS on config #{i}: seq={seq} compl={compl} G={g}")
            return 0
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            print(f"\n[OOM] config seq={seq} compl={compl} G={g} hit the memory wall: "
                  f"{str(e)[:160]}")
            if not args.auto_stepdown:
                print("Re-run with --auto_stepdown to search smaller configs, or "
                      "reduce --max_seq_length / --num_generations manually.")
                return 2
            print("Stepping down...\n")
    print("\nAll stepdown configs OOM'd. This 30B GRPO run needs more VRAM than the "
          "RTX 3090's 24GB -> a rented A100 (40/80GB) is required.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
