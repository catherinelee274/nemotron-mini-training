# Faithful Ultra-RL-stage replica — single H100 (80GB)

This folder holds the **rentable-GPU** version of the Nemotron-3-Ultra RL-stage
replica, kept separate from the local 3090 scripts at the repo root. It scales the
same pipeline up to the point where it needs an 80GB GPU and adds the one piece the
3090 couldn't do faithfully: **per-domain specialist teachers** for MOPD.

## Why this is separate from the 3090 run

On the 3090 we proved the mechanisms but hit two ceilings:
1. **71M student memorizes** instead of generalizing → pass-rates capped.
2. **A general teacher corrupts MOPD** (KL erases verifiable formats) → the stable
   local config had to drop the teacher entirely (`kl_weight=0`).

This run fixes both: a **~0.5B student** (enough to generalize) and **specialist
teachers fine-tuned per environment**, so the MOPD KL term *helps* (the teacher now
emits the exact verifiable format) instead of hurting.

## What it runs (`run_faithful.sh`)

| Stage | Script | Scale |
|---|---|---|
| 0. Specialist teachers | `train_specialist_teachers.py` | 1 per env, Qwen2.5-0.5B LoRA, 600 steps each |
| 1. SFT → RLVR | `../nemotron_mini_rl.py` | ~0.5B LatentMoE, 16 rollouts, masked-IS GRPO, β=0.03 |
| 2. MOPD ×2 | `../mopd.py --teacher_map` | per-domain specialist routing, small KL |

All faithful Ultra mechanics: LatentMoE, masked importance sampling, multi-env RLVR
+ Gaussian curriculum, MTP rollout acceleration available, MOPD multi-teacher ×2.

## Launch (on the RunPod H100)

```bash
git clone <repo> && cd nemotron-mini-training
pip install -r requirements.txt          # torch, transformers, trl, peft, datasets, wandb
echo "WANDB_API_KEY=..." > .env
bash h100_faithful/run_faithful.sh        # the whole pipeline
```

Override scale via env vars, e.g. a bigger student:
```bash
LAYERS=20 HIDDEN=1024 EXPERTS=32 RL_STEPS=1200 bash h100_faithful/run_faithful.sh
```

## Cost & time on 1× H100 80GB @ $6/hr (your $263 RunPod balance)

Estimates for the default ~0.5B config on a single H100:

| Stage | Wall-clock | Cost @ $6/hr |
|---|---|---|
| Specialist teachers (6 × 0.5B LoRA) | ~0.5 hr | ~$3 |
| SFT (~0.5B, 3000 steps) | ~0.5–1 hr | ~$3–6 |
| RLVR (800 steps × 16 rollouts) | ~1–2 hr | ~$6–12 |
| MOPD (400 steps × 2 iters, specialists) | ~1.5–3 hr | ~$9–18 |
| **Total (one full run)** | **~3.5–6.5 hr** | **~$21–39** |

**Your $263 covers ~43 H100-hours** → roughly **6–12 full runs**, or one large run
(bigger student / longer schedules) with plenty of margin. A single comprehensive
run lands around **~$25–40**.

### Runtime note — KV cache is now built in

The LatentMoE attention now has a **KV cache** (transformers Cache API): `generate()`
output is identical to full-recompute, but ~**2× faster** at this scale (256-token
completions on a ~0.3B model), and the gain grows with completion length. Rollout
generation still scales with `--max_completion_length` × `--num_generations`, so those
remain the knobs that drive cost — but the per-token cost is now halved. The cost
estimates above already reflect the cache.

## Honest scope vs. real Ultra

- **Architecture**: real LatentMoE; real masked-IS GRPO; real per-domain-specialist MOPD ×2. ✅
- **Approximated**: synchronous rollouts (true async worker-pipelining is `--async_offpolicy`,
  still single-process here); ~6 envs not ~15; ~0.5B student not 550B; teachers are small
  LoRA specialists not 10+ full models.
- Real Ultra trained on **multi-rack GB200 NVL72** (hundreds–thousands of Blackwell GPUs) —
  not reproducible on one rented card. This reproduces the *recipe*, scaled to fit 80GB.
