#!/usr/bin/env bash
# ===========================================================================
# FAITHFUL Ultra-RL-stage replica — single H100 (80GB) launch.
# Runs the full pipeline at a scale that needs a rented 80GB GPU, NOT the 3090:
#   0) train per-domain SPECIALIST teachers          (h100_faithful/train_specialist_teachers.py)
#   1) SFT  -> RLVR (masked-IS GRPO, KL guard)       (../nemotron_mini_rl.py)
#   2) MOPD with per-domain specialists + small KL   (../mopd.py --teacher_map)
#
# Differences from the 3090 scripts: ~0.5B student (generalizes instead of
# memorizing), 16 rollouts, specialist teachers so the MOPD KL term HELPS,
# longer schedules. Override any scale knob via env vars below.
#
# Usage on the pod:   bash h100_faithful/run_faithful.sh
# ===========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # h100_faithful
ROOT="$(dirname "$HERE")"                               # repo root
cd "$ROOT"
set -a; . ./.env 2>/dev/null || true; set +a           # WANDB_API_KEY etc

# -------- scale knobs (override via env, e.g. STUDENT_LAYERS=20 bash ...) -----
HIDDEN=${HIDDEN:-768}; LATENT=${LATENT:-192}; INTER=${INTER:-768}
SHARED_INTER=${SHARED_INTER:-2048}; LAYERS=${LAYERS:-12}; HEADS=${HEADS:-12}
EXPERTS=${EXPERTS:-16}; TOPK=${TOPK:-4}
MAXP=${MAXP:-256}; MAXC=${MAXC:-256}
SFT_STEPS=${SFT_STEPS:-3000}; RL_STEPS=${RL_STEPS:-800}
MOPD_STEPS=${MOPD_STEPS:-400}; GENS=${GENS:-16}
TEACHER_STEPS=${TEACHER_STEPS:-600}

STUDENT="--hidden_size $HIDDEN --latent_size $LATENT --intermediate_size $INTER \
  --shared_intermediate_size $SHARED_INTER --layers $LAYERS --heads $HEADS \
  --experts $EXPERTS --top_k $TOPK --max_prompt_length $MAXP --max_completion_length $MAXC"

echo "######## STAGE 0: specialist teachers ########"
python "$HERE/train_specialist_teachers.py" --base Qwen/Qwen2.5-0.5B-Instruct \
  --steps "$TEACHER_STEPS" --out "$HERE/teachers"

# assemble --teacher_map from the produced specialist dirs
MAP=""
shopt -s nullglob                                       # empty dir -> no literal glob
for d in "$HERE"/teachers/*/; do env="$(basename "$d")"; MAP="$MAP ${env}=${d%/}"; done
shopt -u nullglob
[ -n "$MAP" ] || { echo "ERROR: stage 0 produced no specialist teachers in $HERE/teachers" >&2; exit 1; }
echo "teacher_map:$MAP"

echo "######## STAGE 1: SFT -> RLVR (H100 scale) ########"
python nemotron_mini_rl.py --stage all --multi_env --train_experts --wandb $STUDENT \
  --sft_steps "$SFT_STEPS" --sft_batch 64 --sft_eval_every 500 \
  --rl_steps "$RL_STEPS" --num_generations "$GENS" --num_iterations 2 \
  --beta 0.03 --n_problems 256

echo "######## STAGE 2: MOPD (per-domain specialists, 2 iterations) ########"
python mopd.py --student_ckpt outputs/nemotron_mini/rlvr.pt --wandb $STUDENT \
  --teacher_map $MAP --steps "$MOPD_STEPS" --iterations 2 \
  --reward_weight 1.0 --kl_weight 0.2 --batch 16

echo "######## FAITHFUL RUN COMPLETE ########"
