"""
Shared reward + dataset module for the GRPO-on-MoE experiments.

GRPO needs *verifiable* rewards (not instruction->response pairs). We build a
tiny deterministic arithmetic dataset with known integer answers and two reward
functions:

  * correctness_reward : +2.0 for an exact numeric match, else 0.0
  * format_reward      : small bonus for a clean <think>...</think> block
                         followed by a final answer.

Used by both grpo_small_moe.py (Stage 1) and grpo_qwen3_30b.py (Stage 2) so the
reward/env wiring is identical across the small-MoE smoke test and the real
30B target.
"""

import random
import re

from datasets import Dataset

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
# Reasoning model convention: think inside <think>...</think>, then give the
# final answer (a bare integer) after the closing tag.
SYSTEM_PROMPT = (
    "You are a careful math assistant. Reason step by step inside a single "
    "<think> ... </think> block, then on a new line output ONLY the final "
    "integer answer with no extra words. For example:\n"
    "<think>\n2 + 3 = 5\n</think>\n5"
)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_INT_RE = re.compile(r"-?\d+")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def _gen_problem(rng):
    """One arithmetic problem with a guaranteed integer answer.

    Difficulty is deliberately tuned so a small model gets it right only *some*
    of the time. GRPO learns from *within-group* reward variance: if every
    sampled completion is correct (trivial problems), the advantage is zero and
    nothing updates. Harder 2-digit multiplication / multi-term sums give the
    policy gradient a signal to climb.
    """
    kind = rng.choice(["mul3x2", "mul3x2", "mul2", "three_term"])
    if kind == "mul3x2":  # 3-digit x 2-digit: even a strong 30B slips here
        a, b = rng.randint(123, 989), rng.randint(13, 89)
        return f"What is {a} * {b}?", str(a * b)
    if kind == "mul2":  # 2-digit x 2-digit
        a, b = rng.randint(13, 89), rng.randint(13, 89)
        return f"What is {a} * {b}?", str(a * b)
    # multi-term: a*b + c*d
    a, b = rng.randint(11, 49), rng.randint(11, 49)
    c, d = rng.randint(11, 49), rng.randint(11, 49)
    return f"What is {a} * {b} + {c} * {d}?", str(a * b + c * d)


def build_dataset(n=50, seed=3407):
    """Return a conversational GRPO dataset with columns `prompt` and `answer`.

    `prompt` is a list of chat messages (the trainer applies the chat
    template); `answer` is the gold integer as a string.
    """
    rng = random.Random(seed)
    rows = []
    seen = set()
    while len(rows) < n:
        q, a = _gen_problem(rng)
        if q in seen:
            continue
        seen.add(q)
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
                "answer": a,
            }
        )
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def _completion_text(completion):
    """Normalize a GRPO completion (conversational list or raw string) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # [{"role": "assistant", "content": ...}, ...]
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def extract_answer(text):
    """Pull the model's final integer answer.

    Prefer the text *after* the </think> block (that's where the final answer
    should be); fall back to the last integer anywhere in the string.
    """
    after = text
    m = _THINK_RE.search(text)
    if m:
        after = text[m.end():]
    nums = _INT_RE.findall(after)
    if not nums:
        nums = _INT_RE.findall(text)  # fallback: anywhere
    return nums[-1] if nums else None


# ---------------------------------------------------------------------------
# Reward functions (TRL GRPO signature: (prompts, completions, **kwargs))
# Dataset columns (e.g. `answer`) arrive as kwargs lists aligned with completions.
# ---------------------------------------------------------------------------
def correctness_reward(prompts, completions, answer, **kwargs):
    """+2.0 for an exact numeric match with the gold answer, else 0.0."""
    rewards = []
    for comp, gold in zip(completions, answer):
        pred = extract_answer(_completion_text(comp))
        ok = False
        if pred is not None:
            try:
                ok = int(pred) == int(gold)
            except ValueError:
                ok = False
        rewards.append(2.0 if ok else 0.0)
    return rewards


def format_reward(prompts, completions, **kwargs):
    """Reward clean reasoning formatting.

    +0.5 for exactly one well-formed <think>...</think> block, and an extra
    +0.5 if a bare integer answer follows the closing tag.
    """
    rewards = []
    for comp in completions:
        text = _completion_text(comp)
        score = 0.0
        if text.count("<think>") == 1 and text.count("</think>") == 1:
            m = _THINK_RE.search(text)
            if m and m.start() < text.index("</think>"):
                score += 0.5
                tail = text[m.end():].strip()
                # tail should be (essentially) just the integer
                if re.fullmatch(r"-?\d+", tail):
                    score += 0.5
                elif _INT_RE.search(tail):
                    score += 0.25
        rewards.append(score)
    return rewards


REWARD_FUNCS = [correctness_reward, format_reward]
# Theoretical max per sample = 2.0 (correct) + 1.0 (perfect format) = 3.0


if __name__ == "__main__":
    ds = build_dataset(50)
    print(f"dataset: {len(ds)} rows; columns={ds.column_names}")
    print("example prompt:", ds[0]["prompt"][-1]["content"], "-> answer", ds[0]["answer"])
    # self-test the reward functions on a perfect and a wrong completion
    good = "<think>\n23 + 47 = 70\n</think>\n70"
    bad = "the answer is probably 99"
    print("correctness good/bad:",
          correctness_reward(None, [good, bad], answer=["70", "70"]))
    print("format good/bad:", format_reward(None, [good, bad]))
