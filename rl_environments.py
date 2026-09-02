"""
Multi-environment verifiable-reward suite for the mini Nemotron-3 RLVR replica.

Ultra's RLVR is a SINGLE unified stage spanning many environments trained
SIMULTANEOUSLY, each with a VERIFIABLE reward (no reward model), plus a
Gaussian-difficulty data mixture built from per-env "reward profiling" before
training (NVIDIA 2025b). This module reproduces that at 3090 scale:

  - Each environment = (problem generator, ideal SFT response, verifiable reward).
  - build_unified_dataset() interleaves environments by curriculum weight.
  - profile_rewards() samples the SFT student on each env to get a pass-rate,
    then gaussian_curriculum() weights envs toward mid-difficulty (~0.5 pass) so
    RL signal (reward variance) is maximal — the Gaussian-mixture idea, scaled.
  - reasoning-effort control: a fraction of prompts request a BRIEF answer with a
    truncated reasoning budget (Ultra introduces medium-effort in SFT, optimizes
    it in RLVR; ~2.5% of RLVR prompts are medium-effort).

Reward functions follow TRL's signature (prompts, completions, **kwargs) and read
the per-example `env`/`spec` dataset columns; each env's reward returns 0 on rows
that belong to other envs, so passing the whole list to GRPOTrainer sums to the
right per-domain contribution AND gives per-domain wandb logging (rewards/<env>/mean).
"""

import json
import random
import re

from datasets import Dataset

from reward_dataset import _completion_text, extract_answer

SYSTEM_PROMPT = (
    "You are a careful assistant. Think briefly inside one <think> ... </think> "
    "block, then output ONLY the final answer on the next line."
)
BRIEF_SUFFIX = " /no_think Answer in as few tokens as possible."

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# Environment generators: each returns (question, spec, ideal_response)
#   spec is a small JSON-serializable dict the reward fn uses to verify.
# ---------------------------------------------------------------------------
def _env_math(rng):
    kind = rng.choice(["add", "add", "sub", "mul1"])
    if kind == "add":
        a, b = rng.randint(0, 20), rng.randint(0, 20); op, ans = "+", a + b
    elif kind == "sub":
        a, b = rng.randint(0, 20), rng.randint(0, 20); a, b = max(a, b), min(a, b); op, ans = "-", a - b
    else:
        a, b = rng.randint(2, 9), rng.randint(2, 9); op, ans = "*", a * b
    q = f"What is {a} {op} {b}?"
    resp = f"<think>\n{a} {op} {b} = {ans}\n</think>\n{ans}"
    return q, {"answer": str(ans)}, resp


def _env_keyword(rng):
    word = rng.choice(["banana", "rocket", "violin", "pyramid", "glacier", "compass"])
    q = f"Write one short sentence that contains the word '{word}'."
    resp = f"<think>\nuse the word {word}.\n</think>\nThe {word} was right there."
    return q, {"keyword": word}, resp


def _env_count(rng):
    n = rng.randint(2, 5)
    items = ["red", "blue", "green", "gold", "gray", "pink"]
    pick = items[:n]
    q = f"List exactly {n} colors, comma-separated, nothing else."
    resp = f"<think>\nneed {n} colors.\n</think>\n" + ", ".join(pick)
    return q, {"count": n}, resp


def _env_json(rng):
    name = rng.choice(["ada", "linus", "grace", "alan"])
    age = rng.randint(20, 60)
    q = "Output a JSON object with keys 'name' (a string) and 'age' (an integer)."
    resp = f'<think>\nbuild json.\n</think>\n{{"name": "{name}", "age": {age}}}'
    return q, {"keys": ["name", "age"]}, resp


def _env_needle(rng):
    code = rng.randint(1000, 9999)
    n_distract = rng.randint(3, 8)
    lines = [f"Note {i}: nothing important here." for i in range(n_distract)]
    pos = rng.randint(0, n_distract)
    lines.insert(pos, f"The secret code is {code}.")
    ctx = "\n".join(lines)
    q = f"{ctx}\n\nWhat is the secret code?"
    resp = f"<think>\nscan for code.\n</think>\n{code}"
    return q, {"answer": str(code)}, resp


def _env_uppercase(rng):
    words = ["hello world", "good morning", "thank you", "well done", "see you"]
    phrase = rng.choice(words)
    q = f"Repeat this phrase in ALL UPPERCASE: {phrase}"
    resp = f"<think>\nuppercase it.\n</think>\n{phrase.upper()}"
    return q, {"target": phrase.upper()}, resp


ENVS = {
    "math": _env_math,
    "keyword": _env_keyword,
    "count": _env_count,
    "json": _env_json,
    "needle": _env_needle,
    "uppercase": _env_uppercase,
}


# ---------------------------------------------------------------------------
# Verifiable reward checks (return float in [0, 1]) given completion text + spec
# ---------------------------------------------------------------------------
def _final(text):
    """Text after the </think> block (the answer region)."""
    m = _THINK_RE.search(text)
    return text[m.end():].strip() if m else text.strip()


def _check(env, text, spec):
    ans = _final(text)
    if env == "math" or env == "needle":
        pred = extract_answer(text)
        return 1.0 if (pred is not None and pred == spec["answer"]) else 0.0
    if env == "keyword":
        return 1.0 if spec["keyword"].lower() in text.lower() else 0.0
    if env == "count":
        parts = [p for p in re.split(r",", ans) if p.strip()]
        return 1.0 if len(parts) == spec["count"] else 0.0
    if env == "json":
        try:
            obj = json.loads(ans)
            return 1.0 if all(k in obj for k in spec["keys"]) else 0.0
        except Exception:
            return 0.0
    if env == "uppercase":
        return 1.0 if ans == spec["target"] else 0.0
    return 0.0


def _format_ok(text):
    return 1.0 if (text.count("<think>") == 1 and text.count("</think>") == 1) else 0.0


# ---------------------------------------------------------------------------
# TRL reward functions: one correctness fn per env (zero on other envs) + a
# shared format reward. All summed by GRPOTrainer.
# ---------------------------------------------------------------------------
def make_reward_funcs(correct_weight=2.0, format_weight=0.5):
    funcs = []
    for env_name in ENVS:
        def _mk(env_name):
            def reward(prompts, completions, env, spec, **kwargs):
                out = []
                for comp, e, s in zip(completions, env, spec):
                    if e != env_name:
                        out.append(0.0)
                        continue
                    sp = json.loads(s) if isinstance(s, str) else s
                    out.append(correct_weight * _check(env_name, _completion_text(comp), sp))
                return out
            reward.__name__ = f"reward_{env_name}"
            return reward
        funcs.append(_mk(env_name))

    def reward_format(prompts, completions, **kwargs):
        return [format_weight * _format_ok(_completion_text(c)) for c in completions]
    funcs.append(reward_format)
    return funcs


# ---------------------------------------------------------------------------
# Dataset construction (curriculum-weighted interleave) + SFT examples
# ---------------------------------------------------------------------------
def _messages(q, brief):
    sys = SYSTEM_PROMPT + (BRIEF_SUFFIX if brief else "")
    return [{"role": "system", "content": sys}, {"role": "user", "content": q}]


def build_unified_dataset(n, seed, weights=None, brief_frac=0.025):
    """Interleave environments by curriculum `weights` (dict env->prob).
    `brief_frac` ~ Ultra's 2.5% medium-effort/budget-limited prompts."""
    rng = random.Random(seed)
    names = list(ENVS)
    if weights:
        w = [max(1e-3, weights.get(k, 0.0)) for k in names]
    else:
        w = [1.0] * len(names)
    rows = []
    for _ in range(n):
        env = rng.choices(names, weights=w, k=1)[0]
        q, spec, _resp = ENVS[env](rng)
        brief = rng.random() < brief_frac
        rows.append({
            "prompt": _messages(q, brief),
            "env": env,
            "spec": json.dumps(spec),
            "brief": brief,
        })
    return Dataset.from_list(rows)


def build_sft_examples(rng, k, brief_frac=0.1):
    """k supervised (question, ideal_response, brief) tuples across ALL envs,
    so the RLVR student has baseline competence in every domain."""
    names = list(ENVS)
    out = []
    for _ in range(k):
        env = rng.choice(names)
        q, _spec, resp = ENVS[env](rng)
        brief = rng.random() < brief_frac
        if brief:
            # truncated reasoning budget: drop the <think> block
            resp = _final(resp)
        out.append((_messages(q, brief), resp))
    return out


# ---------------------------------------------------------------------------
# Reward profiling + Gaussian difficulty curriculum
# ---------------------------------------------------------------------------
def profile_rewards(generate_fn, n_per_env=12, seed=123):
    """Sample the current model on each env; return {env: pass_rate}.
    generate_fn(messages)->completion_text is supplied by the caller."""
    rng = random.Random(seed)
    rates = {}
    for env, gen in ENVS.items():
        hits = 0
        for _ in range(n_per_env):
            q, spec, _ = gen(rng)
            text = generate_fn(_messages(q, False))
            hits += _check(env, text, spec)
        rates[env] = hits / n_per_env
    return rates


def gaussian_curriculum(pass_rates, target=0.5, sigma=0.25):
    """Weight each env by a Gaussian centered on target pass-rate: envs that are
    too easy (≈1.0) or too hard (≈0.0) get downweighted, mid-difficulty upweighted
    — maximizes reward variance / learning signal. Returns normalized dict."""
    import math
    raw = {e: math.exp(-((r - target) ** 2) / (2 * sigma ** 2)) for e, r in pass_rates.items()}
    z = sum(raw.values()) or 1.0
    return {e: v / z for e, v in raw.items()}


if __name__ == "__main__":
    rng = random.Random(0)
    print("environments:", list(ENVS))
    for env, gen in ENVS.items():
        q, spec, resp = gen(rng)
        r = _check(env, resp, spec)
        print(f"\n[{env}] Q: {q[:60]!r}\n   ideal->reward={r}  spec={spec}")
    ds = build_unified_dataset(12, seed=1)
    from collections import Counter
    print("\nunified dataset env mix:", Counter(ds["env"]))
    funcs = make_reward_funcs()
    print("reward funcs:", [f.__name__ for f in funcs])
