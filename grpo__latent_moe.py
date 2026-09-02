"""
GRPO post-training for the local from-scratch LatentMoE.

This is intentionally tiny and self-contained: the model is a byte-level
LatentMoE, warmed up with a short supervised arithmetic trace pass so GRPO has a
non-random policy to sample from, then optimized with verifiable arithmetic
rewards. Routers stay frozen; routed experts and shared latent projections train.

Run:
    python grpo__latent_moe.py
"""

import argparse
import random
import re
import time

import torch
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from latent_moe import LatentMoEConfig, LatentMoEForCausalLM


class ByteTokenizer(PreTrainedTokenizer):
    """Minimal byte tokenizer sufficient for TRL's regular generate path."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, model_max_length=256):
        self._vocab = {f"<byte_{i}>": i for i in range(256)}
        super().__init__(
            pad_token="<pad>",
            eos_token="<eos>",
            padding_side="left",
            model_max_length=model_max_length,
        )

    @property
    def vocab_size(self):
        return 256

    def get_vocab(self):
        return dict(self._vocab, **{self.pad_token: 0, self.eos_token: 4})

    def _tokenize(self, text, **kwargs):
        return [chr(b) for b in text.encode("utf-8")]

    def _convert_token_to_id(self, token):
        if token == self.pad_token:
            return 0
        if token == self.eos_token:
            return 4
        data = token.encode("utf-8", errors="ignore")
        return data[0] if data else 0

    def _convert_id_to_token(self, index):
        if index == 0:
            return self.pad_token
        if index == 4:
            return self.eos_token
        return bytes([int(index) % 256]).decode("utf-8", errors="ignore")

    def __call__(
        self,
        text=None,
        return_tensors=None,
        padding=False,
        padding_side=None,
        max_length=None,
        truncation=False,
        add_special_tokens=False,
        **kwargs,
    ):
        if text is None:
            text = kwargs.get("texts", "")
        texts = [text] if isinstance(text, str) else list(text)
        max_length = max_length or self.model_max_length
        encoded = []
        for item in texts:
            ids = list(str(item).encode("utf-8"))
            if add_special_tokens:
                ids.append(self.eos_token_id)
            if truncation and len(ids) > max_length:
                ids = ids[-max_length:]
            encoded.append(ids)

        if padding:
            side = padding_side or self.padding_side
            width = max(len(ids) for ids in encoded) if encoded else 0
            if max_length is not None:
                width = min(max(width, 1), max_length)
            padded, masks = [], []
            for ids in encoded:
                ids = ids[-width:]
                pad_len = width - len(ids)
                if side == "left":
                    padded.append([self.pad_token_id] * pad_len + ids)
                    masks.append([0] * pad_len + [1] * len(ids))
                else:
                    padded.append(ids + [self.pad_token_id] * pad_len)
                    masks.append([1] * len(ids) + [0] * pad_len)
        else:
            padded = encoded
            masks = [[1] * len(ids) for ids in encoded]

        data = {"input_ids": padded, "attention_mask": masks}
        if return_tensors == "pt":
            data = {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}
        return BatchEncoding(data)

    def batch_decode(self, sequences, skip_special_tokens=True, **kwargs):
        out = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.detach().cpu().tolist()
            vals = []
            for item in seq:
                item = int(item) % 256
                if skip_special_tokens and item in (self.pad_token_id, self.eos_token_id):
                    continue
                vals.append(item)
            out.append(bytes(vals).decode("utf-8", errors="ignore"))
        return out

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


def make_problem(rng):
    kind = rng.choice(["add", "add", "mul"])
    if kind == "mul":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        op, ans = "*", a * b
    else:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op, ans = "+", a + b
    prompt = f"User: What is {a} {op} {b}?\nAssistant:"
    response = f" <think>\n{a} {op} {b} = {ans}\n</think>\n{ans}\n" + chr(4)
    return prompt, response, str(ans)


def build_grpo_dataset(n, seed):
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n:
        prompt, _response, answer = make_problem(rng)
        if prompt in seen:
            continue
        seen.add(prompt)
        rows.append({"prompt": prompt, "answer": answer})
    return Dataset.from_list(rows)


def extract_answer(text):
    after = text
    match = re.search(r"</think>", text)
    if match:
        after = text[match.end():]
    nums = re.findall(r"-?\d+", after)
    if not nums:
        nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None


def correctness_reward(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = extract_answer(completion)
        rewards.append(2.0 if pred == str(gold) else 0.0)
    return rewards


def format_reward(prompts, completions, **kwargs):
    rewards = []
    for text in completions:
        score = 0.0
        if text.count("<think>") == 1 and text.count("</think>") == 1:
            score += 0.5
            tail = text.split("</think>", 1)[1].strip()
            if re.fullmatch(r"-?\d+", tail):
                score += 0.5
            elif re.search(r"-?\d+", tail):
                score += 0.25
        rewards.append(score)
    return rewards


def freeze_routers(model):
    routers = 0
    for name, param in model.named_parameters():
        if name.endswith(".gate.weight"):
            param.requires_grad_(False)
            routers += 1
    return routers


@torch.no_grad()
def routing_diversity(model, device, seq_len=64):
    fired = set()
    handles = []
    topk = model.config.num_experts_per_tok

    def hook(_module, _inp, out):
        idx = torch.topk(out.float(), min(topk, out.shape[-1]), dim=-1).indices
        fired.update(idx.flatten().cpu().tolist())

    for name, module in model.named_modules():
        if name.endswith(".mlp.gate"):
            handles.append(module.register_forward_hook(hook))
    ids = torch.randint(1, 128, (4, seq_len), device=device)
    model(input_ids=ids)
    for handle in handles:
        handle.remove()
    return len(fired), len(handles)


def sft_warmup(model, tokenizer, args, device):
    if args.sft_steps <= 0:
        return []
    rng = random.Random(args.seed + 17)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.sft_lr,
        weight_decay=args.weight_decay,
    )
    losses = []
    model.train()
    for step in range(1, args.sft_steps + 1):
        texts, prompt_lens = [], []
        for _ in range(args.batch_size):
            prompt, response, _answer = make_problem(rng)
            texts.append(prompt + response)
            prompt_lens.append(len(prompt.encode("utf-8")))
        batch = tokenizer(
            text=texts,
            padding=True,
            padding_side="right",
            truncation=True,
            max_length=args.seq_len,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        labels = batch["input_ids"].clone()
        for row, prompt_len in enumerate(prompt_lens):
            labels[row, :prompt_len] = -100
            labels[row, batch["attention_mask"][row] == 0] = -100

        loss = model(input_ids=batch["input_ids"], labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.max_grad_norm,
        )
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % args.log_every == 0 or step == args.sft_steps:
            print(f"[sft] step {step:04d}/{args.sft_steps} loss={losses[-1]:.4f}")
    return losses


class RewardCurveCB(TrainerCallback):
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "reward" in logs:
            self.history.append((state.global_step, logs["reward"]))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--sft_steps", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_generations", type=int, default=2)
    ap.add_argument("--n_problems", type=int, default=24)
    ap.add_argument("--seq_len", type=int, default=160)
    ap.add_argument("--max_prompt_length", type=int, default=64)
    ap.add_argument("--max_completion_length", type=int, default=96)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--latent_size", type=int, default=32)
    ap.add_argument("--intermediate_size", type=int, default=64)
    ap.add_argument("--shared_intermediate_size", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--sft_lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--cpu_threads", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))

    cfg = LatentMoEConfig(
        vocab_size=256,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        intermediate_size=args.intermediate_size,
        shared_intermediate_size=args.shared_intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_experts=args.experts,
        num_experts_per_tok=args.top_k,
        max_position_embeddings=args.seq_len,
        pad_token_id=0,
        eos_token_id=4,
    )
    model = LatentMoEForCausalLM(cfg).to(device)
    tokenizer = ByteTokenizer(model_max_length=args.seq_len)
    routers = freeze_routers(model)
    distinct_before, n_router = routing_diversity(model, device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n########## GRPO LATENT MOE ##########")
    print(f"device={device} params={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")
    print(f"routers frozen: {routers} tensors; routing before: "
          f"{distinct_before}/{args.experts} experts across {n_router} layers")

    t0 = time.time()
    sft_losses = sft_warmup(model, tokenizer, args, device)
    if sft_losses:
        n = min(10, len(sft_losses))
        print(f"[sft] first-window {sum(sft_losses[:n])/n:.4f} -> "
              f"last-window {sum(sft_losses[-n:])/n:.4f}")

    dataset = build_grpo_dataset(args.n_problems, args.seed + 99)
    cfg_kwargs = dict(
        output_dir="outputs/grpo__latent_moe",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        generation_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.steps,
        learning_rate=args.lr,
        warmup_ratio=0.0,
        optim="adamw_torch",
        temperature=0.9,
        top_p=1.0,
        beta=0.0,
        logging_steps=1,
        save_strategy="no",
        save_steps=10_000,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=False,
        use_vllm=False,
        use_cpu=device.type == "cpu",
        dataloader_pin_memory=False,
    )
    config = GRPOConfig(**cfg_kwargs)
    reward_cb = RewardCurveCB()
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=dataset,
        callbacks=[reward_cb],
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer.train()

    distinct_after, _ = routing_diversity(model, device)
    print("\n########## GRPO LATENT MOE RESULTS ##########")
    if reward_cb.history:
        vals = [v for _, v in reward_cb.history]
        print("reward trace:", [f"{v:.2f}" for v in vals])
        print(f"reward first={vals[0]:.3f} last={vals[-1]:.3f}")
    print(f"routing distinct experts before/after: {distinct_before} -> {distinct_after}")
    print(f"elapsed: {time.time() - t0:.1f}s")
    if device.type == "cuda":
        print(f"peak VRAM allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print("########## GRPO LATENT MOE DONE ##########")


if __name__ == "__main__":
    main()
