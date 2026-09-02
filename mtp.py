"""
Phase 2 — Multi-Token Prediction (MTP) head + speculative-decoding rollouts.

Ultra accelerates RL/MOPD rollout generation with MTP speculative decoding:
"the MTP head is applied recurrently to propose k candidate tokens, which the base
model verifies in a single forward pass. Accepted tokens are committed without
additional sequential decoding steps." (Tech report §3.6.1)

This implements that on our LatentMoE:
  - MTPHead: a CHEAP drafter that, from the backbone's final hidden state h_t and the
    embedding of the just-emitted token, predicts the NEXT token (and a new hidden it
    can be applied recurrently to). Output goes through the backbone's (tied) lm_head.
  - train_mtp(): head-only training — backbone FROZEN — to predict token t+2 from
    (h_t, emb(token_{t+1})). This is exactly "MTP Boosting / head-only KL" (Phase 4),
    here done with CE for simplicity.
  - speculative_generate(): draft k tokens with the MTP head, verify in ONE backbone
    forward, accept the longest matching prefix + one correction. GREEDY spec-decoding
    is exact: output is identical to greedy backbone decoding, but uses fewer backbone
    forward passes -> that ratio is the rollout speedup.

Run:  python mtp.py    # train a tiny MTP head, verify exactness, report speedup
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_moe import RMSNorm, SwiGLU


class MTPHead(nn.Module):
    """Cheap recurrent drafter. proj([h; tok_emb]) -> SwiGLU residual -> hidden;
    tokens are read out through the backbone's tied lm_head."""
    def __init__(self, d, inner=None):
        super().__init__()
        inner = inner or 2 * d
        self.norm_h = RMSNorm(d)
        self.norm_e = RMSNorm(d)
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.mlp = SwiGLU(d, inner)
        self.norm_out = RMSNorm(d)

    def forward(self, h, tok_emb):
        # h: [B, d] previous hidden ; tok_emb: [B, d] embedding of last token
        z = self.proj(torch.cat([self.norm_h(h), self.norm_e(tok_emb)], dim=-1))
        z = z + self.mlp(z)
        return self.norm_out(z)        # new hidden; lm_head applied by caller


@torch.no_grad()
def _backbone(model, ids):
    out = model(input_ids=ids)
    return out.logits, out.hidden_states[0]   # logits [B,T,V], hidden [B,T,d]


def train_mtp(model, mtp, tok, device, steps=200, bs=16, seq=48, lr=3e-3,
              data_fn=None, log_every=50, loss_mode="kl"):
    """Phase 4 — MTP Boosting. Head-only (backbone FROZEN), train the MTP head to
    predict token t+2 from (h_t, emb_{t+1}).
      loss_mode="kl" : KL(MTP_draft || backbone) — "align MTP drafts with backbone
                       logits" (Ultra §3.3, the faithful boosting objective).
      loss_mode="ce" : plain cross-entropy to the gold token t+2.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mtp.to(device).train()
    opt = torch.optim.AdamW(mtp.parameters(), lr=lr)
    emb = model.get_input_embeddings()
    losses = []
    for step in range(1, steps + 1):
        ids = data_fn(bs, seq, device)
        with torch.no_grad():
            bb_logits, h = _backbone(model, ids)      # logits [B,T,V], hidden [B,T,d]
        h_t = h[:, :-2]                               # [B,T-2,d]
        tok_next = ids[:, 1:-1]                       # token_{t+1}
        z = mtp(h_t.reshape(-1, h_t.size(-1)),
                emb(tok_next).reshape(-1, h_t.size(-1)))
        draft_logits = model.lm_head(z)
        if loss_mode == "kl":
            # backbone's own distribution for token t+2 is its logits at position t+1
            with torch.no_grad():
                target = F.log_softmax(bb_logits[:, 1:-1].reshape(-1, bb_logits.size(-1)), -1)
            logp = F.log_softmax(draft_logits, -1)
            loss = (logp.exp() * (logp - target)).sum(-1).mean()   # KL(draft || backbone)
        else:
            loss = F.cross_entropy(draft_logits, ids[:, 2:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"[mtp] step {step:04d}/{steps} {loss_mode}_loss={losses[-1]:.4f}")
    return losses


@torch.no_grad()
def speculative_generate(model, mtp, ids, k=4, max_new=64):
    """Greedy speculative decoding. Returns (out_ids, stats).
    Exactness: output == greedy backbone decoding; speedup = tokens / backbone_forwards.
    Reuses the verify forward to seed the next step (no redundant reseed) unless a
    correction token (not processed by the backbone) was appended."""
    model.eval()
    emb = model.get_input_embeddings()
    forwards = 0
    produced = 0
    cur_logits = cur_h = None
    while produced < max_new:
        if cur_logits is None:                        # (re)seed
            logits, h = _backbone(model, ids); forwards += 1
            cur_logits, cur_h = logits[:, -1], h[:, -1]
        x1 = cur_logits.argmax(-1, keepdim=True)      # backbone's next token (verified)
        # --- draft k tokens recurrently with the cheap MTP head ---
        drafts, z, last_tok = [x1], cur_h, x1[:, 0]
        for _ in range(k):
            z = mtp(z, emb(last_tok))
            last_tok = model.lm_head(z).argmax(-1)
            drafts.append(last_tok.unsqueeze(1))
        cand = torch.cat(drafts, dim=1)               # [B, k+1]
        T = ids.size(1)
        # --- verify all drafts in ONE backbone forward ---
        ext = torch.cat([ids, cand], dim=1)
        v_logits, v_h = _backbone(model, ext); forwards += 1
        # v_pred[:, i] = backbone's greedy choice for cand[:, i]
        v_pred = v_logits[:, T - 1:T - 1 + cand.size(1)].argmax(-1)  # [B, k+1]
        # accept longest prefix where draft == backbone greedy (x1 always matches)
        n_accept = 1
        for j in range(1, cand.size(1)):
            if torch.equal(cand[:, j], v_pred[:, j]):
                n_accept += 1
            else:
                break
        if n_accept == cand.size(1):                  # all drafts accepted
            committed = cand
            # last committed token sits at ext pos T+k; its logits predict the next token
            cur_logits, cur_h = v_logits[:, T + k], v_h[:, T + k]
        else:                                         # correction = backbone's token at divergence
            correction = v_pred[:, n_accept:n_accept + 1]
            committed = torch.cat([cand[:, :n_accept], correction], dim=1)
            cur_logits = cur_h = None                 # correction not processed -> reseed
        ids = torch.cat([ids, committed], dim=1)
        produced += committed.size(1)
    return ids, {"forwards": forwards, "tokens": produced,
                 "tokens_per_forward": produced / max(1, forwards)}


@torch.no_grad()
def greedy_generate(model, ids, max_new=64):
    model.eval()
    forwards = 0
    for _ in range(max_new):
        logits, _ = _backbone(model, ids)
        forwards += 1
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return ids, {"forwards": forwards}


def _selftest():
    from latent_moe import LatentMoEConfig, LatentMoEForCausalLM
    torch.manual_seed(0)
    cfg = LatentMoEConfig(vocab_size=256, hidden_size=256, latent_size=64,
                          intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=8, num_experts=8, num_experts_per_tok=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentMoEForCausalLM(cfg).to(device)
    mtp = MTPHead(cfg.hidden_size)

    # learnable structured data so the backbone is predictable -> MTP can draft well
    def data_fn(bs, seq, dev):
        starts = torch.randint(0, cfg.vocab_size, (bs, 1), device=dev)
        offs = torch.arange(seq, device=dev).unsqueeze(0)
        return ((starts + offs) % cfg.vocab_size).long()

    # briefly train the BACKBONE on the cycle task so its greedy output is structured
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(150):
        ids = data_fn(16, 48, device)
        loss = model(input_ids=ids, labels=ids).loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print(f"[selftest] backbone trained on cycle task (final loss {float(loss):.3f})")

    train_mtp(model, mtp, None, device, steps=200, bs=16, seq=48, data_fn=data_fn)

    prompt = data_fn(1, 6, device)
    g_ids, g_stats = greedy_generate(model, prompt.clone(), max_new=48)
    s_ids, s_stats = speculative_generate(model, mtp, prompt.clone(), k=4, max_new=48)
    # exactness: spec decoding must match greedy decoding token-for-token
    L = min(g_ids.size(1), s_ids.size(1))
    exact = torch.equal(g_ids[:, :L], s_ids[:, :L])
    print(f"\n[selftest] greedy forwards: {g_stats['forwards']} for ~48 tokens")
    print(f"[selftest] spec   forwards: {s_stats['forwards']} for {s_stats['tokens']} tokens")
    print(f"[selftest] tokens/forward: {s_stats['tokens_per_forward']:.2f} "
          f"(>0.5 means MTP saved backbone passes)")
    print(f"[selftest] EXACT match greedy==spec: {exact}")
    print("[selftest] " + ("ALL PASS — MTP speculative rollouts work." if exact
                           else "WARN: spec output diverged (check verification)."))


if __name__ == "__main__":
    _selftest()
