"""Quick load/fit test for pre-quantized 4-bit Qwen3-30B-A3B (no training).

Checks: does it load & fit on 24GB, is the MoE router frozen-able, is expert
routing active (not collapsed), and does it generate coherently? Decides whether
the direct-4-bit MoE path is usable here despite the 'broken import' caveat.
"""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import unsloth  # noqa
from unsloth import FastLanguageModel
from moe_utils import confirm_router_frozen, probe_expert_routing

MODEL = sys.argv[1] if len(sys.argv) > 1 else "unsloth/Qwen3-30B-A3B-bnb-4bit"
torch.cuda.reset_peak_memory_stats()
model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=1024, load_in_4bit=True, max_lora_rank=16)
print(">>> LOADED. weights peak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated()/1024**3))

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32, use_gradient_checkpointing="unsloth", random_state=3407)
confirm_router_frozen(model)
FastLanguageModel.for_inference(model)
probe_expert_routing(model, tok)

ids = tok.apply_chat_template([{"role": "user", "content": "What is 23 * 17? Answer with just the number."}],
                              add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=64, temperature=0.7, do_sample=True)
print(">>> GEN:", repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:200]))
print(">>> PEAK VRAM after gen: %.2f GB / 24" % (torch.cuda.max_memory_allocated()/1024**3))
print(">>> LOAD_TEST_OK")
