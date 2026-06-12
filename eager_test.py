import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import unsloth  # noqa  (must import before unsloth_zoo)
from unsloth import FastLanguageModel
import unsloth_zoo.vllm_utils as vu

_orig = vu.load_vllm
def _patched(*a, **k):
    k["enforce_eager"] = True
    print(">>> load_vllm patched: enforce_eager=True", flush=True)
    return _orig(*a, **k)
vu.load_vllm = _patched

model, tok = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat",
    max_seq_length=1024,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=16,
    gpu_memory_utilization=0.85,
)
print(">>> FROM_PRETRAINED_OK (vLLM engine initialized)", flush=True)
out = model.fast_generate(["Q: What is 2+2? A:"],
                          sampling_params=None) if hasattr(model, "fast_generate") else None
print(">>> GEN:", out, flush=True)
