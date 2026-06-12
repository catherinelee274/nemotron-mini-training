import os
import sys

# Force the in-process engine (like Unsloth) so we test the same code path and
# avoid the spawn/__main__ multiprocessing machinery.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def main():
    from vllm import LLM, SamplingParams
    eager = "--eager" in sys.argv
    print("enforce_eager =", eager, flush=True)
    llm = LLM(model="Qwen/Qwen1.5-MoE-A2.7B-Chat", quantization="bitsandbytes",
              load_format="bitsandbytes", dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=1024, enforce_eager=eager)
    out = llm.generate(["Q: What is 2+2? A:"], SamplingParams(max_tokens=16, temperature=0))
    print("GEN_OK:", repr(out[0].outputs[0].text), flush=True)


if __name__ == "__main__":
    main()
