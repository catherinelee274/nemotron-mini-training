"""
Stage 0 environment + preflight verification.

Reports GPU / driver / CUDA / torch / RAM / disk, runs a tiny bf16 CUDA matmul
to confirm the stack is alive, verifies the Unsloth + vLLM + TRL import chain,
and runs the MoE disk/RAM preflight gate for the Qwen3-30B-A3B run.

Run:  python env_check.py
Exit code 0 = good to proceed; non-zero = a hard blocker was found.
"""

import os
import shutil
import subprocess
import sys

# Disk needed to download the ~60GB 16-bit Qwen3-30B-A3B and convert to 4-bit
# on the fly (the MoE import-4bit-directly path is broken, so we must go via
# the full-precision weights). Keep headroom -> require ~70GB.
MOE_DISK_REQUIRED_GB = 70
WORKDIR = os.path.dirname(os.path.abspath(__file__))


def hr(title):
    print(f"\n=== {title} ===")


def gb(num_bytes):
    return num_bytes / 1024**3


def main():
    blockers = []
    warnings = []

    hr("nvidia-smi")
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader"],
            text=True).strip()
        print("GPU:", out)
        smi_cuda = subprocess.check_output(["nvidia-smi"], text=True)
        for line in smi_cuda.splitlines():
            if "CUDA Version" in line:
                print("driver CUDA:", line.split("CUDA Version:")[1].strip().rstrip("|").strip())
                break
    except Exception as e:
        warnings.append(f"nvidia-smi failed: {e}")
        print("nvidia-smi failed:", e)

    hr("python / torch")
    print("python:", sys.version.split()[0])
    import torch
    print("torch:", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print("device:", torch.cuda.get_device_name(0), "| capability sm_%d%d" % cap)
        if cap[0] < 8:
            warnings.append("GPU is pre-Ampere; bf16 path may be slow/unsupported.")
    else:
        blockers.append("CUDA not available to torch.")

    hr("system RAM")
    try:
        import psutil
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        print(f"RAM total {gb(vm.total):.1f}GB | available {gb(vm.available):.1f}GB "
              f"| swap free {gb(sm.free):.1f}GB")
        ram_avail = gb(vm.available) + gb(sm.free)
    except Exception as e:
        print("psutil unavailable:", e)
        ram_avail = None

    hr("disk (workdir)")
    du = shutil.disk_usage(WORKDIR)
    print(f"{WORKDIR}\n  total {gb(du.total):.0f}GB | free {gb(du.free):.0f}GB")
    disk_free_gb = gb(du.free)

    hr("bf16 CUDA matmul")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        a = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print(f"OK -> dtype {c.dtype}, peak alloc {torch.cuda.max_memory_allocated()/1024**2:.1f} MiB")
        del a, b, c
        torch.cuda.empty_cache()
    else:
        blockers.append("Skipped matmul: no CUDA.")

    hr("import chain (unsloth / trl / vllm)")
    os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    try:
        import unsloth
        from unsloth import FastLanguageModel  # noqa: F401
        import trl
        from trl import GRPOTrainer, GRPOConfig  # noqa: F401
        import vllm
        import huggingface_hub
        print("unsloth", unsloth.__version__, "| trl", trl.__version__,
              "| vllm", vllm.__version__, "| hub", huggingface_hub.__version__)
        print("import chain OK")
    except Exception as e:
        blockers.append(f"import chain failed: {e}")
        print("import chain FAILED:", e)

    hr("MoE preflight gate (Qwen3-30B-A3B, Stage 2)")
    print(f"disk free {disk_free_gb:.0f}GB (need ~{MOE_DISK_REQUIRED_GB}GB for 60GB dl + 4bit convert)")
    if disk_free_gb < MOE_DISK_REQUIRED_GB:
        blockers.append(
            f"Insufficient disk for 30B: {disk_free_gb:.0f}GB < {MOE_DISK_REQUIRED_GB}GB.")
    if ram_avail is not None:
        print(f"RAM+swap available ~{ram_avail:.0f}GB (on-the-fly 4bit MoE convert is RAM-hungry)")
        if ram_avail < 24:
            warnings.append(
                f"Low RAM headroom (~{ram_avail:.0f}GB incl swap) for the 60GB->4bit "
                "convert; close apps before Stage 2 or expect heavy swapping.")

    hr("VERDICT")
    for w in warnings:
        print("  WARN:", w)
    if blockers:
        for b_ in blockers:
            print("  BLOCKER:", b_)
        print("\nNOT OK — resolve blockers above.")
        return 1
    print("  Stage 0/1 ready. Stage 2 (30B) disk OK"
          + ("; heed RAM warning." if warnings else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
