"""Regenerate every CSV quoted in the write-up, in order: B (no model), A, C, D.

    python -m eval.run_all            # needs Ollama with gemma4:e2b-it-qat and the TabPFN v2 checkpoint

Experiment A caches Gemma's replies in eval/results/cache, so a rerun only rescores.
"""
import subprocess
import sys

STEPS = [
    ["eval.exp_b_faults"],
    ["eval.exp_a_extract"],
    ["eval.exp_c_forecast"],
    ["eval.exp_d_adversarial"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"== {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, "-m", *step], check=True)
