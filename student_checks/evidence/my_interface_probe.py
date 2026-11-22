"""Independent interface probes for submission/ (student check 3).

Run from anywhere (the script finds the project root itself):
    python student_checks/evidence/my_interface_probe.py 2>&1 | tee student_checks/evidence/my_interface_probe.log
"""
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch

# The script lives in student_checks/evidence/, so the project root is two levels up.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import submission.generate as g  # noqa: E402

EXACT = 0.0    # expected: the sampler is deterministic and keeps each output in its own slot
NOISE = 1e-4   # above exact but below this: report as floating-point noise (see the AGENTS.md pitfall)
results = {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute difference between two tensors."""
    return (a - b).abs().max().item()


def verdict(diff: float) -> str:
    if diff == EXACT:
        return "PASS (identical)"
    if diff <= NOISE:
        return "NOISE (tiny, report it)"
    return "FAIL"


def run_probe(name: str, func):
    """Run one probe; a crash is recorded as an ERROR in the summary instead of vanishing."""
    print(f"\n--- {name} ---")
    try:
        results[name] = func()
    except Exception as e:  # noqa: BLE001
        results[name] = f"ERROR: {e}"
    print(results[name])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | torch {torch.__version__}")
    for name in ("weights.pt", "manifest.csv"):
        print(f"sha256 submission/{name}: {sha256(ROOT / 'submission' / name)}")
    print("Loading model...")
    model = g.load_model(device)

    labels_mixed = torch.tensor([0, 1, 0, 1], device=device)
    seed_a, seed_b = 5, 6

    with torch.no_grad():
        ref = g.generate(model, labels_mixed, seed_a)   # reference output reused by several probes

        def probe1():
            d = max_diff(ref, g.generate(model, labels_mixed, seed_a))
            return f"max diff {d:.3e} -> {verdict(d)}"
        run_probe("Probe 1: Determinism (same labels and seed)", probe1)

        def probe2():
            other = g.generate(model, labels_mixed, seed_b)
            per_image = (ref - other).abs().flatten(1).mean(1)   # one number per image
            ok = bool((per_image > 0.01).all())
            return (f"mean |diff| per image {[round(v, 4) for v in per_image.tolist()]} "
                    f"-> {'PASS' if ok else 'FAIL'} (every image should change)")
        run_probe("Probe 2: Seed sensitivity (seed 5 vs seed 6)", probe2)

        def probe3():
            a = g.generate(model, torch.tensor([0, 0, 0, 0], device=device), seed_a)
            b = g.generate(model, torch.tensor([0, 1, 1, 1], device=device), seed_a)
            d = max_diff(a[0], b[0])                        # same label, same slot; other labels differ
            control = (a[1] - b[1]).abs().mean().item()     # slot 1 has label 0 vs 1: must differ
            return (f"image 0 max diff {d:.3e} -> {verdict(d)}; "
                    f"control (slot 1, label 0 vs 1) mean |diff| {control:.4f} "
                    f"-> {'PASS' if control > 0.01 else 'FAIL (are labels ignored?)'}")
        run_probe("Probe 3: Independence from other labels (image 0) + label control", probe3)

        def probe4():
            diffs = {}
            single = g.generate(model, torch.tensor([0], device=device), seed_a)
            diffs["N=1 vs image 0 of N=4"] = max_diff(single[0], ref[0])
            for n in (40, 65):                               # larger than one execution batch of 32
                labels = torch.tensor([i % 2 for i in range(n)], device=device)   # starts 0,1,0,1
                big = g.generate(model, labels, seed_a)
                diffs[f"first 4 images of N={n} vs N=4"] = max_diff(big[:4], ref)
            lines = "; ".join(f"{k}: {v:.3e}" for k, v in diffs.items())
            return f"{lines} -> {verdict(max(diffs.values()))}"
        run_probe("Probe 4: Single label and prefixes of larger batches", probe4)

        def probe5():
            shape_ok = ref.shape == torch.Size([4, 3, 32, 32])
            type_ok = ref.dtype == torch.float32
            lo, hi = ref.min().item(), ref.max().item()
            range_ok = lo >= 0.0 and hi <= 1.0
            finite = bool(torch.isfinite(ref).all())
            print(f"Shape {tuple(ref.shape)} ok={shape_ok} | dtype {ref.dtype} ok={type_ok} | "
                  f"range [{lo:.4f}, {hi:.4f}] ok={range_ok} | all finite={finite}")
            return "PASS" if (shape_ok and type_ok and range_ok and finite) else "FAIL"
        run_probe("Probe 5: Shape, dtype, value range, NaN/Inf", probe5)

        def probe6():
            torch.manual_seed(123)
            random.seed(123)
            np.random.seed(123)

            def snapshot():
                return (torch.get_rng_state(), random.getstate(), np.random.get_state()[1].copy(),
                        torch.cuda.get_rng_state() if device.type == "cuda" else None)
            before = snapshot()
            g.generate(model, labels_mixed, seed_a)
            after = snapshot()
            same = {"torch CPU": torch.equal(before[0], after[0]),
                    "python random": before[1] == after[1],
                    "numpy": np.array_equal(before[2], after[2]),
                    "torch CUDA": True if before[3] is None else torch.equal(before[3], after[3])}
            return f"{same} -> {'PASS' if all(same.values()) else 'FAIL'}"
        run_probe("Probe 6 (optional): caller's global RNG state is preserved", probe6)

    print("\n" + " SUMMARY ".center(70, "="))
    for name, value in results.items():
        print(f"{name}\n    {value}")
    print("=" * 70)
    print("Note: ~1e-5 differences would be the known GPU floating-point noise from AGENTS.md; report the number.")
    failed = [n for n, v in results.items() if v.startswith("ERROR") or "FAIL" in v]
    print("FAILED PROBES:" if failed else "No probe failed.", *failed, sep="\n  " if failed else " ")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
