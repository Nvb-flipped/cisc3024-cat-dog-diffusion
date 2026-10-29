"""Cache official metric weights separately from curation and training models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.common import ROOT, sha256, write_json
sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.features import load_inception


def main():
    folder = ROOT / 'models/evaluator/hub'
    torch.hub.set_dir(str(folder))
    load_inception('cpu')
    files = list(folder.rglob('*.pth')) + [ROOT / 'starter_kit/catdog_eval/classifier.pt']
    write_json('results/setup/evaluator_weights.json', {
        'files': {str(f.relative_to(ROOT)): sha256(f) for f in files},
        'purpose': 'official aggregate evaluation only; forbidden in curation/training/generation'})
    print('Official evaluator weights cached; no per-image dev features computed here.')


if __name__ == '__main__':
    main()
