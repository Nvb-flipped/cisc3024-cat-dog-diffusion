"""Download only the pinned curation detector; never use it in generation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import read_config, sha256, timestamp, write_json


def main():
    from huggingface_hub import HfApi, snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/curation.yaml')
    args = p.parse_args()
    config = read_config(args.config)
    model_id = config['model_id']
    revision = config.get('revision') or HfApi().model_info(model_id).sha
    folder = Path(config['model_path'])
    info = HfApi().model_info(model_id, revision=revision)
    has_safe = any(item.rfilename.endswith('.safetensors') for item in info.siblings)
    weights_pattern = '*.safetensors' if has_safe else 'pytorch_model.bin'
    snapshot_download(model_id, revision=revision, local_dir=folder,
                      allow_patterns=['*.json', '*.txt', weights_pattern, 'README.md', 'LICENSE*'])
    write_json(folder / 'provenance.json', {
        'model_id': model_id, 'revision': revision, 'downloaded_at': timestamp(),
        'purpose': 'pool-only curation research, excluded from generator training and submission',
        'source': f'https://huggingface.co/{model_id}/tree/{revision}',
        'files': {str(f.relative_to(folder)): sha256(f) for f in folder.rglob('*')
                  if f.is_file() and '.cache' not in f.parts and f.name != 'provenance.json'}})
    print(folder / 'provenance.json')


if __name__ == '__main__':
    main()
