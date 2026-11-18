"""Screen same-label autoguidance on the exact v007 training trajectory."""

import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.run_coverage_comparison import verified_checkpoint
from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.check import check_files
from src.training.optimize import evaluate, quality_improved


def build_export(export, reference, checkpoint, guide, sampler, scale):
    main = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weaker = torch.load(guide, map_location='cpu', weights_only=True)
    for key in ('model', 'diffusion'):
        if main['config'][key] != weaker['config'][key]:
            raise ValueError(f'Autoguidance checkpoints disagree on {key}')
    if main['manifest_sha256'] != weaker['manifest_sha256'] or main['seed'] != weaker['seed']:
        raise ValueError('Autoguidance requires one exact-manifest training trajectory')
    state = {f'{member}.{name}': value for member, source in (('main', main), ('guide', weaker))
             for name, value in source['ema'].items()}
    source = Path(reference['exported'])
    provenance = read(source / 'provenance.json')
    with TemporaryDirectory(dir=export.parent) as temp:
        folder = Path(temp) / export.name
        folder.mkdir()
        for filename in ('__init__.py', 'model.py', 'diffusion.py', 'manifest.csv'):
            shutil.copy2(source / filename, folder / filename)
        shutil.copy2(Path(__file__).with_name('autoguided_generate.py'), folder / 'generate.py')
        torch.save(state, folder / 'weights.pt')
        config = read(source / 'inference.json')
        config['sampling'].update(guidance=sampler['guidance'], pag_scale=0, score_sharpness=1)
        config['autoguidance_scale'] = scale
        write_json(folder / 'inference.json', config)
        provenance.update(guide_checkpoint_sha256=sha256(guide), guide_training_update=weaker['update'],
                          guide_training_segment_id=weaker.get('segment_id'),
                          weights_sha256=sha256(folder / 'weights.pt'),
                          config_sha256=sha256(folder / 'inference.json'),
                          parameters=2 * provenance['parameters'], exported=timestamp())
        if provenance['parameters'] > 18_000_000:
            raise ValueError('Autoguidance exceeds the parameter ceiling')
        provenance['inference_source_sha256']['generate.py'] = sha256(folder / 'generate.py')
        write_json(folder / 'provenance.json', provenance)
        check_files(folder)
        folder.rename(export)


def main():
    name = 'v007_autoguidance'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    manifest = Path('manifests/v007.csv')
    reference_path = Path('results/experiments/v007_32k_temporal103/selection.json')
    reference = read(reference_path)['candidate']
    checkpoint = Path(reference['checkpoint'])
    guide = verified_checkpoint(checkpoint.parent, 20000, sha256(manifest))
    anchor = validated_result(reference['evidence'][0]['path'], 0, reference['exported'])
    sampler = dict(weights='ema', steps=50, guidance=1.0)
    scale = 1.0
    paths = (Path(__file__), Path(__file__).with_name('autoguided_generate.py'), manifest,
             checkpoint, guide, reference_path, Path('scripts/run_v002_dpmpp.py'),
             Path('src/evaluation/run.py'), Path('src/training/optimize.py'))
    spec = dict(inputs={str(path): sha256(path) for path in paths}, sampler=sampler,
                autoguidance_scale=scale, seed=0, n_per_class=1000,
                command=[sys.executable, '-u', '-m', 'scripts.run_v007_autoguidance'],
                source='https://arxiv.org/abs/2406.02507',
                rule='Replace CFG/PAG/temporal rescaling with same-label extrapolation from the 20k EMA to '
                     'the 32k EMA. Keep all DDIM50 indices, execution shape and v007 crop rows fixed.')
    if not plan.exists():
        write_json(plan, spec | {'created': timestamp()})
    if any(read(plan)[key] != value for key, value in spec.items()):
        raise ValueError('Frozen autoguidance inputs changed')
    verify_starter()
    if not export.exists():
        build_export(export, reference, checkpoint, guide, sampler, scale)
    if (read(export / 'inference.json')['autoguidance_scale'],
            read(export / 'provenance.json')['guide_checkpoint_sha256']) != (scale, sha256(guide)):
        raise ValueError('Export does not match the declared guide')
    if not (out / 'interface.json').exists():
        checks = verify_interface(export, export)
        write_json(out / 'interface.json', checks | {'solver': 'ddim', 'autoguidance_scale': scale})
    result = evaluate(checkpoint, manifest, export, out, **sampler)
    advances = quality_improved({'score': score_results([anchor])}, result)
    write_json(out / 'screen.json', {'plan_sha256': sha256(plan), 'candidate': result, 'advances': advances})
    print({'seed': 0, 'marks': result['evidence'][0]['marks'], 'advances': advances}, flush=True)


if __name__ == '__main__':
    main()
