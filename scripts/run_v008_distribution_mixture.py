"""Screen a fixed sample-space mixture of the verified v007 and v008 fits."""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.run_coverage_comparison import rows, verified_checkpoint
from scripts.run_v002_dpmpp import read
from scripts.run_v008_coverage import MANIFEST, PLAN, REFERENCE, RUN
from scripts.setup_inputs import verify_starter
from src.common import sha256, timestamp, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.check import check_files
from src.training.optimize import evaluate, quality_improved


def build(export, original, expanded, template):
    first, second = (Path(row['exported']) for row in (original, expanded))
    if not rows(first / 'manifest.csv') < rows(second / 'manifest.csv'):
        raise ValueError('The declared manifest must be the exact union of both training manifests')
    if read(first / 'inference.json') != read(second / 'inference.json'):
        raise ValueError('The mixture requires the identical architecture and sampler')
    if any(sha256(first / file) != sha256(second / file) for file in ('model.py', 'diffusion.py')):
        raise ValueError('Model and sampler sources differ between the two exports')
    expanded_record = read(second / 'provenance.json')
    original_record = read(first / 'provenance.json')
    state = {}
    for index, source in enumerate((first, second)):
        weights = torch.load(source / 'weights.pt', map_location='cpu', weights_only=True)
        state.update({f'{index}.{name}': value for name, value in weights.items()})
    parameters = expanded_record['parameters'] + original_record['parameters']
    if parameters > 18_000_000:
        raise ValueError('The combined module exceeds the engineering parameter ceiling')
    with TemporaryDirectory(dir=export.parent) as temp:
        folder = Path(temp) / export.name
        folder.mkdir()
        for name in ('__init__.py', 'model.py', 'diffusion.py', 'manifest.csv', 'inference.json'):
            shutil.copy2(second / name, folder / name)
        shutil.copy2(template, folder / 'generate.py')
        torch.save(state, folder / 'weights.pt')
        expanded_record.update(parameters=parameters, weights_sha256=sha256(folder / 'weights.pt'),
                               mixture_members=[dict(checkpoint_sha256=r['checkpoint_sha256'],
                                                     manifest_sha256=r['manifest_sha256'],
                                                     weights_sha256=r['weights_sha256'])
                                                for r in (original_record, expanded_record)], exported=timestamp())
        expanded_record['inference_source_sha256']['generate.py'] = sha256(folder / 'generate.py')
        write_json(folder / 'provenance.json', expanded_record)
        check_files(folder)
        folder.rename(export)


def main():
    name = 'v007_v008_distribution_mixture'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    expanded_path = Path('results/experiments/v008_coverage32k/screen.json')
    original, expanded = read(REFERENCE)['candidate'], read(expanded_path)['candidate']
    original_result = validated_result(original['evidence'][0]['path'], 0, original['exported'])
    expanded_result = validated_result(expanded['evidence'][0]['path'], 0, expanded['exported'])
    if score_results([original_result]) is None or score_results([expanded_result]) is None:
        raise ValueError('Both mixture members require valid full official evidence')
    checkpoint = verified_checkpoint(RUN, 32000, sha256(MANIFEST))
    template = Path(__file__).with_name('coverage_mixture_generate.py')
    paths = (Path(__file__), template, PLAN, REFERENCE, expanded_path, MANIFEST, checkpoint)
    spec = dict(inputs={str(p): sha256(p) for p in paths}, seed=0, n_per_class=1000,
                source_exports=[r['export_hashes'] for r in (original, expanded)],
                rule='Fixed equal mixture; one private seeded model choice per output position. '
                     'Keep the exact checkpoints, both internal samplers and the complete union training manifest.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen mixture inputs changed')
    verify_starter()
    if not export.exists():
        build(export, original, expanded, template)
    preflight = out / 'offline_cuda_smoke.json'
    if not preflight.exists():
        subprocess.run([sys.executable, '-m', 'src.release.check', '--submission', str(export), '--device',
                        'cuda', '--counts', '1,2,20,50', '--out', str(preflight)], check=True)
    prefix_path = out / 'prefix_cuda.json'
    if not prefix_path.exists():
        api = importlib.import_module('.'.join((*export.parts, 'generate')))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = api.load_model(torch.device('cuda'))
        labels = torch.arange(35, device='cuda') % 2
        full = api.generate(model, labels, 17)
        if any(not torch.equal(full[:count], api.generate(model, labels[:count], 17)) for count in (1, 7)):
            raise ValueError('Mixture routing changed the output prefix')
        write_json(prefix_path, dict(device='cuda', label_count=35, prefixes=[1, 7],
                                    passed=True, weights_sha256=sha256(export / 'weights.pt')))
    screen = evaluate(checkpoint, MANIFEST, export, out, weights='ema', steps=50, guidance=3.25)
    advances = quality_improved({'score': score_results([original_result])}, screen)
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen, advances=advances))
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)


if __name__ == '__main__':
    main()
