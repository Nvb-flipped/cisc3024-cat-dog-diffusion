"""Check whether a modest guidance reduction rescues the 36k copy-rate failure."""

from pathlib import Path

from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    name = 'v009_extension36k_cfg3'
    manifest = Path('manifests/v009.csv')
    parent = Path('results/experiments/v009_extension36k/screen.json')
    reference = Path('results/experiments/v007_class_annealing010/selection.json')
    candidate, fallback = read(parent)['candidate'], read(reference)['candidate']
    checkpoint = Path(candidate['checkpoint'])
    if sha256(checkpoint) != candidate['checkpoint_sha256']:
        raise ValueError('Expected exact v009 36k weights')
    folder, export = Path('results/experiments') / name, Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    sampler = dict(weights='ema', steps=50, guidance=3.0)
    options = dict(pag_scale=.15, score_sharpness=1.03, score_sigma=.5, class_noise_max=.10)
    paths = (Path(__file__), manifest, parent, reference, checkpoint,
             *map(Path, ('submission/model.py', 'submission/diffusion.py',
                         'submission/generate.py', 'src/release/export.py', 'src/evaluation/run.py')))
    spec = dict(inputs={str(p): sha256(p) for p in paths}, sampler=sampler | options, seed=0,
                n_per_class=1000, rule='Change only CFG from 3.25 to 3.0 on the one-copy-over-cap checkpoint.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen guidance comparison inputs changed')
    verify_starter()
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **spec['sampler'])
    expected = read(Path(candidate['exported']) / 'inference.json')
    expected['sampling']['guidance'] = 3.
    if read(export / 'inference.json') != expected:
        raise ValueError('Expected guidance to be the only configuration change')
    if not (folder / 'interface.json').exists():
        write_json(folder / 'interface.json', verify_interface(candidate['exported'], export) | {'solver': 'ddim'})
    anchor = validated_result(fallback['evidence'][0]['path'], 0, fallback['exported'])
    result = evaluate(checkpoint, manifest, export, folder, **sampler)
    improves = quality_improved({'score': score_results([anchor])}, result)
    write_json(folder / 'screen.json', dict(plan_sha256=sha256(plan), candidate=result, advances=improves))
    print({'seed': 0, 'marks': result['evidence'][0]['marks'], 'advances': improves}, flush=True)


if __name__ == '__main__':
    main()
