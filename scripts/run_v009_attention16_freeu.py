"""Transfer the verified FreeU feature profile to the matched attention checkpoint."""

from pathlib import Path
import sys

from scripts.run_v002_dpmpp import read, verify_interface
from scripts.setup_inputs import verify_starter
from src.common import sha256, write_json
from src.evaluation.select import validated_result
from src.evaluation.sweep import score_results
from src.release.export import export_checkpoint
from src.training.optimize import evaluate, quality_improved


def main():
    name = 'v009_attention16_freeu'
    out = Path('results/experiments') / name
    export = Path('private_build') / name
    plan = Path('results/plans') / f'{name}.json'
    manifest = Path('manifests/v009.csv')
    matched_path = Path('results/experiments/v009_attention16_32k/screen.json')
    lead_path = Path('results/experiments/v007_freeu_cfg310/selection.json')
    matched, lead = (read(p)['candidate'] for p in (matched_path, lead_path))
    matched_export = Path(matched['exported'])
    checkpoint = Path(matched['checkpoint'])
    if sha256(checkpoint) != matched['checkpoint_sha256']:
        raise ValueError('Expected the exact 16x16 attention checkpoint')
    runtime = dict(weights='ema', steps=50, guidance=3.25)
    settings = runtime | dict(pag_scale=.15, score_sharpness=1.03, score_sigma=.5,
                              class_noise_max=.10, freeu=True)
    inputs = (Path(__file__), matched_path, lead_path, manifest, checkpoint,
              *map(Path, ('submission/model.py', 'submission/diffusion.py', 'submission/generate.py',
                          'src/release/export.py', 'src/evaluation/run.py', 'src/training/optimize.py')))
    spec = dict(inputs={str(p): sha256(p) for p in inputs}, sampler=settings, seeds=[0, 101, 202],
                n_per_class=1000, command=[sys.executable, '-u', '-m', 'scripts.run_v009_attention16_freeu'],
                rule='Transfer only the established decoder feature reweight. Advance after valid improvement '
                     'over both unmodified attention and the three-seed incumbent on the matched screen seed.')
    if not plan.exists():
        write_json(plan, spec)
    if read(plan) != spec:
        raise ValueError('Frozen attention FreeU inputs changed')
    verify_starter()
    if not export.exists():
        export_checkpoint(checkpoint, manifest, export, **settings)
    expected = read(matched_export / 'inference.json')
    expected['model']['freeu'] = True
    if (read(export / 'inference.json') != expected or
            any(sha256(export / item) != matched['export_hashes'][item]
                for item in ('weights.pt', 'manifest.csv', 'model.py', 'generate.py', 'diffusion.py'))):
        raise ValueError('The FreeU switch must be the only inference configuration change')
    if not (out / 'interface.json').exists():
        checks = verify_interface(export, export)
        write_json(out / 'interface.json', checks | {'solver': 'ddim', 'freeu': True})
    paired = [validated_result(item['evidence'][0]['path'], 0, item['exported']) for item in (matched, lead)]
    screen = evaluate(checkpoint, manifest, export, out, **runtime)
    beats_matched, beats_lead = [quality_improved({'score': score_results([row])}, screen) for row in paired]
    advances = beats_matched and beats_lead
    write_json(out / 'screen.json', dict(plan_sha256=sha256(plan), candidate=screen,
                                        improves_matched=beats_matched, advances=advances))
    print({'seed': 0, 'marks': screen['evidence'][0]['marks'], 'advances': advances}, flush=True)
    if advances:
        selected = evaluate(checkpoint, manifest, export, out, seeds=spec['seeds'], **runtime)
        write_json(out / 'selection.json', dict(screen_sha256=sha256(out / 'screen.json'), candidate=selected,
                   improves_reference=quality_improved(lead, selected)))
        print({'selection_marks': [row['marks'] for row in selected['evidence']]}, flush=True)


if __name__ == '__main__':
    main()
