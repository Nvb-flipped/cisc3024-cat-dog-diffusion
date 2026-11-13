"""Export exactly one trained checkpoint and its exact immutable manifest."""
import argparse
import math
import shutil
from pathlib import Path
import torch
from src.common import ROOT, sha256, timestamp, write_json
from submission.model import ConditionalUNet

INFERENCE_FILES = ['__init__.py', 'generate.py', 'model.py', 'diffusion.py']


def export_checkpoint(checkpoint, manifest, out, *, weights='ema', steps=None, guidance=None,
                      microbatch=None, execution_batch=None, start_timestep=None,
                      guidance_min_timestep=None, guidance_max_timestep=None, eta=None, solver=None,
                      guidance_rescale=None):
    checkpoint, manifest, out = Path(checkpoint), Path(manifest), Path(out)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('diagnostic_only'):
        raise ValueError('Diagnostic weights must never become a baseline/final submission')
    digest = sha256(manifest)
    if state['manifest_sha256'] != digest:
        raise ValueError('Checkpoint/manifest mismatch; final model must list all and only its training rows')
    model = ConditionalUNet(**state['config']['model'])
    model.load_state_dict(state[weights], strict=True)
    if any(not torch.isfinite(value).all() for value in state[weights].values()):
        raise ValueError('Checkpoint contains nonfinite weights')
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > 18_000_000:
        raise ValueError('Model exceeds 18M engineering ceiling')
    if (out / 'weights.pt').exists() or (out / 'manifest.csv').exists():
        raise FileExistsError('Candidate export already exists; use a new immutable output directory')
    sampling = dict(state['config']['sampling']) | state['config']['diffusion']
    prediction_type = state['config'].get('training', {}).get('prediction_type', 'epsilon')
    if prediction_type not in ('epsilon', 'velocity', 'edm'):
        raise ValueError('Unsupported checkpoint denoising target')
    if sampling.get('prediction_type', prediction_type) != prediction_type:
        raise ValueError('Sampling denoising target differs from the trained target')
    if prediction_type != 'epsilon':
        sampling['prediction_type'] = prediction_type
    overrides = [('steps', steps), ('guidance', guidance), ('microbatch', microbatch),
                 ('execution_batch', execution_batch), ('start_timestep', start_timestep),
                 ('guidance_min_timestep', guidance_min_timestep),
                 ('guidance_max_timestep', guidance_max_timestep), ('eta', eta), ('solver', solver),
                 ('guidance_rescale', guidance_rescale)]
    for key, value in overrides:
        if value is not None:
            sampling[key] = value
    sampling.setdefault('execution_batch', 32)
    integers = ['steps', 'timesteps', 'microbatch', 'execution_batch']
    if any(not isinstance(sampling[key], int) or isinstance(sampling[key], bool) for key in integers) or \
       not 1 <= sampling['steps'] <= sampling['timesteps'] or sampling['timesteps'] < 2 or \
       sampling['microbatch'] < 1 or sampling['execution_batch'] < 1 or \
       not 0 <= sampling.get('eta', 0.0) <= 1 or \
       not math.isfinite(sampling['guidance']) or \
       not math.isfinite(sampling['cosine_offset']) or sampling['cosine_offset'] < 0:
        raise ValueError('Invalid sampler configuration')
    final_timestep = sampling['timesteps'] - 1
    start = sampling.get('start_timestep', final_timestep)
    guidance_min = sampling.get('guidance_min_timestep', 0)
    guidance_max = sampling.get('guidance_max_timestep', final_timestep)
    if any(type(value) is not int for value in (start, guidance_min, guidance_max)) or \
       not sampling['steps'] - 1 <= start <= final_timestep or \
       not 0 <= guidance_min <= guidance_max <= final_timestep:
        raise ValueError('Invalid sampler timestep bounds')
    if prediction_type == 'edm' and (sampling['steps'] < 2 or sampling.get('eta', 0) or
                                     'start_timestep' in sampling or guidance_min or
                                     'guidance_max_timestep' in sampling):
        raise ValueError('Invalid EDM sampler configuration')
    solver = sampling.get('solver', 'ddim')
    if solver not in ('ddim', 'dpmpp_2m') or (solver == 'dpmpp_2m' and (
            prediction_type != 'epsilon' or sampling.get('eta', 0) or sampling['steps'] < 2)):
        raise ValueError('DPM-Solver++ requires deterministic epsilon sampling with at least two steps')
    rescale = sampling.get('guidance_rescale', 0)
    if not 0 <= rescale <= 1 or (rescale and prediction_type != 'epsilon'):
        raise ValueError('Guidance rescale requires epsilon prediction and a strength in [0,1]')
    inference_files = INFERENCE_FILES + ['edm.py'] if prediction_type == 'edm' else INFERENCE_FILES
    out.mkdir(parents=True, exist_ok=True)
    for name in inference_files:
        source = ROOT / 'submission' / name
        dest = out / name
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
    shutil.copy2(manifest, out / 'manifest.csv')
    torch.save({name: tensor.detach().cpu() for name, tensor in state[weights].items()}, out / 'weights.pt')
    write_json(out / 'inference.json', {'model': state['config']['model'], 'sampling': sampling})
    write_json(out / 'provenance.json', {'checkpoint_sha256': sha256(checkpoint),
        'manifest_sha256': digest, 'weights_sha256': sha256(out / 'weights.pt'),
        'config_sha256': sha256(out / 'inference.json'), 'weights_variant': weights,
        'training_update': state['update'], 'training_seed': state['seed'],
        'pool_content_sha256': state.get('stats', {}).get('pool_content_sha256'),
        'training_source_code_sha256': state.get('source_code_sha256'),
        'data_review_provenance_sha256': state.get('data_review_provenance_sha256'),
        'training_segment_id': state.get('segment_id'),
        'parameters': parameters, 'exported': timestamp(),
        'inference_source_sha256': {name: sha256(out / name) for name in inference_files}})
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    if size > 90_000_000:
        raise ValueError(f'Export exceeds internal 90,000,000-byte ceiling: {size}')
    return {'path': str(out), 'parameters': parameters, 'bytes': size,
            'manifest_sha256': digest, 'weights': weights, 'sampling': sampling}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path, help='Selected update_XXXXXX.pt file (no implicit selection)')
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--out', default='submission', type=Path)
    p.add_argument('--weights', choices=['ema', 'model'], default='ema')
    p.add_argument('--steps', type=int)
    p.add_argument('--guidance', type=float)
    p.add_argument('--microbatch', type=int)
    p.add_argument('--execution-batch', type=int)
    p.add_argument('--start-timestep', type=int)
    p.add_argument('--guidance-min-timestep', type=int)
    p.add_argument('--guidance-max-timestep', type=int)
    p.add_argument('--eta', type=float)
    p.add_argument('--solver', choices=('ddim', 'dpmpp_2m'))
    p.add_argument('--guidance-rescale', type=float)
    args = p.parse_args()
    print(export_checkpoint(args.run, args.manifest, args.out, weights=args.weights,
                            steps=args.steps, guidance=args.guidance, microbatch=args.microbatch,
                            execution_batch=args.execution_batch, start_timestep=args.start_timestep,
                            guidance_min_timestep=args.guidance_min_timestep,
                            guidance_max_timestep=args.guidance_max_timestep, eta=args.eta, solver=args.solver,
                            guidance_rescale=args.guidance_rescale))


if __name__ == '__main__':
    main()
