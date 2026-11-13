"""Verify EDM preconditioning on a fixed, reviewed two-crop batch."""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from src.common import ROOT, environment, sha256, timestamp, write_json
from src.training.data import load_manifest_data
from src.training.diagnostics import OVERFIT_UPDATES
from src.training.lineage import reviewed_manifest_provenance
from submission.diffusion import cosine_alpha_bar
from submission.edm import (LOG_SIGMA_MEAN, LOG_SIGMA_STD, SIGMA_DATA,
                            noise_condition, training_pair)
from submission.model import ConditionalUNet


def run(manifest, out):
    if out.exists():
        raise FileExistsError(f'Diagnostic directory already exists: {out}')
    provenance = reviewed_manifest_provenance(manifest)
    torch.set_num_threads(8)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    clean, labels, manifest_hash, stats = load_manifest_data(manifest)
    rows = [(labels == species).nonzero(as_tuple=True)[0][0].item() for species in (0, 1)]
    clean, labels = clean[rows].cuda(), labels[rows].cuda()
    model_config = dict(channels=[16, 32, 48, 64], blocks_per_level=1,
                        conditioning_dim=64, attention_heads=4, dropout=0)
    model = ConditionalUNet(**model_config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    generator = torch.Generator(device='cuda').manual_seed(29)
    probe_generator = torch.Generator(device='cpu').manual_seed(43)
    probe_noise = torch.randn(clean.shape, generator=probe_generator).cuda()
    probe_alpha = cosine_alpha_bar()[[250, 750]].cuda()
    probe_sigma = ((1 - probe_alpha) / probe_alpha).sqrt()[:, None, None, None]
    probe_input, probe_target, probe_noisy, probe_skip, probe_scale_out = training_pair(
        clean, probe_noise, probe_sigma)

    # Check across both the training noise regime and inference endpoints.
    oracle_sigma = torch.tensor([0.002, 0.1, 0.5, 2.0, 80.0], device='cuda',
                                dtype=torch.float64)[:, None, None, None]
    _, oracle_target, oracle_noisy, oracle_skip, oracle_out = training_pair(
        clean[:1].double(), probe_noise[:1].double(), oracle_sigma)
    torch.testing.assert_close(oracle_skip * oracle_noisy + oracle_out * oracle_target,
                               clean[:1].double().expand_as(oracle_target), atol=1e-12, rtol=1e-12)
    prediction = oracle_target + 0.25
    residual_error = oracle_skip * oracle_noisy + oracle_out * prediction - clean[:1].double()
    torch.testing.assert_close(residual_error.square() / oracle_out.square(),
                               (prediction - oracle_target).square(), atol=1e-12, rtol=1e-12)

    @torch.no_grad()
    def measure():
        model.eval()
        prediction = model(probe_input, noise_condition(probe_sigma), labels)
        reconstruction = probe_skip * probe_noisy + probe_scale_out * prediction
        return {'probe_mse': F.mse_loss(prediction, probe_target).item(),
                'reconstruction_mse': F.mse_loss(reconstruction, clean).item()}

    initial = measure()
    started = time.perf_counter()
    progress = []
    for update in range(1, OVERFIT_UPDATES + 1):
        model.train()
        log_sigma = torch.randn((len(clean), 1, 1, 1), generator=generator, device='cuda')
        sigma = (log_sigma * LOG_SIGMA_STD + LOG_SIGMA_MEAN).exp()
        noise = torch.randn(clean.shape, generator=generator, device='cuda')
        model_input, target, _, _, _ = training_pair(clean, noise, sigma)
        loss = F.mse_loss(model(model_input, noise_condition(sigma), labels), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 100 == 0:
            row = {'update': update, 'loss': loss.item(), **measure()}
            progress.append(row)
            print(row, flush=True)

    final = progress[-1]
    reduction = 1 - final['probe_mse'] / initial['probe_mse']
    passed = reduction >= 0.80 and final['reconstruction_mse'] < initial['reconstruction_mse']
    evidence = {'diagnostic_only': True, 'manifest_sha256': manifest_hash,
                'data_review_provenance_sha256': provenance, 'row_indices_zero_based': rows,
                'pool_content_sha256': stats['pool_content_sha256'], 'model': model_config,
                'command': [sys.executable, '-m', 'src.training.edm_diagnostic',
                            '--manifest', str(manifest), '--out', str(out)],
                'source_sha256': {name: sha256(ROOT / name) for name in (
                    'src/training/edm_diagnostic.py', 'src/training/data.py',
                    'submission/model.py', 'submission/diffusion.py', 'submission/edm.py')},
                'training_seed': 17, 'noise_seed': 29, 'probe_seed': 43,
                'sigma_data': SIGMA_DATA, 'log_sigma_mean': LOG_SIGMA_MEAN,
                'log_sigma_std': LOG_SIGMA_STD, 'probe_sigmas': probe_sigma.flatten().tolist(),
                'updates': OVERFIT_UPDATES, 'required_reduction': 0.80,
                'initial': initial, 'final': final, 'reduction': reduction, 'passed': passed,
                'oracle_reconstruction_and_loss_parity': True, 'progress': progress,
                'seconds': time.perf_counter() - started, 'environment': environment(),
                'completed': timestamp(), 'weights_retained': False}
    write_json(out / 'result.json', evidence)
    if not passed:
        raise RuntimeError('Fixed 2,000-update EDM diagnostic failed the declared threshold')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.out)
