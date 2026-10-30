"""Precompute full-pool proposal metadata; accept no images and make no human decisions."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from src.common import ROOT, timestamp, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    status_path = ROOT / 'results/curation/overnight_status.json'
    commands = [
        [sys.executable, '-m', 'src.curation.build', '--config', 'configs/curation.yaml',
         '--limit', '76250', '--out', 'results/curation/full', '--reuse', 'results/curation/pilot'],
        [sys.executable, '-m', 'src.curation.photo_screen', '--config', 'configs/photo_screen_full.yaml'],
        [sys.executable, '-m', 'src.curation.duplicates', '--candidates', 'results/curation/full/candidates.csv',
         '--out', 'results/curation/full/duplicate_candidates.csv'],
    ]
    if args.resume:
        commands[0].append('--resume')
    start = time.perf_counter()
    for index, command in enumerate(commands):
        completed = [ROOT / 'results/curation/full/summary.json',
                     ROOT / 'results/curation/photo_screen_full/experiment.json',
                     ROOT / 'results/curation/full/duplicate_candidates.json'][index]
        if args.resume and completed.exists():
            continue
        write_json(status_path, {'state': 'running', 'stage': index + 1, 'command': command,
                   'updated': timestamp(), 'pid': os.getpid(), 'human_decisions': 'none'})
        log_path = ROOT / f'logs/curation_stage_{index + 1}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Starting stage {index + 1}: {" ".join(command)}; log={log_path}', flush=True)
        with log_path.open('a', encoding='utf-8') as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            write_json(status_path, {'state': 'failed', 'stage': index + 1, 'returncode': process.returncode,
                       'log': str(log_path), 'updated': timestamp(), 'command': command})
            raise SystemExit(process.returncode)
    write_json(status_path, {'state': 'complete', 'updated': timestamp(),
               'seconds': time.perf_counter() - start, 'human_decisions': 'none',
               'next': 'Student calibration, original-versus-corrected quality checks and reviewed manifest freeze.'})


if __name__ == '__main__':
    main()
