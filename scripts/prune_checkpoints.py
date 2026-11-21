"""Prune intermediate training checkpoints of the Linux-era runs (v010-v014) without losing evidence.

    python -m scripts.prune_checkpoints                 # dry run (default): prints the plan, deletes nothing
    python -m scripts.prune_checkpoints --apply         # deletes, after writing a ledger

Per run directory it KEEPS
  * every checkpoint referenced by a recorded evaluation or export (results/experiments/<run>/uNNNNNN_*,
    private_build/<run>/uNNNNNN_*) AND every checkpoint any text record cites by path or by recorded SHA-256
    (results, private_build, README, report, configs, logs...), so every result still traces to a checkpoint file;
  * the last checkpoint of the run;
  * every N-th update for runs named with --keep-every RUN=N (extra lineage for the final model);
  * all small files (receipts *.json, loss CSV, run/completed JSON, logs).
It DELETES only the other update_*.pt files. A ledger (path, bytes, the receipt's SHA-256) is written to
results/pruning/ before anything is removed, so each deletion stays replayable and verifiable.
Directories with a file modified in the last --live-minutes minutes (a running job) are skipped entirely.
Git history is not affected: checkpoints are untracked.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from src.common import timestamp, write_json

CKPT = Path('checkpoints')
LINUX_ERA = re.compile(r'^v01[0-4]')


def referenced_updates(run):
    updates = set()
    for base in ('results/experiments', 'private_build'):
        for d in Path(base, run).glob('u[0-9][0-9][0-9][0-9][0-9][0-9]_*'):
            updates.add(int(d.name[1:7]))
    return updates


def record_references():
    """Checkpoint files that ANY record cites, by path ('checkpoints/<run>/update_N.pt') or by recorded SHA-256."""
    sha_to_file = {}
    for receipt in CKPT.glob('*/update_*.json'):
        try:
            data = json.loads(receipt.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and 'sha256' in data:
            sha_to_file[data['sha256']] = receipt.with_suffix('.pt')
    path_re = re.compile(r'checkpoints[/\\]+([\w.\-]+)[/\\]+update_(\d+)\.pt')
    hash_re = re.compile(r'\b[0-9a-f]{64}\b')
    protected = set()
    for root in ('results', 'private_build', 'README.md', 'EXECUTION_STATUS.md', 'report', 'figures', 'configs',
                 'logs', 'MIGRATION_HANDOFF.md', 'PROJECT_PLAN.md', 'REQUIREMENTS_TRACE.md'):
        base = Path(root)
        paths = [base] if base.is_file() else [q for q in base.rglob('*') if q.is_file()] if base.exists() else []
        for q in paths:
            if q.suffix not in ('.json', '.jsonl', '.csv', '.md', '.txt', '.log', '.tex', '.yaml', '.yml'):
                continue
            try:
                if q.stat().st_size > 80e6:
                    continue
                text = q.read_text(errors='ignore')
            except OSError:
                continue
            for m in path_re.finditer(text):
                protected.add(CKPT / m.group(1) / f'update_{int(m.group(2)):06d}.pt')
            protected.update(sha_to_file[h] for h in hash_re.findall(text) if h in sha_to_file)
    return protected


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def plan(args):
    backup = {}
    if args.legacy_backup_manifest:
        backup = {e['sha256']: e['archive'] for e in json.loads(Path(args.legacy_backup_manifest).read_text())['files']}
    keep_every = dict(item.split('=') for item in args.keep_every)
    protected = record_references()
    rows, ledger = [], []
    for directory in sorted(p for p in CKPT.iterdir() if p.is_dir() and (LINUX_ERA.match(p.name) or args.legacy_backup_manifest)):
        run = directory.name.removesuffix('_seed17')
        files = {int(m.group(1)): f for f in directory.glob('update_*.pt') if (m := re.search(r'update_(\d+)\.pt$', f.name))}
        if not files:
            continue
        newest = max(f.stat().st_mtime for f in directory.rglob('*') if f.is_file())
        if time.time() - newest < args.live_minutes * 60 or run in args.skip:
            rows.append((run, len(files), len(files), 0, 0.0, 'SKIPPED (live job or --skip)'))
            continue
        step = int(keep_every.get(run, 0))
        keep = (referenced_updates(run) | {max(files)} | ({u for u in files if u % step == 0} if step else set())
                | {u for u in files if files[u] in protected or files[u].resolve() in {q.resolve() for q in protected if q.exists()}})
        delete = sorted(u for u in files if u not in keep)
        legacy = not LINUX_ERA.match(directory.name)
        archives = {}
        if legacy:   # older (Windows-era) runs: delete only files whose ACTUAL hash is in the verified backup
            for u in delete:
                archives[u] = backup.get(file_sha256(files[u]))
            delete = [u for u in delete if archives[u]]
        freed = sum(files[u].stat().st_size for u in delete)
        for u in delete:
            receipt = files[u].with_suffix('.json')
            sha = json.loads(receipt.read_text())['sha256'] if receipt.exists() else None
            ledger.append({'path': str(files[u]), 'bytes': files[u].stat().st_size, 'receipt_sha256': sha,
                           **({'backup_archive': archives[u]} if legacy else {})})
        rows.append((run, len(files), len(files) - len(delete), len(delete), freed / 1e9, ''))
    return rows, ledger


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--apply', action='store_true', help='actually delete (default: dry run)')
    p.add_argument('--keep-every', nargs='*', default=[], metavar='RUN=N', help='also keep every N-th update of RUN')
    p.add_argument('--skip', nargs='*', default=[], help='run names (without _seed17) to leave untouched')
    p.add_argument('--live-minutes', type=int, default=45)
    p.add_argument('--legacy-backup-manifest', metavar='JSON', help='also prune older (non v010-v014) runs, but only files whose SHA-256 is in this backup manifest')
    args = p.parse_args()
    rows, ledger = plan(args)
    print(f"{'run':30s} {'before':>6s} {'kept':>5s} {'delete':>6s} {'freed GB':>9s}")
    for run, before, kept, delete, freed, note in rows:
        print(f'{run:30s} {before:6d} {kept:5d} {delete:6d} {freed:9.1f}  {note}')
    total = sum(r['bytes'] for r in ledger)
    print(f'\n{len(ledger)} files, {total / 1e9:.1f} GB would be deleted' + ('' if args.apply else ' (dry run: nothing deleted)'))
    if not args.apply:
        return
    out = Path('results/pruning') / f"ledger_{timestamp().replace(':', '')}.json"
    write_json(out, {'created': timestamp(), 'rule': __doc__.split('\n\n')[1], 'keep_every': args.keep_every,
                     'skipped': args.skip, 'files': ledger, 'total_bytes': total})
    print('ledger written:', out)
    for entry in ledger:
        Path(entry['path']).unlink()
    print('deleted', len(ledger), 'files')


if __name__ == '__main__':
    main()
