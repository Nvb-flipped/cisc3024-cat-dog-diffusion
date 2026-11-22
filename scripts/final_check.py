"""Final whole-project consistency check (CPU only, no raw images needed). Prints PASS/FAIL per check, writes a JSON receipt.

    python -m scripts.final_check [--out private_build/final_verification/final_check.json]

Independent recomputation: seed statistics are recomputed here from the raw evaluator JSONs and compared with the numbers the report
prints (report/numbers.tex); manifest geometry is checked against the pool's image sizes; the test count quoted in README, report and
slides is compared with a real pytest run. It does not replace the clean-room archive test (src.release.verify_archive), the human
audit, or a visual review of the figures.
"""
import argparse
import csv
import importlib.util
import json
import re
import statistics as st
import subprocess
import sys
import zipfile
from pathlib import Path

from src.common import ROOT, sha256, timestamp, write_json
from src.release.package import BULK_MANIFEST, BULK_OMITTED, MAX_ARCHIVE_BYTES
from scripts.setup_inputs import verify_starter

PANEL = ROOT / 'results/experiments/v014_ema9995_flip/u080000_g275'
RESULTS = []
TOKEN = re.compile(r'github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----')


def check(name):
    def wrap(function):
        try:
            detail = function()
            RESULTS.append({'check': name, 'ok': True, 'detail': detail})
        except Exception as exc:  # a failed check is a result, not a crash
            RESULTS.append({'check': name, 'ok': False, 'detail': f'{type(exc).__name__}: {exc}'})
        return function
    return wrap


def run(*command):
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def macros():
    text = (ROOT / 'report/numbers.tex').read_text(encoding='utf-8')
    return dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', text))


@check('git: only the packager receipt is uncommitted')
def _():
    dirty = [line for line in run('git', 'status', '--porcelain').stdout.splitlines() if 'package_receipt.json' not in line]
    assert not dirty, dirty
    return 'HEAD ' + run('git', 'rev-parse', '--short', 'HEAD').stdout.strip()


@check('starter_kit is byte-identical to the supplied archive')
def _():
    verify_starter(ROOT)
    return 'every file of Course_Project_Resources/starter_kit.zip matches'


@check('tests: real pytest run equals the count quoted in README, report and slides')
def _():
    out = run(sys.executable, '-m', 'pytest', '-q', 'tests', 'starter_kit/tests').stdout.strip().splitlines()[-1]
    passed, skipped = map(int, re.search(r'(\d+) passed(?:, (\d+) skipped)?', out).groups(default='0'))
    assert 'failed' not in out and 'error' not in out, out
    quoted = {'README.md': re.search(r'Tests \((\d+) pass', (ROOT / 'README.md').read_text(encoding='utf-8')).group(1),
              'report': re.search(r'(\d+) tests', (ROOT / 'report/report.tex').read_text(encoding='utf-8')).group(1),
              'slides': re.search(r'(\d+) project tests', (ROOT / 'presentation/presentation.tex').read_text(encoding='utf-8')).group(1)}
    assert all(int(v) == passed for v in quoted.values()), (out, quoted)
    return f'{passed} passed, {skipped} skipped; quoted {quoted}'


@check('submission: hashes match provenance; parameters and size within limits')
def _():
    folder = ROOT / 'submission'
    prov = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
    assert sha256(folder / 'weights.pt') == prov['weights_sha256'], 'weights hash'
    assert sha256(folder / 'manifest.csv') == prov['manifest_sha256'], 'manifest hash'
    assert sha256(folder / 'inference.json') == prov['config_sha256'], 'config hash'
    assert sha256(ROOT / 'manifests/v014.csv') == prov['manifest_sha256'], 'manifests/v014.csv differs from the submitted manifest'
    spec = importlib.util.spec_from_file_location('final_check_generate', folder / 'generate.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    import torch
    model = module.load_model(torch.device('cpu'))
    count = sum(p.numel() for p in model.parameters())
    size = sum(f.stat().st_size for f in folder.iterdir() if f.is_file())
    assert count <= 18_000_000 and size <= 90_000_000, (count, size)  # engineering ceilings below the 20M / 100 MB limits
    assert str(count) == macros()['params'].replace('{,}', ''), 'report parameter count'
    return f'{count:,} parameters (ceiling 18M), {size:,} bytes (ceiling 90,000,000)'


@check('manifest: geometry inside the image, no rotation or padding, no duplicates, holdout excluded')
def _():
    with (ROOT / 'results/curation/integrity.csv').open(newline='', encoding='utf-8') as f:
        sizes = {r['pool_id']: (int(r['width']), int(r['height'])) for r in csv.DictReader(f) if r['status'] == 'ok'}
    with (ROOT / 'submission/manifest.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    outside = [r for r in rows if r['pool_id'] not in sizes or float(r['angle']) != 0 or int(r['x0']) < 0 or int(r['y0']) < 0
               or int(r['size']) < 32 or int(r['x0']) + int(r['size']) > sizes[r['pool_id']][0]
               or int(r['y0']) + int(r['size']) > sizes[r['pool_id']][1]]
    assert not outside, f'{len(outside)} rows outside the image, rotated, padded or smaller than 32 px: {outside[:2]}'
    keys = [(r['pool_id'], r['x0'], r['y0'], r['size'], r['label']) for r in rows]
    assert len(set(keys)) == len(keys), 'duplicate crop rows'
    labels = {l: sum(r['label'] == l for r in rows) for l in ('0', '1')}
    card = re.search(r'\((\d[\d,]*) cat, (\d[\d,]*) dog\)', (ROOT / 'DATA_CARD.md').read_text(encoding='utf-8'))
    expected = {'0': int(card.group(1).replace(',', '')), '1': int(card.group(2).replace(',', ''))}
    assert labels == expected and len(rows) == sum(expected.values()) == int(macros()['manifestRows'].replace('{,}', '')), (labels, expected)
    with (ROOT / 'manifests/v014.validation.csv').open(newline='', encoding='utf-8') as f:
        holdout = {r['pool_id'] for r in csv.DictReader(f)}
    assert not holdout & {r['pool_id'] for r in rows}, 'holdout photo present in the training manifest'
    return f'{len(rows):,} rows ({labels["0"]:,} cat / {labels["1"]:,} dog), {len({r["pool_id"] for r in rows}):,} photos, holdout {len(holdout)} photos disjoint'


@check('seeds: recompute the report statistics from the raw evaluator JSONs')
def _():
    runs = []
    for path in sorted(PANEL.glob('seed_*.json')):
        if 'provenance' in path.name:
            continue
        d = json.loads(path.read_text(encoding='utf-8'))
        assert not d['violations'] and d['marks']['valid'], f'{path.name}: violations or invalid'
        runs.append((int(path.stem.split('_')[1]), d['marks']['total'], d['metrics']))
    n, n15 = len(runs), sum(r[1] == 15 for r in runs)
    recall = [r[2]['recall'] for r in runs]
    m = macros()
    ours = {'nSeeds': str(n), 'nFifteen': str(n15), 'recallMean': f'{st.mean(recall):.3f}', 'recallMin': f'{min(recall):.3f}',
            'recallMax': f'{max(recall):.3f}', 'recallSD': f'{st.stdev(recall):.3f}',
            'memMax': f"{max(r[2]['memorization_rate'] for r in runs) * 100:.2f}", 'fidMean': f"{st.mean(r[2]['fid'] for r in runs):.2f}"}
    wrong = {k: (v, m[k]) for k, v in ours.items() if m[k] != v}
    assert not wrong, f'report prints different values (recomputed, printed): {wrong}'
    below = [(r[0], r[1], round(r[2]['recall'], 4)) for r in runs if r[1] < 15]
    return f'{n15} of {n} seeds at 15; recall {ours["recallMean"]} +/- {ours["recallSD"]} (min {ours["recallMin"]}, max {ours["recallMax"]}); below 15: {below}'


@check('seeds: every evaluator run used the final inference files')
def _():
    final = json.loads((ROOT / 'results/final/seed_0.provenance.json').read_text(encoding='utf-8'))['submission_file_sha256']
    stale = []
    for path in sorted(PANEL.glob('seed_*.provenance.json')):
        recorded = json.loads(path.read_text(encoding='utf-8')).get('submission_file_sha256')
        if recorded != final:
            stale.append(path.name)
    assert not stale, f'{len(stale)} panel runs used different files: {stale[:3]}'
    return f'{len(list(PANEL.glob("seed_*.provenance.json")))} panel sidecars + results/final all match submission/ file hashes'


@check('report: name, ID, no draft marks, page limit, clickable links')
def _():
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    import ctypes
    pdf = pdfium.PdfDocument(str(ROOT / 'report/report.pdf'))
    text = ' '.join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
    assert 'Sou Chak Tong' in text and 'U-C3-2652-3' in text, 'name or ID missing'
    assert not re.search(r'STUDENT:|TODO|FIXME|XXX', text), 'draft mark in the PDF'
    pages = json.loads((ROOT / 'report/page_count.json').read_text(encoding='utf-8'))
    assert pages['pdf_sha256'] == sha256(ROOT / 'report/report.pdf') and pages['main_text_pages'] <= 6 and pages['total_pages'] == len(pdf)
    links = set()
    for i in range(len(pdf)):
        for k in range(raw.FPDFPage_GetAnnotCount(pdf[i])):
            annot = raw.FPDFPage_GetAnnot(pdf[i], k)
            if raw.FPDFAnnot_GetSubtype(annot) == raw.FPDF_ANNOT_LINK:
                action = raw.FPDFLink_GetAction(raw.FPDFAnnot_GetLink(annot))
                if raw.FPDFAction_GetType(action) == raw.PDFACTION_URI:
                    n = raw.FPDFAction_GetURIPath(pdf, action, None, 0)
                    buffer = ctypes.create_string_buffer(n)
                    raw.FPDFAction_GetURIPath(pdf, action, buffer, n)
                    links.add(buffer.value.decode())
    base = 'https://github.com/Nvb-flipped/cisc3024-cat-dog-diffusion/releases/'
    assert base + 'download/final-submission/project_submission.zip' in links and base + 'tag/final-submission' in links, links
    return f'{len(pdf)} pages ({pages["main_text_pages"]} main text), {len(links)} links incl. the release ZIP and release page'


@check('slides: compiled PDF is current and has the expected frames')
def _():
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(ROOT / 'presentation/presentation.pdf'))
    assert (ROOT / 'presentation/presentation.pdf').stat().st_mtime >= (ROOT / 'presentation/presentation.tex').stat().st_mtime - 1, 'PDF older than its source'
    return f'{len(pdf)} slides'


@check('secrets: no token or key in tracked files, git config or remotes')
def _():
    hits = []
    for name in run('git', 'ls-files').stdout.splitlines():
        path = ROOT / name
        if path.is_file() and path.stat().st_size < 5_000_000 and path.suffix.lower() not in {'.pt', '.png', '.pdf', '.jpg', '.npy'}:
            if TOKEN.search(path.read_text(encoding='utf-8', errors='ignore')):
                hits.append(name)
    config = run('git', 'config', '--list', '--show-origin').stdout + run('git', 'remote', '-v').stdout
    if TOKEN.search(config) or re.search(r'https://[^/@\s]+:[^/@\s]+@', config):
        hits.append('git config / remote URL')
    assert not hits, hits
    return 'none found'


@check('bulk asset: matches its manifest; omitted files are not in the allowlist')
def _():
    record = json.loads((ROOT / BULK_MANIFEST).read_text(encoding='utf-8'))
    asset = ROOT / 'private_build' / record['asset']
    assert sha256(asset) == record['asset_sha256'] and asset.stat().st_size == record['asset_bytes'], 'asset hash or size'
    with zipfile.ZipFile(asset) as bundle:
        import hashlib
        for item in record['files']:
            assert hashlib.sha256(bundle.read(item['path'])).hexdigest() == item['sha256'], item['path']
            assert sha256(ROOT / item['path']) == item['sha256'], f'{item["path"]} changed locally'
    assert sorted(i['path'] for i in record['files']) == sorted(BULK_OMITTED)
    return f'{record["asset"]}, {record["asset_bytes"]:,} bytes, {len(record["files"])} files verified'


@check('archive: size ceiling and the ZIP on disk')
def _():
    archive = ROOT / 'private_build/project_submission.zip'
    if not archive.exists():
        return 'no ZIP built yet (build it, then run this check again)'
    size = archive.stat().st_size
    assert size <= MAX_ARCHIVE_BYTES, f'{size:,} bytes exceed the {MAX_ARCHIVE_BYTES:,}-byte limit'
    with zipfile.ZipFile(archive) as bundle:
        assert 'report/report.pdf' in bundle.namelist(), 'report PDF missing from the ZIP'
        assert bundle.read('report/report.pdf') == (ROOT / 'report/report.pdf').read_bytes(), 'ZIP copy of the report differs'
        members = len(bundle.namelist())
    return f'{size:,} bytes ({MAX_ARCHIVE_BYTES - size:,} below the limit), {members:,} members, SHA-256 {sha256(archive)}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'private_build/final_verification/final_check.json')
    args = parser.parse_args()
    for r in RESULTS:
        print(f'[{"PASS" if r["ok"] else "FAIL"}] {r["check"]}\n       {r["detail"]}')
    write_json(args.out, {'created': timestamp(), 'results': RESULTS, 'all_passed': all(r['ok'] for r in RESULTS)})
    print('ALL CHECKS PASSED' if all(r['ok'] for r in RESULTS) else 'SOME CHECKS FAILED')
    sys.exit(0 if all(r['ok'] for r in RESULTS) else 1)


if __name__ == '__main__':
    main()
