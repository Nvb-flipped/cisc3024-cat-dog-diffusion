"""Build report/report.pdf with Tectonic and write report/page_count.json (main-text pages, total pages, PDF hash).

    python -m scripts.build_report_pdf

Refuses to build while a red \\student{...} placeholder call is still in report.tex. The main text ends at the label `mainend`
(just before the references and the appendix).
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.common import ROOT, sha256, timestamp, write_json

REPORT = ROOT / 'report'


def main():
    source = (REPORT / 'report.tex').read_text(encoding='utf-8')
    leftover = [line for line in source.splitlines() if '\\student{' in line and 'newcommand' not in line]
    if leftover:
        raise SystemExit(f'{len(leftover)} placeholder(s) \\student{{...}} remain in report.tex; fill them first.')
    tectonic = ROOT / '.venvs/tools/bin/tectonic'
    out = Path(tempfile.mkdtemp(prefix='report_build_'))
    env = os.environ | {'TECTONIC_CACHE_DIR': str(ROOT / 'private_build/report_tex_cache_linux')}
    subprocess.run([str(tectonic), '-X', 'compile', 'report.tex', '--outdir', str(out), '--keep-intermediates', '--keep-logs'],
                   cwd=REPORT, env=env, check=True, capture_output=True)
    main_pages = int(re.search(r'\\newlabel\{mainend\}\{\{[^}]*\}\{(\d+)\}', (out / 'report.aux').read_text()).group(1))
    total = int(subprocess.run([str(ROOT / '.venvs/tools/bin/python'), '-c',
                                'import sys, pypdfium2; print(len(pypdfium2.PdfDocument(sys.argv[1])))', str(out / 'report.pdf')],
                               check=True, capture_output=True, text=True).stdout)
    shutil.copy2(out / 'report.pdf', REPORT / 'report.pdf')
    record = {'pdf_sha256': sha256(REPORT / 'report.pdf'), 'main_text_pages': main_pages, 'total_pages': total,
              'main_text_limit': 6, 'built': timestamp(), 'engine': 'Tectonic 0.17.0'}
    write_json(REPORT / 'page_count.json', record)
    shutil.rmtree(out, ignore_errors=True)
    print(json.dumps(record, indent=1))
    if main_pages > 6:
        raise SystemExit('main text exceeds six pages')


if __name__ == '__main__':
    main()
