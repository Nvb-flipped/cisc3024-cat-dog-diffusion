"""Bind the LaTeX main-text boundary and physical page count to a report PDF."""

import argparse
import re
import subprocess
from pathlib import Path

from src.common import sha256, write_json


def verify(pdf, aux, pdfinfo='pdfinfo'):
    boundary = re.findall(r'^\\newlabel\{maintextend\}\{\{\}\{(\d+)\}\}\s*$',
                          aux.read_text(encoding='utf-8'), re.MULTILINE)
    if len(boundary) != 1:
        raise ValueError('Expected one main-text boundary in the report auxiliary file')
    info = subprocess.run([pdfinfo, str(pdf)], check=True, capture_output=True, text=True)
    total = re.search(r'^Pages:\s+(\d+)\s*$', info.stdout, re.MULTILINE)
    if total is None:
        raise ValueError('pdfinfo did not report the physical page count')
    main_pages, total_pages = int(boundary[0]), int(total[1])
    if not 1 <= main_pages <= min(total_pages, 6):
        raise ValueError(f'Invalid or over-limit report length: {main_pages} main pages, {total_pages} total')
    return {'pdf_sha256': sha256(pdf), 'aux_sha256': sha256(aux),
            'main_text_pages': main_pages, 'total_pages': total_pages,
            'references_and_appendix_pages': total_pages - main_pages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--aux', type=Path, required=True, help='Auxiliary file from the same compile as the PDF')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pdfinfo', default='pdfinfo')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Use a fresh page-count receipt: {args.out}')
    record = verify(args.pdf, args.aux, args.pdfinfo)
    write_json(args.out, record)
    print(f"{record['main_text_pages']} main-text pages and {record['total_pages']} total pages")


if __name__ == '__main__':
    main()
