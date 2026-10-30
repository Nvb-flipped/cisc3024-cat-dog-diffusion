"""Refresh an explicitly incomplete report outline from recorded implementation evidence."""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import sha256, write_json


def main():
    report = ROOT / 'report'
    report.mkdir(exist_ok=True)
    snapshot_path = report / 'evidence_snapshot.json'
    if snapshot_path.exists():
        old_snapshot = json.loads(snapshot_path.read_text())
        for name, expected in old_snapshot.get('generated_file_sha256', {}).items():
            if not (report / name).exists() or sha256(report / name) != expected:
                raise ValueError(f'Generated report evidence was edited; reconcile manually: {name}')
    paths = {'E001': 'results/curation/integrity.json', 'E002': 'results/curation/full/summary.json',
             'E003': 'results/diagnostics/overfit/result.json',
             'E004': 'results/diagnostics/benchmark/result.json',
             'E005': 'results/diagnostics/gpu_interface.json',
             'E006': 'configs/ddpm.yaml', 'E007': 'submission/model.py',
             'E008': 'results/setup/all_fast_tests.xml'}
    records = {key: json.loads((ROOT / path).read_text()) for key, path in paths.items() if path.endswith('.json')}
    integrity, full, overfit, benchmark, interface = [records[f'E{i:03d}'] for i in range(1, 6)]
    sources = []
    for identifier, path in paths.items():
        sources.append({'evidence_id': identifier, 'source_type': 'software' if path.endswith(('.py', '.yaml')) else 'report',
            'title': f'Local project evidence: {path}', 'authors': [], 'year': 2026,
            'identifiers': {'doi': '', 'pmid': '', 'pmcid': '', 'isbn': '', 'url': (ROOT / path).as_uri()},
            'locator': path, 'confidentiality': 'restricted',
            'verification': {'status': 'unverified', 'source_opened': False, 'verified_by': '', 'verified_on': ''}})
    literature = [
        ('E101', 'Denoising Diffusion Probabilistic Models', ['Jonathan Ho', 'Ajay Jain', 'Pieter Abbeel'],
         2020, 'https://arxiv.org/abs/2006.11239', 'Noise prediction objective and reverse diffusion', 'preprint'),
        ('E102', 'Denoising Diffusion Implicit Models', ['Jiaming Song', 'Chenlin Meng', 'Stefano Ermon'],
         2020, 'https://arxiv.org/abs/2010.02502', 'Deterministic sampling, eta=0', 'preprint'),
        ('E103', 'Improved Denoising Diffusion Probabilistic Models', ['Alexander Quinn Nichol', 'Prafulla Dhariwal'],
         2021, 'https://proceedings.mlr.press/v139/nichol21a.html', 'Section 3.2 cosine schedule', 'conference_paper'),
        ('E104', 'Classifier-Free Diffusion Guidance', ['Jonathan Ho', 'Tim Salimans'],
         2022, 'https://arxiv.org/abs/2207.12598', 'Section 3.2, guidance equation; our s=1+w', 'preprint'),
        ('E105', 'Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection',
         ['Shilong Liu', 'Zhaoyang Zeng', 'Tianhe Ren', 'Feng Li', 'Hao Zhang', 'Jie Yang', 'Qing Jiang',
          'Chunyuan Li', 'Jianwei Yang', 'Hang Su', 'Jun Zhu', 'Lei Zhang'],
         2023, 'https://arxiv.org/abs/2303.05499', 'Pool curation detector; author metadata checked against arXiv v5 record', 'preprint'),
        ('E106', 'Learning Transferable Visual Models From Natural Language Supervision',
         ['Alec Radford', 'Jong Wook Kim', 'Chris Hallacy', 'Aditya Ramesh', 'Gabriel Goh', 'Sandhini Agarwal',
          'Girish Sastry', 'Amanda Askell', 'Pamela Mishkin', 'Jack Clark', 'Gretchen Krueger', 'Ilya Sutskever'],
         2021, 'https://proceedings.mlr.press/v139/radford21a.html', 'CLIP review ranking; bibliographic metadata checked against PMLR', 'conference_paper'),
        ('E107', 'Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents',
         ['Timothy Kassis', 'Vinayak Agarwal', 'Yuhuan He', 'Darshil Patel', 'Aubrey M. Brueckner'],
         2026, 'https://arxiv.org/abs/2609.00065', 'Materially used scientific writing/visualization procedures', 'preprint')]
    for identifier, title, authors, year, url, locator, kind in literature:
        sources.append({'evidence_id': identifier, 'source_type': kind, 'title': title, 'authors': authors,
            'year': year, 'identifiers': {'url': url, 'doi': '10.48550/arXiv.2609.00065' if identifier == 'E107' else '',
                                        'pmid': '', 'pmcid': '', 'isbn': ''},
            'locator': locator, 'confidentiality': 'public',
            'verification': {'status': 'unverified', 'source_opened': False, 'verified_by': '', 'verified_on': ''}})
    # A refresh must never discard a named human's existing verification record.
    manifest_path = report / 'source_manifest.json'
    if manifest_path.exists():
        prior = {item['evidence_id']: item for item in json.loads(manifest_path.read_text())['sources']}
        for source in sources:
            old = prior.get(source['evidence_id'])
            if old and old['verification']['status'] != 'unverified':
                raise ValueError('Source verification exists; reconcile evidence manually before refreshing')
    write_json(manifest_path, {'schema_version': '1.0', 'sources': sources})
    write_json(report / 'evidence_snapshot.json', {'status': 'implementation_evidence_only_not_final_results',
        'files': {identifier: {'path': path, 'sha256': sha256(ROOT / path)} for identifier, path in paths.items()},
        'verification': 'Agent inspected local files; named human source/claim checks remain pending.'})
    claims = [
        ('C001', 'Data', 'numeric', f"All {integrity['decoded_ok']:,} supplied pool images passed the recorded integrity scan.", ['E001']),
        ('C002', 'Data', 'numeric', f"The full detector run produced {full['proposals']:,} proposals and {full['geometrically_valid_proposals']:,} valid square proposals from {full['images_with_valid_proposals']:,} sources; these are not student acceptances.", ['E002']),
        ('C003', 'Method', 'numeric', f"The implemented channel fallback has {interface['parameters']:,} parameters.", ['E005','E006','E007']),
        ('C004', 'Verification', 'result', f"The separate two-image diagnostic reduced fixed-probe noise MSE from {overfit['probe_mse_before']:.6f} to {overfit['probe_mse_after']:.6f} in {overfit['updates']:,} updates; it is not evidence of generalization.", ['E003']),
        ('C005', 'Verification', 'result', f"The actual-architecture diagnostic benchmark measured {benchmark['steady_seconds_per_update']:.5f} steady seconds per update; its 50,000-update projection excludes evaluation and full-dataset overhead.", ['E004']),
        ('C006', 'Verification', 'result', f"The recorded CUDA interface stress test measured a maximum output difference of {interface['microbatch_max_absolute_difference']} when logical microbatch sizes changed while physical execution size stayed fixed.", ['E005'])]
    headings = ['# Evidence outline — incomplete course report', '',
        'This file organizes existing engineering evidence. No trained baseline, final personal audit, quality result, hidden-test result, or student independent check is claimed. The student must check sources and conclusions before submission.', '',
        'Planned main-text allocation: summary/problem 0.5 page; data/audit 1.25; methods 1.25; results/ablations 1.75; limitations 0.5; agent verification/conclusion 0.75. References and appendix are separate.', '']
    rows = []
    for cid, section, kind, sentence, ids in claims:
        headings.extend([f'## {section}: {cid}', '', f'{sentence} [claim:{cid}] [evidence:{",".join(ids)}]', ''])
        rows.append({'claim_id': cid, 'section': section, 'claim_kind': kind,
            'claim_text_sha256': hashlib.sha256(sentence.encode()).hexdigest(), 'evidence_ids': ';'.join(ids),
            'verification_status': 'unverified', 'uncertainty': 'not_estimated', 'analysis_intent': 'descriptive'})
    headings.extend(['## Evidence still required', '',
        '- Personal calibration, duplicate-family judgments, final manifest counts and holdout provenance.',
        '- Baseline and planned equal-budget ablations with exact checkpoint/manifest/code lineage.',
        '- Full official five-seed metric table: accuracy, FID, KID and variability, precision, recall, memorization; parameter count, hardware, measured training/sampling time.',
        '- Personal official 200-row audit with Wilson intervals, concrete errors, and class denominators.',
        '- Predeclared sample grid and nearest-neighbor diagnostics, with no selected best images.',
        '- Three authentic student independent checks, actual failures/agreements and resulting rules.',
        '- Actual student name/ID, source verification, final page count, figure selection, and rehearsal.', '',
        'The completed placeholder evaluator smoke only establishes environment operation. Its metrics must never enter the project result table.', '',
        'Scientific Agent Skills materially informed evidence handling and visualization. Its verified bibliographic record is retained as E107 and in references.bib; human confirmation remains pending.', ''])
    (report / 'evidence_outline.md').write_text('\n'.join(headings), encoding='utf-8')
    with (report / 'claims.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ids = ['acl23-0074_fig1', 'iclr26-0034_fig1', 'aaai25-0110_fig1', 'icml23-0003_fig1',
           'aaai25-0124_fig1', 'iclr23-0049_fig1', 'iclr25-0058_g1', 'cvpr25-0063_fig1',
           'aaai25-0026_fig1', 'neurips23-0070_fig1']
    cards = {item['id']: item for item in map(json.loads, (ROOT / '.agents/skills/figgenie-paper-diagram/assets/exemplars/ai/cards.jsonl').read_text(encoding='utf-8').splitlines())}
    write_json(ROOT / 'figures/review/exemplars.json', {'status': 'pending_student_choice',
        'copyright': 'Authors/publishers own reference figures. Study only; excluded from course archive.',
        'candidates': [dict(number=i+1, id=identifier, title=cards[identifier]['title'],
                            paper_url=cards[identifier]['paper_url'], layout=cards[identifier]['layout'])
                       for i, identifier in enumerate(ids)]})
    snapshot = json.loads(snapshot_path.read_text())
    snapshot['generated_file_sha256'] = {name: sha256(report / name)
        for name in ['source_manifest.json', 'evidence_outline.md', 'claims.csv']}
    write_json(snapshot_path, snapshot)
    print('Updated incomplete outline, source/claim registries, hashes and exemplar index.')


if __name__ == '__main__':
    main()
