"""Prepare near-duplicate candidates for student confirmation; never auto-reject them."""
import argparse
import csv
from pathlib import Path
from src.common import write_json


class HammingTree:
    def __init__(self):
        self.root = None

    def add(self, value, identifier):
        if self.root is None:
            self.root = [value, [identifier], {}]
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if distance == 0:
                node[1].append(identifier)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [identifier], {}]
                return
            node = node[2][distance]

    def find(self, value, radius):
        stack = [self.root] if self.root else []
        found = []
        while stack:
            node = stack.pop()
            distance = (node[0] ^ value).bit_count()
            if distance <= radius:
                found.extend((item, distance) for item in node[1])
            stack.extend(child for edge, child in node[2].items()
                         if distance - radius <= edge <= distance + radius)
        return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', default='results/curation/pilot/candidates.csv', type=Path)
    p.add_argument('--integrity', default='results/curation/integrity.csv', type=Path)
    p.add_argument('--radius', default=8, type=int)
    p.add_argument('--out', default='results/curation/duplicate_candidates.csv', type=Path)
    args = p.parse_args()
    with args.candidates.open(newline='', encoding='utf-8') as f:
        candidates = [r for r in csv.DictReader(f) if r['size']]
    with args.integrity.open(newline='', encoding='utf-8') as f:
        integrity = {r['pool_id']: r for r in csv.DictReader(f) if r['status'] == 'ok'}
    tree = HammingTree()
    by_source = {}
    pairs = []
    for row in sorted(candidates, key=lambda r: r['candidate_id']):
        record = integrity[row['pool_id']]
        value = int(record['dhash'], 16)
        for other, distance in tree.find(value, args.radius):
            pairs.append({'candidate_a': other, 'candidate_b': row['candidate_id'],
                          'dhash_distance': distance, 'status': 'pending_student_confirmation'})
        tree.add(value, row['candidate_id'])
        by_source.setdefault(row['pool_id'], []).append(row['candidate_id'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate_a', 'candidate_b', 'dhash_distance', 'status'])
        writer.writeheader()
        writer.writerows(pairs)
    template = args.out.with_name('duplicate_groups.template.csv')
    with template.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate_id', 'group_id', 'keep', 'reviewer', 'timestamp', 'reason'])
        writer.writeheader()
        for row in candidates:
            writer.writerow({'candidate_id': row['candidate_id']})
    write_json(args.out.with_suffix('.json'), {'candidate_count': len(candidates), 'pair_count': len(pairs),
        'radius': args.radius, 'meaning': 'Source dHash is a review aid, not proof of duplicate heads.',
        'review': 'Raw proposal diagnostic only. After personal crop approvals, use '
                  'python -m src.curation.duplicate_review for signature-bound family/keep decisions. '
                  'Do not use this raw template as the final reviewed-group file. Never prefill human judgments.'})
    print(f'{len(pairs)} candidate pairs; blank group template: {template}')


if __name__ == '__main__':
    main()
