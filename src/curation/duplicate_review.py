"""Review duplicate families only among personally approved crops; no automatic verdicts."""
import argparse
import csv
import hashlib
import io
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from PIL import Image, ImageOps
from src.common import ROOT, sha256, timestamp, write_json
from src.curation.duplicates import HammingTree
from src.curation.freeze import latest_decisions
from src.curation.provenance import crop_review_identity
from src.curation.review import CRITERIA, parse_geometry, rendered_crop

GROUP_FIELDS = ['candidate_id', 'group_id', 'keep', 'reviewer', 'timestamp', 'reason',
                'origin', 'crop_review_sha256', 'diagnostic_only']


def dhash(image):
    pixels = list(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS).getdata())
    return sum(int(pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))


def suggested_components(items, features, source_radius=8, crop_radius=4):
    """Similarity edges suggest comparisons, not semantic duplicate families."""
    trees = {'source': HammingTree(), 'crop': HammingTree()}
    edges, parent = {}, {item['candidate_id']: item['candidate_id'] for item in items}
    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier
    for item in items:
        identifier = item['candidate_id']
        feature = features[identifier]
        for kind, values, radius in [('source', [feature['source']], source_radius),
                                     ('crop', [feature['crop'], feature['mirror']], crop_radius)]:
            for value in values:
                for other, distance in trees[kind].find(value, radius):
                    key = tuple(sorted([identifier, other]))
                    edge = edges.setdefault(key, {'a': key[0], 'b': key[1], 'suggestions': {}})
                    edge['suggestions'][kind] = min(distance, edge['suggestions'].get(kind, 64))
                    parent[find(identifier)] = find(other)
            for value in set(values):
                trees[kind].add(value, identifier)
    groups = {}
    for item in items:
        groups.setdefault(find(item['candidate_id']), []).append(item['candidate_id'])
    components = [sorted(group) for group in groups.values()]
    components.sort(key=lambda group: (-len(group), group[0]))
    return components, list(edges.values())


class DuplicateReview:
    def __init__(self, pool, decisions, *, output='results/curation/approved_duplicates',
                 integrity='results/curation/integrity.csv', groups='annotations/duplicate_groups.csv',
                 log='annotations/duplicate_decisions.jsonl', diagnostic_only=False):
        self.pool, self.decisions = pool, Path(decisions)
        self.output, self.integrity = Path(output), Path(integrity)
        self.groups, self.log = Path(groups), Path(log)
        self.diagnostic_only = diagnostic_only
        self.lock = threading.RLock()
        self.features, self.source_hashes, self.cached = {}, None, None
        self.latest = latest_decisions(self.log) if self.log.exists() else {}

    def approved(self):
        decisions = latest_decisions(self.decisions) if self.decisions.exists() else {}
        approved = [row for key, row in sorted(decisions.items())
                    if not key.startswith('reject:') and row['verdict'] == 'good']
        for row in approved:
            origin = 'software_fixture_crop_review' if self.diagnostic_only else 'student_browser_review'
            if row.get('origin') != origin or not row.get('reviewer') or \
               not row.get('timestamp') or not all(row.get('criteria', {}).get(key) for key in CRITERIA):
                raise ValueError('Duplicate queue requires complete personally saved crop review provenance')
        return approved

    def checked_pixels(self, row):
        png = rendered_crop(self.pool, row['pool_id'], parse_geometry(row))
        if hashlib.sha256(png).hexdigest() != row['preview_sha256']:
            raise ValueError(f'Crop pixels changed since personal review: {row["candidate_id"]}')
        return png

    def queue(self):
        with self.lock:
            approved = self.approved()
            identities = {row['candidate_id']: crop_review_identity(row) for row in approved}
            digest = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
            if self.cached is None or self.cached['snapshot_sha256'] != digest:
                if approved and self.source_hashes is None:
                    with self.integrity.open(newline='', encoding='utf-8') as stream:
                        self.source_hashes = {row['pool_id']: int(row['dhash'], 16)
                                              for row in csv.DictReader(stream) if row['status'] == 'ok'}
                items, features = [], {}
                for row in approved:
                    identifier, identity = row['candidate_id'], identities[row['candidate_id']]
                    if identity not in self.features:
                        png = self.checked_pixels(row)
                        with Image.open(io.BytesIO(png)) as image:
                            self.features[identity] = {'source': self.source_hashes[row['pool_id']],
                                'crop': dhash(image), 'mirror': dhash(ImageOps.mirror(image))}
                    features[identifier] = self.features[identity]
                    items.append({key: row[key] for key in ['candidate_id', 'pool_id', 'label',
                        'x0', 'y0', 'size', 'angle', 'preview_sha256']} | {'crop_review_sha256': identity})
                components, edges = suggested_components(items, features)
                self.cached = {'snapshot_sha256': digest, 'items': items, 'components': components,
                    'edges': edges, 'source_radius': 8, 'crop_radius': 4,
                    'status': 'candidate_comparisons_only' if items else 'pending_personally_approved_crops',
                    'meaning': 'Source/crop/mirrored-crop dHash suggests comparisons. It does not prove duplicates or distinctness. Check across components too.'}
                write_json(self.output / digest / 'queue.json', self.cached)
            result = json.loads(json.dumps(self.cached))
            result['diagnostic_only'] = self.diagnostic_only
            for row in result['items']:
                prior = self.latest.get(row['candidate_id'])
                row['prior'] = prior if prior and prior['crop_review_sha256'] == row['crop_review_sha256'] else None
            return result

    def require_unchanged_groups(self):
        sidecar = self.groups.with_suffix('.provenance.json')
        if self.groups.exists() and (not sidecar.exists() or
                json.loads(sidecar.read_text())['csv_sha256'] != sha256(self.groups)):
            raise ValueError('Existing duplicate CSV was edited manually; preserve and reconcile it before browser export')

    def export_groups(self):
        self.require_unchanged_groups()
        sidecar = self.groups.with_suffix('.provenance.json')
        self.groups.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.groups.with_suffix('.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, GROUP_FIELDS, extrasaction='ignore')
            writer.writeheader()
            for crop in self.approved():
                prior = self.latest.get(crop['candidate_id'])
                if prior and prior['crop_review_sha256'] == crop_review_identity(crop):
                    writer.writerow(prior)
        temporary.replace(self.groups)
        write_json(sidecar, {'csv_sha256': sha256(self.groups), 'log_sha256': sha256(self.log),
            'origin': 'materialized personal duplicate-review events; not automatic similarity groups'})

    def save(self, data):
        with self.lock:
            self.require_unchanged_groups()
            if not str(data.get('reviewer', '')).strip() or data.get('personally_compared') is not True:
                raise ValueError('Enter your name and confirm personal source/crop comparison')
            queue = self.queue()
            if data.get('snapshot_sha256') != queue['snapshot_sha256']:
                raise ValueError('Approved crop set changed; reload duplicate review')
            rows = data.get('rows')
            if not isinstance(rows, list) or not rows:
                raise ValueError('No duplicate decisions supplied')
            identifiers = [row['candidate_id'] for row in rows]
            if len(set(identifiers)) != len(identifiers) or sorted(identifiers) not in queue['components']:
                raise ValueError('Review every crop in the displayed comparison component exactly once')
            approved = {row['candidate_id']: row for row in self.approved()}
            events = []
            now = timestamp()
            for row in rows:
                crop = approved[row['candidate_id']]
                group = str(row.get('group_id', '')).strip()
                if not group or len(group) > 150 or row.get('keep') not in ['yes', 'no']:
                    raise ValueError('Every crop needs a confirmed family ID and explicit keep=yes/no')
                if row['keep'] == 'no' and not str(row.get('reason', '')).strip():
                    raise ValueError('Give a reason for each crop excluded by duplicate review')
                self.checked_pixels(crop)
                events.append({'candidate_id': row['candidate_id'], 'group_id': group, 'keep': row['keep'],
                    'reason': str(row.get('reason', '')).strip(), 'reviewer': str(data['reviewer']).strip(),
                    'timestamp': now, 'origin': 'software_fixture_duplicate_review' if self.diagnostic_only else 'student_duplicate_review',
                    'diagnostic_only': self.diagnostic_only,
                    'crop_review_sha256': crop_review_identity(crop)})
            # A group may span components, but it can never keep two heads or contradictory species.
            combined = self.latest | {event['candidate_id']: event for event in events}
            families = {}
            for identifier, row in combined.items():
                if identifier not in approved or row['crop_review_sha256'] != crop_review_identity(approved[identifier]):
                    continue
                family = families.setdefault(row['group_id'], {'labels': set(), 'kept': 0})
                family['labels'].add(int(approved[identifier]['label']))
                family['kept'] += row['keep'] == 'yes'
            if any(len(family['labels']) != 1 or family['kept'] > 1 for family in families.values()):
                raise ValueError('A confirmed family must have one species and at most one kept crop')
            kept_pixels = [approved[key]['preview_sha256'] for key, row in combined.items()
                           if key in approved and row['crop_review_sha256'] == crop_review_identity(approved[key])
                           and row['keep'] == 'yes']
            if len(set(kept_pixels)) != len(kept_pixels):
                raise ValueError('Exactly identical crop pixels cannot both be kept, even with different family IDs')
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open('a', encoding='utf-8') as stream:
                stream.write(''.join(json.dumps(event, allow_nan=False) + '\n' for event in events))
                stream.flush()
            self.latest.update({event['candidate_id']: event for event in events})
            self.export_groups()
            return {'saved': len(events), 'verdicts': 'personally supplied', 'groups_csv': str(self.groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', default='data/student')
    parser.add_argument('--decisions', default='annotations/decisions.jsonl')
    parser.add_argument('--port', default=8767, type=int)
    parser.add_argument('--output', default='results/curation/approved_duplicates')
    parser.add_argument('--integrity', default='results/curation/integrity.csv')
    parser.add_argument('--groups', default='annotations/duplicate_groups.csv')
    parser.add_argument('--log', default='annotations/duplicate_decisions.jsonl')
    parser.add_argument('--diagnostic-only', action='store_true', help='Synthetic UI fixture; all inputs/outputs must stay under private_build')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.diagnostic_only and any(not Path(value).resolve().is_relative_to(ROOT / 'private_build')
            for value in [args.pool, args.decisions, args.output, args.integrity, args.groups, args.log]):
        parser.error('Diagnostic UI inputs/outputs must all use separate private_build paths')
    from catdog_eval.data import Pool
    store = DuplicateReview(Pool(args.pool), args.decisions, output=args.output, integrity=args.integrity,
                            groups=args.groups, log=args.log, diagnostic_only=args.diagnostic_only)
    if args.prepare_only:
        queue = store.queue()
        return print(queue['status'], len(queue['items']), 'approved crops; no duplicate verdicts inferred')
    token = secrets.token_urlsafe(32)
    html = (Path(__file__).parent / 'duplicate_review.html').read_text(encoding='utf-8').replace('__TOKEN__', token)
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, kind='application/json', status=200):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                url = urlparse(self.path)
                query = {key: values[0] for key, values in parse_qs(url.query).items()}
                if url.path == '/':
                    return self.send(html.encode(), 'text/html; charset=utf-8')
                if url.path == '/queue':
                    return self.send(json.dumps(store.queue()).encode())
                if url.path in ['/source', '/crop']:
                    pool_id = query['pool_id']
                    if url.path == '/crop':
                        from src.curation.review import parse_geometry
                        png = rendered_crop(store.pool, pool_id, parse_geometry(query))
                    else:
                        stream = io.BytesIO()
                        store.pool.image(pool_id).save(stream, 'PNG')
                        png = stream.getvalue()
                    return self.send(png, 'image/png')
                return self.send(b'{"error":"Not found"}', status=404)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return self.send(json.dumps({'error': str(exc)}).encode(), status=400)

        def do_POST(self):
            if self.path != '/decisions' or self.headers.get('X-Review-Token') != token:
                return self.send(b'{"error":"Invalid review session"}', status=403)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length < 500000:
                    raise ValueError('Invalid duplicate-review request size')
                return self.send(json.dumps(store.save(json.loads(self.rfile.read(length)))).encode())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return self.send(json.dumps({'error': str(exc)}).encode(), status=400)
    print(f'Personal duplicate review at http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
