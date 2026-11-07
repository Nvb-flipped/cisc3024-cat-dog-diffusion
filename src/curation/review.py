"""Local source/crop review with explicit student decisions and append-only history."""
import argparse
import csv
import hashlib
import io
import json
import math
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.common import ROOT, timestamp
from src.curation.geometry import crop, inside

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool

CRITERIA = ['species_correct', 'real_photo', 'one_head', 'ears_visible', 'upright', 'centered', 'occupancy_valid']


def parse_geometry(data):
    geometry = {}
    for name in ['x0', 'y0', 'size']:
        value = float(data[name])
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f'{name} must be an integer')
        geometry[name] = int(value)
    return geometry | {'angle': float(data['angle'])}


def rendered_crop(pool, pool_id, geometry):
    image = pool.image(pool_id)
    if not inside(*image.size, **geometry):
        raise ValueError('Crop is outside source, smaller than 40 px, or angle is outside +/-45 degrees')
    result = crop(image, **geometry)
    stream = io.BytesIO()
    result.save(stream, format='PNG')
    return stream.getvalue()


class ReviewStore:
    def __init__(self, candidates, selection, decisions, full_candidates=None, scores=None, shortlists=()):
        with Path(candidates).open(newline='', encoding='utf-8') as f:
            self.candidates = {r['candidate_id']: r for r in csv.DictReader(f)}
        self.pilot_ids = sorted(self.candidates)
        self.selection = json.loads(Path(selection).read_text(encoding='utf-8'))
        self.scores = {}
        if scores and Path(scores).exists():
            with Path(scores).open(newline='', encoding='utf-8') as f:
                self.scores = {row['candidate_id']: row for row in csv.DictReader(f)}
        self.full_queue = []
        if full_candidates and Path(full_candidates).exists():
            with Path(full_candidates).open(newline='', encoding='utf-8') as f:
                full = [row for row in csv.DictReader(f) if row['size']]
            self.candidates.update({row['candidate_id']: row for row in full})
            full.sort(key=lambda row: (-float(self.scores.get(row['candidate_id'], {}).get('review_priority_score', -100)),
                                      row['candidate_id']))
            by_source = {}
            for row in full:
                by_source.setdefault(row['pool_id'], row)
            balanced = [[], []]
            for row in by_source.values():
                hint = self.scores.get(row['candidate_id'], {}).get('crop_species_hint', row['label'])
                balanced[int(hint)].append(row['candidate_id'])
            for index in range(max(map(len, balanced), default=0)):
                for group in balanced:
                    if index < len(group):
                        self.full_queue.append(group[index])
        self.shortlists = []
        for shortlist in shortlists:
            batch = json.loads(Path(shortlist).read_text(encoding='utf-8'))
            for row in batch['recovered_candidates']:
                key = row['candidate_id']
                if key in self.candidates or key.startswith('reject:'):
                    raise ValueError(f'Recovery proposal ID conflicts with an existing item: {key}')
                self.candidates[key] = row
            keys = batch['candidate_ids']
            if len(keys) != len(set(keys)) or any(
                not self.candidates.get(key, {}).get('size') for key in keys
            ):
                raise ValueError('Shortlist must contain unique, available crop proposal IDs')
            self.shortlists.append({'id': batch['batch'], 'candidate_ids': keys})
        self.path = Path(decisions)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.latest = {}
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                event = json.loads(line)
                self.latest[event['candidate_id']] = event

    def items(self, mode, page=0, page_size=200):
        if mode == 'rejected':
            return [{'candidate_id': f'reject:{pid}', 'pool_id': pid, 'label': 0,
                     'prior': self.latest.get(f'reject:{pid}')}
                    for pid in self.selection['rejected_pool_ids']]
        if mode == 'full':
            keys = self.full_queue[page * page_size:(page + 1) * page_size]
        elif mode == 'shortlist':
            keys = self.shortlists[page]['candidate_ids'] if page < len(self.shortlists) else []
        else:
            keys = self.selection['proposed_candidate_ids'] if mode == 'calibration' else self.pilot_ids
        return [self.candidates[key] | {'prior': self.latest.get(key), 'screening_hint':
            self.scores.get(key, {}).get('crop_species_hint') if mode in ('full', 'shortlist') else None} for key in keys
                if self.candidates[key].get('size')]

    def validate(self, data, pool):
        key = data['candidate_id']
        reviewer = str(data.get('reviewer', '')).strip()
        if not reviewer:
            raise ValueError('Enter the name of the student personally performing this review')
        event = {'candidate_id': key, 'reviewer': reviewer, 'timestamp': timestamp(),
                 'verdict': data['verdict'], 'reason': str(data.get('reason', '')).strip(),
                 'origin': 'student_browser_review'}
        if key.startswith('reject:'):
            pool_id = key.split(':', 1)[1]
            if pool_id not in self.selection['rejected_pool_ids']:
                raise ValueError('Unknown rejected source')
            if data['verdict'] not in ['missed_head', 'correct_rejection']:
                raise ValueError('Rejected-source review requires missed_head or correct_rejection')
            if data['verdict'] == 'missed_head' and not event['reason']:
                raise ValueError('Describe the missed species/head and why it was missed')
            return event | {'pool_id': pool_id}
        candidate = self.candidates[key]
        if data['verdict'] not in ['good', 'bad', 'uncertain']:
            raise ValueError('Choose good, bad, or uncertain')
        geometry = parse_geometry(data)
        png = rendered_crop(pool, candidate['pool_id'], geometry)
        digest = hashlib.sha256(png).hexdigest()
        if data.get('preview_sha256') != digest:
            raise ValueError('Preview the exact current crop before saving')
        label = int(data['label'])
        if label not in (0, 1):
            raise ValueError('Species must be cat or dog')
        criteria = {name: bool(data.get('criteria', {}).get(name, False)) for name in CRITERIA}
        head_width = float(data.get('head_width') or 0)
        occupancy = float(data.get('occupancy') or 0)
        if not math.isfinite(head_width) or not math.isfinite(occupancy):
            raise ValueError('Head width and occupancy must be finite numbers')
        basis = data.get('head_width_basis', 'student_entry')
        if basis not in ['student_entry', 'source_clicks', 'proposal_confirmed']:
            raise ValueError('Unknown head-width measurement basis')
        if basis == 'source_clicks':
            measurement = data.get('head_measurement')
            width, height = pool.image(candidate['pool_id']).size
            if not isinstance(measurement, list) or len(measurement) != 2:
                raise ValueError('A width measurement needs two source-image points')
            if any(not 0 <= p['x'] <= width or not 0 <= p['y'] <= height for p in measurement):
                raise ValueError('Measurement points are outside the delivered source')
            if abs(abs(measurement[1]['x'] - measurement[0]['x']) - head_width) > .11:
                raise ValueError('Head width does not match clicked source coordinates')
        if data['verdict'] == 'good':
            if not all(criteria.values()) or head_width < 32 or not 0.60 <= occupancy <= 0.80:
                raise ValueError('Good requires every criterion, confirmed source head width >=32 px and occupancy 0.60-0.80')
        elif not event['reason']:
            raise ValueError('Give a concrete reason for bad or uncertain crops')
        return event | {'pool_id': candidate['pool_id'], 'label': label, **geometry,
                        'head_width': head_width, 'occupancy': occupancy, 'criteria': criteria,
                        'preview_sha256': digest, 'proposed_label': int(candidate['label']),
                        'head_width_basis': basis,
                        'head_measurement': data.get('head_measurement'),
                        'angle_reason': str(data.get('angle_reason', '')).strip()}

    def save(self, event):
        with self.lock:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(event, allow_nan=False) + '\n')
                f.flush()
            self.latest[event['candidate_id']] = event


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', default='results/curation/pilot/candidates.csv', type=Path)
    p.add_argument('--pool', default='data/student')
    p.add_argument('--decisions', default='annotations/decisions.jsonl', type=Path)
    p.add_argument('--port', default=8765, type=int)
    p.add_argument('--full-candidates', default='results/curation/full/candidates.csv', type=Path)
    p.add_argument('--scores', default='results/curation/photo_screen_full/scores.csv', type=Path)
    p.add_argument('--shortlist', action='append', type=Path, help='Load a shortlist file; repeat for multiple batches')
    args = p.parse_args()
    shortlists = args.shortlist or sorted(Path('results/curation').glob('priority_shortlist_v[0-9][0-9][0-9].json'))
    store = ReviewStore(args.candidates, args.candidates.parent / 'calibration_selection.json', args.decisions,
                        args.full_candidates, args.scores, shortlists)
    pool = Pool(args.pool)
    csrf = secrets.token_urlsafe(32)
    html = (Path(__file__).parent / 'review.html').read_text(encoding='utf-8').replace('__TOKEN__', csrf)

    class Handler(BaseHTTPRequestHandler):
        def send(self, body, kind='application/json', status=200, extra=None):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                url = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                if url.path == '/':
                    return self.send(html.encode(), 'text/html; charset=utf-8')
                if url.path == '/review.js':
                    return self.send((Path(__file__).parent / 'review.js').read_bytes(), 'application/javascript; charset=utf-8')
                if url.path == '/items':
                    mode = query.get('mode', 'calibration')
                    page = max(0, int(query.get('page', 0)))
                    if mode == 'shortlist':
                        page = min(page, max(0, len(store.shortlists) - 1))
                    items = store.items(mode, page=page)
                    total = len(store.full_queue) if mode == 'full' else len(items)
                    result = {'items': items, 'total': total, 'page': page}
                    if mode == 'shortlist':
                        result['batch_count'] = len(store.shortlists)
                        result['batch_id'] = store.shortlists[page]['id'] if store.shortlists else None
                    return self.send(json.dumps(result).encode())
                if url.path in ['/source', '/crop']:
                    pool_id = query['pool_id']
                    if pool_id not in pool:
                        raise ValueError('Unknown source ID')
                    if url.path == '/crop':
                        png = rendered_crop(pool, pool_id, parse_geometry(query))
                    else:
                        stream = io.BytesIO()
                        pool.image(pool_id).save(stream, format='PNG')
                        png = stream.getvalue()
                    return self.send(png, 'image/png', extra={'X-Crop-SHA256': hashlib.sha256(png).hexdigest()})
                return self.send(b'{"error":"Not found"}', status=404)
            except (ValueError, KeyError, TypeError) as exc:
                return self.send(json.dumps({'error': str(exc)}).encode(), status=400)

        def do_POST(self):
            if self.path != '/decision' or self.headers.get('X-Review-Token') != csrf:
                return self.send(b'{"error":"Invalid review session"}', status=403)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length < 20000:
                    raise ValueError('Invalid decision size')
                data = json.loads(self.rfile.read(length))
                event = store.validate(data, pool)
                if event.get('angle') and not event['angle_reason']:
                    raise ValueError('Explain the nonzero rotation')
                store.save(event)
                return self.send(json.dumps(event).encode())
            except (ValueError, KeyError, TypeError) as exc:
                return self.send(json.dumps({'error': str(exc)}).encode(), status=400)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Review at http://127.0.0.1:{args.port} — student decisions required', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
