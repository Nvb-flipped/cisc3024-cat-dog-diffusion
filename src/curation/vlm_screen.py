"""Stage 4/5 semantic screening of head candidates with a local Qwen3.5 VLM (curation only).

The VLM judges qualitative specification items (real photo, species, one head with ears,
roughly centred and upright, occlusion/blur). Geometry (head width >= 32, 60-80% occupancy,
official crop bounds) is enforced by deterministic code, never by the VLM. Each request shows
two screening views of the *same* official square: the source image with the square outlined
(context) and the square rendered at 256 px by the official crop function (detail). The 32x32
training pixels are reconstructed separately by the evaluator's code at freeze time.

Results are appended to a JSONL ledger, one line per candidate, with the model id/revision,
prompt hash, input-image hashes and raw response. Work proceeds in small chunks and resumes by
skipping recorded candidate ids, so interruptions on the shared GPU lose little.
These are model decisions (origin `vlm_screen`), never personal verdicts.
"""
import argparse
import base64
import concurrent.futures as futures
import csv
import hashlib
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

from src.common import ROOT, timestamp

sys.path.insert(0, str(ROOT / 'starter_kit'))
from catdog_eval.data import Pool, crop  # noqa: E402

PROMPT_VERSION = 'v2'
SYSTEM = ('You are a strict data curator checking training images for a cat/dog head generator. '
          'When in doubt about any criterion, answer false. Answer only with the requested JSON.')
INSTRUCTIONS = """Image 1 is a downscaled source photo; the red square marks a candidate crop.
Image 2 is exactly that square region, enlarged. Judge image 2 against every item:
- is_real_photo: a real photograph of a living cat or dog (false for cartoons, drawings,
  paintings, 3D renders, toys, plush, statues, or animals printed on products, posters, screens);
- exactly_one_head: exactly one animal head in the square; false if any other head, face or
  animal body part of another animal enters the square, even partially at the edge;
- both_ears_fully_inside: both ears are visible and completely inside the square (false if an
  ear tip touches or leaves the edge, or an ear is hidden);
- head_clearly_visible: the head is sharp enough to see eyes/nose and is not occluded by
  objects, hands or blankets, and is clearly separable from the background;
- upright: roughly upright (a slight tilt is fine; sideways or upside-down is not);
- centered: the head is roughly in the middle of the square;
- head_fraction: the share of the square's side spanned by the head including ears
  (the larger of its width and height).
Unusual breeds, dark fur, side views, puppies/kittens and busy backgrounds are acceptable.
Code-derived metadata: {metadata}
decision is PASS only if every boolean is true and head_fraction is "60_80"."""
SCHEMA = {
    'type': 'object',
    'properties': {
        'species': {'enum': ['cat', 'dog', 'other', 'uncertain']},
        'is_real_photo': {'type': 'boolean'},
        'exactly_one_head': {'type': 'boolean'},
        'both_ears_fully_inside': {'type': 'boolean'},
        'head_clearly_visible': {'type': 'boolean'},
        'upright': {'type': 'boolean'},
        'centered': {'type': 'boolean'},
        'head_fraction': {'enum': ['under_50', '50_60', '60_80', '80_90', 'over_90']},
        'decision': {'enum': ['PASS', 'REJECT', 'UNCERTAIN']},
        'short_reason': {'type': 'string', 'maxLength': 80},
    },
    'required': ['species', 'is_real_photo', 'exactly_one_head', 'both_ears_fully_inside',
                 'head_clearly_visible', 'upright', 'centered', 'head_fraction', 'decision', 'short_reason'],
    'additionalProperties': False,
}
CHECKS = ('is_real_photo', 'exactly_one_head', 'both_ears_fully_inside', 'head_clearly_visible',
          'upright', 'centered')
SYSTEM_V3 = ('You are an adversarial quality auditor. Your job is to find any reason this crop must NOT be '
             'used to train a cat/dog head generator. Answer only with the requested JSON.')
INSTRUCTIONS_V3 = """Image 1: source photo with the candidate square in red. Image 2: that square, enlarged.
Look for ANY of these problems in image 2 and set the matching field to false:
- is_real_photo=false: not a real living domestic cat/dog photo - cartoon, drawing, painting,
  render, toy, plush, statue, mask, costume animal, product print, poster, screen, collage,
  heavy watermark or text over the face; also set species "other" for wolves, foxes, lions,
  rabbits, bears and any non-cat/dog animal;
- exactly_one_head=false: any second head, face, nose, ear or body of another animal or person
  visible anywhere inside the square, even a small part at the edge;
- both_ears_fully_inside=false: an ear is cut by the square edge, folded out of view, hidden
  behind the head, or covered by a hat, hood, bow, flowers, hands or fabric;
- head_clearly_visible=false: blurry, too dark, overexposed, occluded by paws, legs, objects or
  blankets, or the head blends into the background so its outline is unclear;
- upright=false: head lying sideways, upside-down, tilted more than about 30 degrees, sleeping
  with the head down;
- centered=false: the head is clearly off-centre.
- head_fraction: share of the square's side spanned by the whole head with ears.
Code-derived metadata: {metadata}
decision is PASS only if no problem was found and head_fraction is "60_80"."""

SYSTEM_V4 = 'You inspect photos precisely and answer only with the requested JSON.'
INSTRUCTIONS_V4 = """Image 1 is a source photo with a red square. Image 2 shows only the content of that square.
Answer about image 2 only:
- heads_or_faces_visible: count every animal head and every human face visible in image 2,
  including partial ones cut by the edges (a single animal head close-up = 1);
- view: "head_closeup" if the head fills most of the image, "head_and_neck" if head plus some
  neck/chest, "most_of_body" if much of the torso/legs is visible, "whole_body" if the whole animal;
- edited_or_costume: true if there is added text, logos, graphics, frames, collage, or the
  animal wears glasses, a hat, headband, costume or other prop on its head;
- is_real_photo: real photo of a real living animal (false for drawings, renders, toys, statues);
- species: the animal's species.
Code-derived metadata: {metadata}"""
SCHEMA_V4 = {
    'type': 'object',
    'properties': {
        'heads_or_faces_visible': {'type': 'integer', 'minimum': 0, 'maximum': 9},
        'view': {'enum': ['head_closeup', 'head_and_neck', 'most_of_body', 'whole_body']},
        'edited_or_costume': {'type': 'boolean'},
        'is_real_photo': {'type': 'boolean'},
        'species': {'enum': ['cat', 'dog', 'other', 'uncertain']},
    },
    'required': ['heads_or_faces_visible', 'view', 'edited_or_costume', 'is_real_photo', 'species'],
    'additionalProperties': False,
}

SYSTEM_V5 = ('You are a strict data curator for a cat/dog head image generator. Inspect carefully; '
             'when in doubt, answer conservatively. Answer only with the requested JSON.')
INSTRUCTIONS_V5 = """Image 1 is a source photo with a red square. Image 2 shows only the content of that square.
Judge image 2:
- species: "cat" or "dog" only for a domestic cat or dog; "other" for bears, foxes, wolves,
  cheetahs, lions, tigers, rabbits or any other animal;
- is_real_photo: a real, unedited photograph of a living animal (false for drawings, cartoons,
  renders, toys, plush, statues, or animals printed on products, posters or screens);
- heads_or_faces_visible: count every animal head and every human face in image 2, including
  partial ones cut by the edges;
- view: "head_closeup" if the head fills most of image 2, "head_and_neck", "most_of_body",
  or "whole_body";
- head_width_share: how much of image 2's width the head (with ears) spans:
  "under_half", "half_to_60", "60_to_80", "over_80";
- both_ears_fully_inside: both ears visible and completely inside image 2, not cut by the edge,
  hidden, or covered;
- head_clearly_visible: eyes/nose visible, head not blurred, dark, or occluded;
- upright: head roughly upright (not lowered, lying sideways, upside-down);
- centered: head roughly in the middle;
- edited_or_costume: added text, logos, graphics, frames, collage, or a hat, glasses, headband,
  costume or other prop on the head.
Source size and crop side (pixels): {metadata}"""
SCHEMA_V5 = {
    'type': 'object',
    'properties': {
        'species': {'enum': ['cat', 'dog', 'other', 'uncertain']},
        'is_real_photo': {'type': 'boolean'},
        'heads_or_faces_visible': {'type': 'integer', 'minimum': 0, 'maximum': 9},
        'view': {'enum': ['head_closeup', 'head_and_neck', 'most_of_body', 'whole_body']},
        'head_width_share': {'enum': ['under_half', 'half_to_60', '60_to_80', 'over_80']},
        'both_ears_fully_inside': {'type': 'boolean'},
        'head_clearly_visible': {'type': 'boolean'},
        'upright': {'type': 'boolean'},
        'centered': {'type': 'boolean'},
        'edited_or_costume': {'type': 'boolean'},
    },
    'required': ['species', 'is_real_photo', 'heads_or_faces_visible', 'view', 'head_width_share',
                 'both_ears_fully_inside', 'head_clearly_visible', 'upright', 'centered', 'edited_or_costume'],
    'additionalProperties': False,
}

SYSTEM_V6 = 'You judge photo framing precisely. Answer only with the requested JSON.'
INSTRUCTIONS_V6 = """Image 1 is a source photo with a red square; image 2 is exactly that square.
Describe the framing of the animal in image 2:
- head_size: "small_less_than_half" if the head spans less than half of image 2's width,
  "about_half", "most_of_image" if the head spans most of the width but is fully visible, or
  "cut_off_by_edges" if part of the head or an ear is cut off by the border;
- ears_cut_by_border: true if any ear tip touches or leaves the border;
- gaze: "camera_or_side" if the face looks at the camera or to the side, "looking_up",
  "looking_down", or "turned_away" (back of the head);
- body_visible: "little_or_none", "some_chest_or_neck", or "most_of_body";
- other_animals_or_people: any other animal or any person (or body part) visible;
- text_or_graphics: added text, logos or graphics.
Source size and crop side (pixels): {metadata}"""
SCHEMA_V6 = {
    'type': 'object',
    'properties': {
        'head_size': {'enum': ['small_less_than_half', 'about_half', 'most_of_image', 'cut_off_by_edges']},
        'ears_cut_by_border': {'type': 'boolean'},
        'gaze': {'enum': ['camera_or_side', 'looking_up', 'looking_down', 'turned_away']},
        'body_visible': {'enum': ['little_or_none', 'some_chest_or_neck', 'most_of_body']},
        'other_animals_or_people': {'type': 'boolean'},
        'text_or_graphics': {'type': 'boolean'},
    },
    'required': ['head_size', 'ears_cut_by_border', 'gaze', 'body_visible', 'other_animals_or_people', 'text_or_graphics'],
    'additionalProperties': False,
}

SYSTEM_V7 = 'You check photos for added graphic elements. Answer only with the requested JSON.'
INSTRUCTIONS_V7 = """Image 1 is a source photo with a red square; image 2 is exactly that square.
In image 2:
- added_graphics: true if there is any ADDED design element on or around the photo: text
  banners or captions, product names or logos, website/app interface elements (buttons, menus),
  decorative frames or borders, stickers, drawn lines or shapes, cartoon elements, or a collage
  edge where two photos meet. Faint semi-transparent stock-photo watermarks do NOT count;
- second_animal_or_person: any part of another animal or of a person is visible;
- painting_or_illustration: the animal is painted, drawn or digitally illustrated rather than
  photographed.
Source size and crop side (pixels): {metadata}"""
SCHEMA_V7 = {
    'type': 'object',
    'properties': {
        'added_graphics': {'type': 'boolean'},
        'second_animal_or_person': {'type': 'boolean'},
        'painting_or_illustration': {'type': 'boolean'},
    },
    'required': ['added_graphics', 'second_animal_or_person', 'painting_or_illustration'],
    'additionalProperties': False,
}

SYSTEM_V8 = 'You help find usable training photos of cat and dog heads. Answer only with the requested JSON.'
INSTRUCTIONS_V8 = """Image 1 is a source photo with a red square; image 2 is exactly that square.
Could image 2 be used as a head-shot training image of a cat or dog? It is usable when:
- it shows a real cat or dog (photographs with faint stock watermarks, studio backgrounds or
  collars/harnesses are fine; drawings, toys, statues, prints on products and other species are not);
- one animal head is the main subject, roughly centred, roughly upright (the body may be lying
  down as long as the head itself is upright), and most of the head including ears is visible;
- no other animal head or human face is prominent in the square.
Small imperfections (an ear tip touching the edge, slight blur, a little text in a corner) are
acceptable. Source size and crop side (pixels): {metadata}"""
SCHEMA_V8 = {
    'type': 'object',
    'properties': {
        'species': {'enum': ['cat', 'dog', 'other', 'none']},
        'usable_head_shot': {'type': 'boolean'},
        'main_problem': {'enum': ['none', 'not_real_animal', 'other_species', 'no_clear_head', 'head_too_small',
                                 'head_cut_off', 'not_upright', 'second_animal_or_person', 'heavy_text_or_graphics',
                                 'too_blurry_or_dark']},
    },
    'required': ['species', 'usable_head_shot', 'main_problem'],
    'additionalProperties': False,
}
PROMPTS = {'v8': (SYSTEM_V8, INSTRUCTIONS_V8), 'v7': (SYSTEM_V7, INSTRUCTIONS_V7), 'v6': (SYSTEM_V6, INSTRUCTIONS_V6), 'v2': (SYSTEM, INSTRUCTIONS), 'v3': (SYSTEM_V3, INSTRUCTIONS_V3), 'v4': (SYSTEM_V4, INSTRUCTIONS_V4),
           'v5': (SYSTEM_V5, INSTRUCTIONS_V5)}
SCHEMAS = {'v2': SCHEMA, 'v3': SCHEMA, 'v4': SCHEMA_V4, 'v5': SCHEMA_V5, 'v6': SCHEMA_V6, 'v7': SCHEMA_V7, 'v8': SCHEMA_V8}
NO_BOX_HINTS = {'v5', 'v6', 'v7', 'v8'}  # v2-v4 showed the model copies detector-derived size hints


def prompt_sha256(version):
    system, instructions = PROMPTS[version]
    schema = SCHEMAS[version]
    if version in ('v2', 'v3'):  # keep the recorded hashes of the original prompts
        schema = SCHEMA
    return hashlib.sha256(json.dumps([version, system, instructions, schema], sort_keys=True).encode()).hexdigest()


PROMPT_SHA256 = prompt_sha256(PROMPT_VERSION)


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def render(pool, row, context_side=640, detail_side=256):
    source = pool.image(row['pool_id'])
    x0, y0, size, angle = int(row['x0']), int(row['y0']), int(row['size']), float(row['angle'] or 0)
    detail = crop(source, x0, y0, size, angle, res=detail_side)
    scale = min(1.0, context_side / max(source.size))
    context = source.resize((max(1, round(source.width * scale)), max(1, round(source.height * scale))),
                            Image.Resampling.LANCZOS) if scale < 1 else source.copy()
    draw = ImageDraw.Draw(context)
    if angle:
        import math
        cx, cy, half, a = (x0 + size / 2) * scale, (y0 + size / 2) * scale, size * scale / 2, math.radians(-angle)
        corners = [(cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a))
                   for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))]
        draw.polygon(corners, outline=(255, 0, 0), width=3)
    else:
        draw.rectangle([x0 * scale, y0 * scale, (x0 + size) * scale, (y0 + size) * scale],
                       outline=(255, 0, 0), width=3)
    meta = {'source_width': source.width, 'source_height': source.height, 'crop_side': size}
    if row.get('box_width'):
        meta['detector_head_box_width'] = float(row['box_width'])
        meta['implied_head_fraction_of_side'] = round(float(row['box_width']) / size, 2)
    return png_bytes(context), png_bytes(detail), meta


def request(server, model, context_png, detail_png, meta, max_tokens=160, prompt='v2'):
    system, instructions = PROMPTS[prompt]
    content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}
               for png in (context_png, detail_png)]
    content.append({'type': 'text', 'text': instructions.format(metadata=json.dumps(meta))})
    body = {'model': model, 'temperature': 0, 'max_tokens': max_tokens,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'verdict', 'schema': SCHEMAS[prompt]}},
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(server + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        reply = json.loads(response.read())
    return reply['choices'][0]['message']['content'], reply.get('usage', {})


def resolved(verdict):
    """Deterministic acceptance from the VLM fields; the VLM's own decision must agree."""
    if 'usable_head_shot' in verdict:  # v8 recall-oriented recovery question
        return 'PASS' if verdict['usable_head_shot'] and verdict['species'] in ('cat', 'dog') else 'REJECT'
    if 'added_graphics' in verdict:  # v7 graphics/second-subject check
        return 'REJECT' if (verdict['added_graphics'] or verdict['second_animal_or_person']
                            or verdict['painting_or_illustration']) else 'PASS'
    if 'gaze' in verdict:  # v6 framing second opinion
        ok = (verdict['head_size'] in ('most_of_image', 'about_half') and not verdict['ears_cut_by_border']
              and verdict['gaze'] == 'camera_or_side' and verdict['body_visible'] != 'most_of_body'
              and not verdict['other_animals_or_people'] and not verdict['text_or_graphics'])
        return 'PASS' if ok else 'REJECT'
    if 'head_width_share' in verdict:  # v5: combined checklist, no detector hints
        # The VLM size band is ignored: Qwen3.5-9B rated every valid close-up "over_80" on the blind
        # sample; occupancy is fixed by geometry and re-measured by src.curation.head_measure instead.
        ok = (verdict['species'] in ('cat', 'dog') and verdict['is_real_photo'] and verdict['heads_or_faces_visible'] == 1
              and verdict['view'] in ('head_closeup', 'head_and_neck')
              and verdict['both_ears_fully_inside'] and verdict['head_clearly_visible'] and verdict['upright']
              and verdict['centered'] and not verdict['edited_or_costume'])
        return 'PASS' if ok else ('UNCERTAIN' if verdict['species'] == 'uncertain' else 'REJECT')
    if 'heads_or_faces_visible' in verdict:  # focused v4 checks
        ok = (verdict['heads_or_faces_visible'] == 1 and verdict['view'] in ('head_closeup', 'head_and_neck')
              and not verdict['edited_or_costume'] and verdict['is_real_photo'] and verdict['species'] in ('cat', 'dog'))
        return 'PASS' if ok else ('UNCERTAIN' if verdict['species'] == 'uncertain' else 'REJECT')
    checks = (all(verdict[k] for k in CHECKS) and verdict['head_fraction'] == '60_80'
              and verdict['species'] in ('cat', 'dog'))
    if verdict['decision'] == 'PASS' and checks:
        return 'PASS'
    if verdict['decision'] == 'UNCERTAIN' or verdict['species'] == 'uncertain' or (verdict['decision'] == 'PASS') != checks:
        return 'UNCERTAIN'
    return 'REJECT'


def done(ledger):
    ids = set()
    if ledger.exists():
        with ledger.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    ids.add(json.loads(line)['candidate_id'])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def screen(queue, ledger, server, model, model_revision, *, concurrency=16, chunk=500, limit=None,
           pool_path='data/student', prompt='v2', only_pass_in=None):
    with open(queue, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    finished = done(ledger)
    if only_pass_in is not None:  # second-opinion pass: only candidates another ledger passed
        passed = {json.loads(line)['candidate_id'] for line in open(only_pass_in, encoding='utf-8')
                  if json.loads(line)['resolved'] == 'PASS'}
        rows = [row for row in rows if row['candidate_id'] in passed]
    rows = [row for row in rows if row['candidate_id'] not in finished][:limit]
    print(f'{len(rows)} candidates to screen; {len(finished)} already recorded', flush=True)
    pool = Pool(pool_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)

    def work(row):
        context_png, detail_png, meta = render(pool, row)
        if prompt in NO_BOX_HINTS:
            meta = {k: meta[k] for k in ('source_width', 'source_height', 'crop_side')}
        started = time.perf_counter()
        error, content, usage, verdict = None, None, {}, None
        for attempt in range(2):  # one retry for malformed output
            try:
                content, usage = request(server, model, context_png, detail_png, meta, prompt=prompt)
            except OSError:
                # Server unreachable/killed (shared GPU): do not record a verdict. Abort so the
                # supervisor retries later; finished candidates in this chunk are already saved.
                raise
            try:
                verdict = json.loads(content)
                break
            except json.JSONDecodeError as exc:  # recorded as ERROR for later escalation
                error = f'{type(exc).__name__}: {exc}'
        return {'candidate_id': row['candidate_id'], 'pool_id': row['pool_id'],
                'geometry': {k: row[k] for k in ('x0', 'y0', 'size', 'angle')},
                'origin': 'vlm_screen', 'model': model, 'model_revision': model_revision,
                'prompt_version': prompt, 'prompt_sha256': prompt_sha256(prompt),
                'context_png_sha256': hashlib.sha256(context_png).hexdigest(),
                'detail_png_sha256': hashlib.sha256(detail_png).hexdigest(), 'metadata': meta,
                'response': content, 'verdict': verdict, 'resolved': resolved(verdict) if verdict else 'ERROR',
                'error': None if verdict else error, 'usage': usage,
                'seconds': round(time.perf_counter() - started, 3), 'timestamp': timestamp()}

    begin, completed = time.perf_counter(), 0
    with ledger.open('a', encoding='utf-8') as stream, futures.ThreadPoolExecutor(concurrency) as executor:
        for start in range(0, len(rows), chunk):
            try:
                for record in executor.map(work, rows[start:start + chunk]):
                    stream.write(json.dumps(record) + '\n')
                    completed += 1
            finally:
                stream.flush()
            elapsed = time.perf_counter() - begin
            print(f'{completed}/{len(rows)} screened, {completed / elapsed:.2f}/s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--server', default='http://prp-vllm:8000')
    parser.add_argument('--model', default='qwen3.5-4b')
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--chunk', type=int, default=500)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--prompt', default='v2', choices=sorted(PROMPTS))
    parser.add_argument('--only-pass-in', type=Path, help='Screen only candidates PASSed in this ledger')
    args = parser.parse_args()
    screen(args.queue, args.ledger, args.server, args.model, args.model_revision,
           concurrency=args.concurrency, chunk=args.chunk, limit=args.limit, prompt=args.prompt,
           only_pass_in=args.only_pass_in)


if __name__ == '__main__':
    main()
