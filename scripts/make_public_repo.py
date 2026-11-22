"""Build the public repository from the real project history. Local only: this script never pushes.

    python -m scripts.make_public_repo build  --out private_build/public_repo
    python -m scripts.make_public_repo verify --out private_build/public_repo

build   clones the project history and rewrites it with git-filter-repo:
          * keeps only paths the course-archive allowlist permits (src.release.package.allowed) plus .gitignore, so no photo,
            raw data, checkpoint or private build file is published; paths whose content ever embedded a photo (data:image)
            and files over 90 MB are dropped from every commit;
          * keeps every commit with its real author date and order, and replaces each message by the one in
            scripts/public_history_messages.tsv (area: summary), keeping the original body without trailer lines
            (Co-Authored-By and similar);
          * unifies the author/committer identity (--name, --email);
        then adds the (git-ignored) final weights in one last commit.
verify  scans the rewritten history: image/archive/key material, tokens, embedded photos, trailer lines, identities, sizes.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.release.package import allowed  # noqa: E402

MESSAGES = ROOT / 'scripts/public_history_messages.tsv'
EMBED = ('data:image', '<image', 'image=data:')
TEXT_SUFFIXES = ('.xml', '.drawio', '.html', '.json', '.jsonl')
MAX_BYTES = 90_000_000
MAGIC = {b'\x89PNG': 'png', b'\xff\xd8\xff': 'jpeg', b'GIF8': 'gif', b'PK\x03\x04': 'zip/pt', b'\x93NUM': 'npy', b'%PDF': 'pdf',
         b'RIFF': 'riff', b'SQLi': 'sqlite'}
TOKEN = re.compile(rb'github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----')


def git(*args, cwd, text=True, **kw):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=text, **kw).stdout


def history_blobs(repo):
    """(blob sha, path) for every path version in history. `rev-list --objects` is not enough: it prints identical contents once."""
    out = git('log', '--all', '--raw', '--no-abbrev', '--no-renames', '--pretty=format:', cwd=repo)
    pairs = set()
    for line in out.splitlines():
        if not line.startswith(':'):
            continue
        meta, _, path = line.partition('\t')
        fields = meta.split()
        if fields[4] != 'D' and set(fields[3]) != {'0'}:
            pairs.add((fields[3], path))
    return sorted(pairs)


def read_blobs(repo, shas):
    """Yield (sha, size, first 8 bytes, bytes-or-None) via one cat-file --batch process; content only for text paths."""
    proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for sha in shas:
        proc.stdin.write(sha.encode() + b'\n')
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)
        yield sha, size, data
    proc.stdin.close()
    proc.wait()


def build(out, name, email):
    if out.exists():
        shutil.rmtree(out)
    subprocess.run(['git', 'clone', '-q', '--no-local', '--no-hardlinks', str(ROOT), str(out)], check=True)
    blobs = history_blobs(out)
    paths = sorted({p for _, p in blobs})
    keep = {p for p in paths if allowed(p) or p == '.gitignore'}
    by_path = {}
    for sha, p in blobs:
        by_path.setdefault(p, set()).add(sha)
    dropped = {'embedded photo': [], 'too large': []}
    candidates = [(sha, p) for p in keep for sha in by_path[p]]
    reason_of = {}
    for sha, size, data in read_blobs(out, sorted({s for s, _ in candidates})):
        if size > MAX_BYTES:
            reason_of[sha] = 'too large'
        elif any(m in data.decode('utf-8', 'ignore').lower() for m in EMBED):
            reason_of[sha] = 'embedded photo'
    for sha, p in candidates:   # every path that ever held an offending content is dropped (identical contents share a blob)
        if sha in reason_of and (reason_of[sha] == 'too large' or p.lower().endswith(TEXT_SUFFIXES)):
            dropped[reason_of[sha]].append(p)
    for reason, items in dropped.items():
        keep -= set(items)
    keep_file = out.parent / (out.name + '_keep.txt')
    keep_file.write_text('\n'.join(sorted(keep)) + '\n', encoding='utf-8')
    mailmap = out.parent / (out.name + '_mailmap.txt')
    mailmap.write_text(''.join(f'{name} <{email}> <{old}>\n' for old in sorted(set(
        git('log', '--all', '--format=%ae%n%ce', cwd=out).split()))), encoding='utf-8')
    callback = f'''
import re
short = commit.original_id[:7].decode()
new = None
for line in open({str(MESSAGES)!r}, encoding='utf-8'):
    h, _, s = line.rstrip('\\n').partition('\\t')
    if h == short:
        new = s
parts = commit.message.decode('utf-8', 'replace').split('\\n', 1)
if new is None:   # commits made after the map was written already have a clean subject
    new = parts[0]
rest = parts[1] if len(parts) > 1 else ''
lines = [l for l in rest.split('\\n') if not re.match(r'(?i)^\\s*(co-authored-by:|generated with|\\U0001F916)', l)]
body = '\\n'.join(lines).strip('\\n')
commit.message = (new + ('\\n\\n' + body if body else '') + '\\n').encode('utf-8')
'''
    tool = shutil.which('git-filter-repo') or str(ROOT / '.venvs/tools/bin/git-filter-repo')
    subprocess.run([tool, '--force', '--paths-from-file', str(keep_file), '--mailmap', str(mailmap),
                    '--commit-callback', callback], cwd=out, check=True, capture_output=True)
    # the weights are git-ignored in the project; add them in one final commit
    weights = ROOT / 'submission/weights.pt'
    (out / 'submission').mkdir(exist_ok=True)
    shutil.copy2(weights, out / 'submission/weights.pt')
    digest = subprocess.run(['sha256sum', str(weights)], capture_output=True, text=True).stdout.split()[0]
    git('add', '-f', 'submission/weights.pt', cwd=out)
    env = {'GIT_AUTHOR_NAME': name, 'GIT_AUTHOR_EMAIL': email, 'GIT_COMMITTER_NAME': name, 'GIT_COMMITTER_EMAIL': email, 'PATH': '/usr/bin:/bin'}
    subprocess.run(['git', 'commit', '-q', '-m', 'release: add final model weights (v014, EMA 0.9995, 80k updates)\n\n'
                    f'submission/weights.pt, SHA-256 {digest}. Weights are git-ignored in the working repository.'],
                   cwd=out, check=True, env=env)
    # the course archive also holds files that are git-ignored during development (mainly *.log); add them in one last commit
    from src.release.package import selected_files
    tracked = set(git('ls-files', cwd=out).splitlines())
    extra = [str(f.relative_to(ROOT)).replace('\\', '/') for f in selected_files(ROOT)]
    extra = [p for p in extra if p not in tracked]
    for rel in extra:
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    if extra:
        git('add', '-f', '--', *extra, cwd=out)
        subprocess.run(['git', 'commit', '-q', '-m', 'logs: add experiment and evaluator logs\n\nThese files are part of the course archive but git-ignored in the working '
                        f'repository (*.log and a few untracked evidence files): {len(extra)} files.'], cwd=out, check=True, env=env)
    summary = {'commits': int(git('rev-list', '--count', 'HEAD', cwd=out).strip()), 'files_kept': len(keep), 'files_added_last': len(extra),
               'dropped_embedded_photo': sorted(set(dropped['embedded photo'])), 'dropped_too_large': sorted(set(dropped['too large']))}
    print(json.dumps(summary, indent=1))


def verify(out):
    problems = []
    blobs = history_blobs(out)
    sizes = []
    shas = sorted({s for s, _ in blobs})
    path_of = {s: p for s, p in blobs}
    for sha, size, data in read_blobs(out, shas):
        p = path_of[sha]
        sizes.append((size, p))
        kind = MAGIC.get(data[:4])
        if kind and not (p == 'submission/weights.pt' or (kind == 'pdf' and p.startswith('presentation/'))):
            problems.append(f'{kind} data in {p}')
        if TOKEN.search(data):
            problems.append(f'token or key pattern in {p}')
        if p.lower().endswith(TEXT_SUFFIXES) and any(m in data.decode('utf-8', 'ignore').lower() for m in EMBED):
            problems.append(f'embedded photo in {p}')
    messages = git('log', '--format=%H%x1f%B%x1e', cwd=out)
    for pattern in ('co-authored-by', 'noreply@anthropic', 'generated with'):   # attribution trailers are errors
        hits = len(re.findall(pattern, messages, flags=re.I))
        if hits:
            problems.append(f'{hits} commit-message mention(s) of "{pattern}"')
    info = {name: len(re.findall(name, messages, flags=re.I)) for name in ('claude', 'codex')}   # plain tool names in text: informational
    identities = sorted(set(git('log', '--format=%an <%ae>%n%cn <%ce>', cwd=out).splitlines()))
    sizes.sort(reverse=True)
    report = {'commits': int(git('rev-list', '--count', 'HEAD', cwd=out).strip()),
              'first_commit': git('log', '--reverse', '--format=%ad %s', '--date=short', cwd=out).splitlines()[0],
              'last_commit': git('log', '-1', '--format=%ad %s', '--date=short', cwd=out).strip(),
              'tool_names_in_messages': info, 'identities': identities, 'tracked_files': len(git('ls-files', cwd=out).splitlines()),
              'largest_blobs_MB': [(round(s / 1e6, 1), p) for s, p in sizes[:4]], 'problems': problems}
    print(json.dumps(report, indent=1))
    return not problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('build', 'verify'))
    p.add_argument('--out', type=Path, default=ROOT / 'private_build/public_repo')
    p.add_argument('--name', default='Sou Chak Tong')
    p.add_argument('--email', default='169404840+Nvb-flipped@users.noreply.github.com')
    args = p.parse_args()
    if args.action == 'build':
        build(args.out.resolve(), args.name, args.email)
    else:
        sys.exit(0 if verify(args.out.resolve()) else 1)


if __name__ == '__main__':
    main()
