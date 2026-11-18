"""Create verified private-release snapshot parts, then upload and verify them."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'private_build/github_migration_20261119'
SKIP_DIRS = {'.git', '.venvs', '__pycache__', '.pytest_cache', '.pytest_tmp', 'tmp', '.cache'}
LIMIT = 1_500_000_000


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def gh(*args):
    result = subprocess.run(['gh', *args], cwd=ROOT, capture_output=True, text=True, timeout=7200)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout


def upload(path, repo, tag):
    for attempt in range(3):
        try:
            gh('release', 'upload', tag, str(path), '--repo', repo)
            return
        except (RuntimeError, subprocess.TimeoutExpired):
            # A successful upload can outlive the client response; never replace an existing asset blindly.
            assets = json.loads(gh('release','view',tag,'--repo',repo,'--json','assets'))['assets']
            existing = [a for a in assets if a['name'] == path.name]
            if existing and existing[0]['size'] == path.stat().st_size:
                return
            if attempt == 2:
                raise
            time.sleep(5)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--tag', default='migration-2026-11-19')
    parser.add_argument('--upload-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    manifest_path = OUT/'snapshot-manifest.json'
    if not args.upload_only:
        if manifest_path.exists() or list(OUT.glob('project-snapshot.part*.zip')):
            raise ValueError('Snapshot already exists; use --upload-only to resume uploads')
        paths = []
        for root, dirs, names in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and (Path(root)/d).resolve() != OUT]
            for name in names:
                path = Path(root)/name
                if name == '.env' or name.startswith('.env.') or path.suffix.lower() in {'.pem','.key','.pfx','.p12'}:
                    raise ValueError(f'Secret-like file requires exclusion review: {path.relative_to(ROOT)}')
                if path.is_symlink():
                    raise ValueError(f'Symlink needs explicit migration handling: {path.relative_to(ROOT)}')
                paths.append(path)
        paths.sort(key=lambda p: (p.parts[len(ROOT.parts)] in {'checkpoints','models','student','data'},str(p)))
        manifest = {'origin':'user_authorized_private_pc_migration','created_unix':time.time(),
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'excluded_directories':sorted(SKIP_DIRS),'archives':[],'files':[]}
        groups, current, size = [], [], 0
        for path in paths:
            n = path.stat().st_size
            if n > LIMIT:
                raise ValueError(f'File exceeds archive partition budget: {path}')
            if current and size+n > LIMIT:
                groups.append(current);current=[];size=0
            current.append(path);size+=n
        if current:groups.append(current)
        print(f'PACKING {len(paths)} files in {len(groups)} parts',flush=True)
        for number, group in enumerate(groups,1):
            target = OUT/f'project-snapshot.part{number:03}.zip'
            with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
                for path in group:
                    before = path.stat()
                    relative = path.relative_to(ROOT).as_posix()
                    h = hashlib.sha256()
                    with path.open('rb') as source, archive.open(relative,'w',force_zip64=True) as dest:
                        for block in iter(lambda: source.read(8*1024*1024),b''):
                            dest.write(block);h.update(block)
                    after = path.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                        raise ValueError(f'File changed during snapshot: {relative}')
                    manifest['files'].append({'path':relative,'size':after.st_size,
                                              'sha256':h.hexdigest(),'archive':target.name})
            manifest['archives'].append({'name':target.name,'size':target.stat().st_size,'sha256':sha(target)})
            save_json(OUT/'pack-progress.json',manifest)
            print(f'PACKED {number}/{len(groups)} {target.stat().st_size} bytes',flush=True)
        save_json(manifest_path,manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    metadata = json.loads(gh('repo','view',args.repo,'--json','visibility'))
    if metadata['visibility'] != 'PRIVATE':raise ValueError('Repository must remain PRIVATE')
    assets = json.loads(gh('release','view',args.tag,'--repo',args.repo,'--json','assets'))['assets']
    present = {a['name']:a for a in assets}
    pending=[]
    for item in manifest['archives']:
        path=OUT/item['name']
        if path.stat().st_size != item['size'] or sha(path)!=item['sha256']:raise ValueError('Local archive changed')
        if item['name'] not in present:pending.append(path)
        elif present[item['name']]['size']!=item['size']:raise ValueError('Remote asset size conflict')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as workers:
        futures={workers.submit(upload,p,args.repo,args.tag):p for p in pending}
        for future in concurrent.futures.as_completed(futures):
            future.result();print('UPLOADED '+futures[future].name,flush=True)
    upload(manifest_path,args.repo,args.tag)
    release = json.loads(gh('release','view',args.tag,'--repo',args.repo,'--json','id,assets,url,isDraft'))
    remote_pages = json.loads(gh('api','--paginate','--slurp',f'repos/{args.repo}/releases/{release["id"]}/assets'))
    remote = {a['name']:a for page in remote_pages for a in page}
    verified=[]
    for item in manifest['archives']+[{'name':manifest_path.name,'size':manifest_path.stat().st_size,'sha256':sha(manifest_path)}]:
        asset=remote[item['name']]
        if asset['size']!=item['size']:raise ValueError('Remote asset size mismatch')
        if asset.get('digest')!='sha256:'+item['sha256']:raise ValueError('Remote SHA-256 unavailable or mismatched')
        verified.append({'name':item['name'],'size':item['size'],'sha256':item['sha256']})
    gh('release','edit',args.tag,'--repo',args.repo,'--draft=false')
    save_json(OUT/'upload-verification.json',{'verified_unix':time.time(),'repo':args.repo,
        'release':args.tag,'assets':verified,'private':True,'status':'complete'})
    upload(OUT/'upload-verification.json',args.repo,args.tag)
    print('COMPLETE: private snapshot uploaded with remote SHA-256 parity',flush=True)


if __name__=='__main__':main()
