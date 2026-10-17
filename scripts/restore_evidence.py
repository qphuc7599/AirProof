"""Download/restore final experimental arrays and verify all archived file bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true', help='Download missing release ZIPs')
    p.add_argument('--assets-dir', type=Path, default=ROOT / '.evidence-downloads')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--metadata-only', action='store_true', help='Check files stored directly in Git')
    args = p.parse_args()
    m = json.loads((ROOT / 'evidence/manifest.json').read_text())
    if not args.verify_only and not args.metadata_only:
        args.assets_dir.mkdir(parents=True, exist_ok=True)
        for a in m['assets']:
            path = args.assets_dir / a['name']
            if not path.exists() and args.download:
                url = f"https://github.com/qphuc7599/AirProof/releases/download/{m['release_tag']}/{a['name']}"
                part = path.with_suffix('.partial')
                with urllib.request.urlopen(url, timeout=120) as r, part.open('wb') as f:
                    shutil.copyfileobj(r, f, length=4 * 1024**2)
                part.replace(path)
            if not path.exists():
                raise SystemExit(f'Missing {path}; supply --download or --assets-dir')
            if path.stat().st_size != a['bytes'] or digest(path) != a['sha256']:
                raise SystemExit(f'Asset checksum mismatch: {path.name}')
            with zipfile.ZipFile(path) as z:
                for record in m['files']:
                    if record.get('asset') != a['name']:
                        continue
                    target = (ROOT / record['path']).resolve()
                    if not target.is_relative_to(ROOT.resolve()):
                        raise SystemExit('Unsafe manifest path')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and digest(target) == record['sha256']:
                        continue
                    if target.exists():
                        raise SystemExit(f'Refusing to overwrite different bytes: {record["path"]}')
                    with z.open(record['object']) as source, target.open('wb') as output:
                        shutil.copyfileobj(source, output, length=4 * 1024**2)
    checked = 0
    for record in m['files']:
        if args.metadata_only and record['storage'] != 'git':
            continue
        path = ROOT / record['path']
        if not path.exists() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise SystemExit(f'File missing or changed: {record["path"]}')
        checked += 1
    print(f'Verified {checked} files against the SHA-256 manifest.')


if __name__ == '__main__':
    main()
