"""Clean SM86 release build. Publishes an immutable candidate, never replaces active binaries."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def source_identity():
    paths = []
    for folder in ('src', 'include', 'cmake', 'serve', 'tools', 'tests', 'third_party/ggml'):
        paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend([ROOT / 'CMakeLists.txt', ROOT / 'backend-build.sh'])
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    sources = source_identity()
    release_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:12]
    releases = ROOT / 'releases'
    releases.mkdir(exist_ok=True)
    build_root = ROOT / 'build-releases'
    build_root.mkdir(exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=release_id + '-', dir=build_root))
    configure = ['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                 '-DSTRATA_ENABLE_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=86', '-DSTRATA_NATIVE_EXPERTS=ON',
                 '-DSTRATA_BUILD_CONVERSATION_TESTS=ON', '-DSTRATA_BUILD_STARTUP_TESTS=ON']
    if os.environ.get('STRATA_GGML_DIR'):
        configure += ['-DSTRATA_GGML_DIR=' + os.environ['STRATA_GGML_DIR']]
    subprocess.run(configure, check=True)
    subprocess.run(['cmake', '--build', str(build), '-j', os.environ.get('STRATA_BUILD_JOBS', '8')], check=True)
    tests = ['ctest', '--test-dir', str(build), '--output-on-failure']
    if os.environ.get('STRATA_FULL_TESTS') != '1':
        tests += ['-LE', 'external_fixture']
    subprocess.run(tests, check=True)
    if source_identity() != sources:
        raise SystemExit('Source changed during build; candidate not published. Rerun after edits finish.')
    with tempfile.TemporaryDirectory(prefix='.stage-', dir=releases) as tmp:
        stage = Path(tmp) / release_id
        stage.mkdir()
        for folder in ('serve', 'tools'):
            shutil.copytree(ROOT / folder, stage / folder, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(build / 'strata', stage / 'strata')
        manifest = {'release': release_id, 'revision': command('git', 'rev-parse', 'HEAD'),
                    'dirty': command('git', 'status', '--porcelain'), 'sources': sources,
                    'tracked_diff': command('git', 'diff', 'HEAD', '--binary'),
                    'configure': configure, 'test_command': tests, 'external_fixtures_required_for_full_parity': True, 'cmake': command('cmake', '--version'),
                    'cxx': command(os.environ.get('CXX', 'c++'), '--version'),
                    'ggml_revision': command('git', '-C', os.environ['STRATA_GGML_DIR'], 'rev-parse', 'HEAD') if os.environ.get('STRATA_GGML_DIR') else (ROOT/'third_party/ggml/VERSION.txt').read_text(),
                    'cuda': command(os.environ.get('CUDACXX', '/opt/cuda/bin/nvcc'), '--version'),
                    'binary_sha256': hashlib.sha256((stage / 'strata').read_bytes()).hexdigest(),
                    'cmake_cache': (build / 'CMakeCache.txt').read_text()}
        (stage / 'BUILD.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(releases / release_id)
    print(json.dumps({'release': str(releases / release_id), 'build': str(build)}))


if __name__ == '__main__':
    main()
