import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import build_release


class ReleasePackaging(unittest.TestCase):
    def test_mid_copy_mutation_never_publishes_release(self):
        # Both a copied mutation restored in the live tree and a mutation after
        # copying must be rejected: staged bytes and live identity are separate gates.
        for restored in (True, False):
            with self.subTest(restored=restored), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ('CMakeLists.txt', 'backend-build.sh', 'serve/server.py',
                             'tools/converter.py', 'third_party/ggml/VERSION.txt'):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('original')
                original_copytree = build_release.shutil.copytree

                def compiler(cmd, **kwargs):
                    if '--build' in cmd:
                        build = Path(cmd[cmd.index('--build') + 1])
                        (build / 'strata').write_bytes(b'compiled binary')
                        (build / 'CMakeCache.txt').write_text('configured')

                def copytree(src, dst, **kwargs):
                    server = root / 'serve/server.py'
                    if src == root / 'serve' and restored:
                        server.write_text('changed while copying')
                    result = original_copytree(src, dst, **kwargs)
                    if src == root / 'serve':
                        server.write_text('original' if restored else 'changed after copying')
                    return result

                with patch.object(build_release, 'ROOT', root), \
                        patch.dict(os.environ, {'STRATA_GGML_DIR': ''}), \
                        patch.object(build_release.subprocess, 'run', side_effect=compiler), \
                        patch.object(build_release.shutil, 'copytree', side_effect=copytree):
                    with self.assertRaisesRegex(SystemExit, 'candidate not published'):
                        build_release.main()
                self.assertEqual(list((root / 'releases').iterdir()), [])
