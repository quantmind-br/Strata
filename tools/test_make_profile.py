import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import make_profile as m


class Ranking(unittest.TestCase):
    def test_trace_reranks_complete_base_and_is_checkpoint_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';out=root/'trained';trace=root/'trace'
            ranked=[(l,e) for e in range(2) for l in range(48)]
            m.write_profile(base,ranked,2)
            trace.write_bytes(struct.pack('<iii f',47,1,1,1.0)*4)
            args=['make_profile.py',str(trace),'--base',str(base),'--out',str(out),'--n-expert','2','--checkpoint','synthetic']
            with patch.object(sys,'argv',args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):m.main()
            with patch.object(sys,'argv',args+['--rerank']),contextlib.redirect_stdout(io.StringIO()):m.main()
            self.assertEqual(m.read_profile(out,2)[0],(47,1))
            self.assertEqual(len(set(m.read_profile(out,2))),96)
            self.assertEqual(m.read_profile(base,2),ranked)
            self.assertEqual(json.loads(Path(str(out)+'.json').read_text())['checkpoint'],'synthetic')

    def test_training_requires_checkpoint_and_separate_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / 'trace'
            trace.write_bytes(struct.pack('<iii f', 0, 1, 1, 1.0))
            out = root / 'trained'
            argv = ['make_profile.py', str(trace), '--no-base', '--n-expert', '2', '--out', str(out)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                m.main()
            self.assertFalse(out.exists())
            canonical = root / 'data' / 'expert-profile.bin'
            canonical.parent.mkdir()
            canonical.write_bytes(b'original profile')
            with patch.object(m, 'ROOT', root), \
                    patch.object(sys, 'argv', argv[:-1] + [str(canonical), '--checkpoint', 'synthetic']), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                m.main()
            self.assertEqual(canonical.read_bytes(), b'original profile')

    def test_trace_frequency_ties_base_fallback_and_identity(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, out, trace = root / 'base', root / 'trained', root / 'trace'
            pairs = [(layer, expert) for expert in range(2) for layer in range(48)]
            m.write_profile(base, pairs, 2)
            trace.write_bytes(struct.pack('<iii f', 47, 1, 1, 1.0) * 3 +
                              struct.pack('<iii f', 3, 1, 1, 1.0) * 2 +
                              struct.pack('<iii f', 2, 1, 1, 1.0) * 2)
            argv = ['make_profile.py', str(trace), '--base', str(base), '--out', str(out),
                    '--n-expert', '2', '--checkpoint', 'checkpoint-A', '--rerank']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                m.main()
            trained = m.read_profile(out, 2)
            self.assertEqual(trained[:3], [(47, 1), (2, 1), (3, 1)])
            self.assertEqual(trained[3:], [pair for pair in pairs if pair not in trained[:3]])
            identity = json.loads(Path(str(out) + '.json').read_text())
            self.assertEqual(identity, {'checkpoint': 'checkpoint-A', 'rerank': True,
                                        'pairs': 96, 'traces': {str(trace): hashlib.sha256(trace.read_bytes()).hexdigest()}})
            original = out.read_bytes()
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                m.main()
            self.assertEqual(out.read_bytes(), original)
            self.assertEqual(m.read_profile(base, 2), pairs)

    def test_no_traces_preserves_complete_base_and_no_base_uses_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, out, trace = root / 'base', root / 'out', root / 'trace'
            pairs = [(layer, expert) for expert in range(2) for layer in reversed(range(48))]
            m.write_profile(base, pairs, 2)
            argv = ['make_profile.py', '--base', str(base), '--out', str(out), '--n-expert', '2']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                m.main()
            self.assertEqual(out.read_bytes(), base.read_bytes())
            trace.write_bytes(struct.pack('<iii f', 47, 1, 1, 1.0))
            argv = ['make_profile.py', str(trace), '--no-base', '--out', str(out),
                    '--n-expert', '2', '--checkpoint', 'checkpoint-B']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                m.main()
            self.assertEqual(m.read_profile(out, 2)[0], (47, 1))
            self.assertEqual(len(set(m.read_profile(out, 2))), 96)
