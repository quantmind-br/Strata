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
