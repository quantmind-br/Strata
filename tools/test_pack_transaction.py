import argparse
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import iq_pack


class PackTransaction(unittest.TestCase):
    def test_identity_interruption_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'model.gguf';src.write_bytes(b'AAAA')
            dst=root/'pack'
            a=argparse.Namespace(gguf=str(src),base=None,out=str(dst),compat_bf16=False,experts_bin=True,skip_experts=False)
            def build(args):
                stage=Path(args.out);stage.mkdir()
                (stage/'dense.bin').write_bytes(src.read_bytes())
                (stage/'experts.bin').write_bytes(src.read_bytes())
                return 0
            with patch.object(iq_pack,'Model',return_value=types.SimpleNamespace(paths=[src])),patch.object(iq_pack,'build_pack',side_effect=build) as builder:
                self.assertEqual(iq_pack.transactional_pack(a,dst),0)
                original={p.name:p.read_bytes() for p in dst.iterdir()}
                self.assertEqual(iq_pack.transactional_pack(a,dst),0)
                self.assertEqual(builder.call_count,1)
                src.write_bytes(b'BBBB')
                with self.assertRaisesRegex(ValueError,'identity mismatch'):iq_pack.transactional_pack(a,dst)
                self.assertEqual(original,{p.name:p.read_bytes() for p in dst.iterdir()})
                src.write_bytes(b'AAAA');(dst/'experts.bin').write_bytes(b'CCCC')
                with self.assertRaisesRegex(ValueError,'integrity mismatch'):iq_pack.transactional_pack(a,dst)
            def broken(args):
                Path(args.out).mkdir();(Path(args.out)/'dense.bin').write_bytes(b'partial')
                raise RuntimeError('interrupted')
            with patch.object(iq_pack,'Model',return_value=types.SimpleNamespace(paths=[src])),patch.object(iq_pack,'build_pack',side_effect=broken):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):iq_pack.transactional_pack(a,root/'new')
                self.assertFalse((root/'new').exists())
                self.assertEqual((dst/'dense.bin').read_bytes(),b'AAAA')

    def test_legacy_pack_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';src.write_bytes(b'x');dst=root/'old';dst.mkdir();(dst/'experts.bin').write_bytes(b'old')
            a=argparse.Namespace(gguf=str(src),base=None,out=str(dst),compat_bf16=False,experts_bin=True,skip_experts=False)
            with patch.object(iq_pack,'Model',return_value=types.SimpleNamespace(paths=[src])):
                with self.assertRaisesRegex(ValueError,'legacy'):iq_pack.transactional_pack(a,dst)
            self.assertEqual((dst/'experts.bin').read_bytes(),b'old')
