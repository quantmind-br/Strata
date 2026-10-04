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

    def test_real_converter_publishes_verified_artifacts_and_refuses_changes(self):
        import contextlib
        import io
        import sys
        from test_iq_pack import split_model, expected_experts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = split_model(root)
            dst = root / "published"
            argv = ["iq_pack.py", "--gguf", str(shards[0]), "--out", str(dst), "--experts-bin"]

            def export(cmd, **kwargs):
                tokenizer = Path(cmd[cmd.index("--out") + 1]) / "tokenizer"
                tokenizer.mkdir()
                (tokenizer / "vocab.json").write_text('{}')
                (tokenizer / "chat_template.jinja").write_text('fixture')

            with patch.object(sys, "argv", argv), patch.object(iq_pack.subprocess, "run", side_effect=export), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.main(), 0)
                model = iq_pack.Model(shards[0])
                self.assertEqual((dst / "experts.bin").read_bytes(), expected_experts(model))
                manifest = json.loads((dst / "PACK.json").read_text())
                self.assertEqual(manifest["identity"]["format"], 1)
                self.assertEqual(manifest["artifacts"], iq_pack.tree_identity(dst))
                before = {p.relative_to(dst): (p.read_bytes(), p.stat().st_mtime_ns)
                          for p in dst.rglob("*") if p.is_file()}
                self.assertEqual(iq_pack.main(), 0)
                self.assertEqual(before, {p.relative_to(dst): (p.read_bytes(), p.stat().st_mtime_ns)
                                          for p in dst.rglob("*") if p.is_file()})
                with patch.object(sys, "argv", argv[:-1]):
                    with self.assertRaisesRegex(ValueError, "identity mismatch"):
                        iq_pack.main()
                self.assertEqual((dst / "experts.bin").read_bytes(), expected_experts(model))
            self.assertFalse(list(root.glob(".stage-published-*")))
            self.assertTrue((root / ".published.pack.lock").exists())

    def test_dependency_changes_invalidate_published_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source"
            src.write_bytes(b"source")
            dependency = root / "quants.py"
            dependency.write_text("old converter")
            dst = root / "pack"
            args = argparse.Namespace(gguf=str(src), base=None, out=str(dst),
                                      compat_bf16=False, experts_bin=False, skip_experts=False)

            def build(a):
                stage = Path(a.out)
                stage.mkdir()
                (stage / "dense.bin").write_bytes(src.read_bytes())
                return 0

            with patch.object(iq_pack, "Model", return_value=types.SimpleNamespace(paths=[src])), \
                    patch.object(iq_pack, "converter_paths", return_value={"gguf/quants.py": dependency}), \
                    patch.object(iq_pack, "build_pack", side_effect=build):
                self.assertEqual(iq_pack.transactional_pack(args, dst), 0)
                before = (dst / "PACK.json").read_bytes()
                dependency.write_text("new converter")
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    iq_pack.transactional_pack(args, dst)
                self.assertEqual((dst / "PACK.json").read_bytes(), before)
                self.assertEqual((dst / "dense.bin").read_bytes(), b"source")

    def test_source_change_and_failed_build_leave_no_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source"
            src.write_bytes(b"AAAA")
            dst = root / "pack"
            args = argparse.Namespace(gguf=str(src), base=None, out=str(dst),
                                      compat_bf16=False, experts_bin=False, skip_experts=False)

            def build(a):
                stage = Path(a.out)
                stage.mkdir()
                (stage / "dense.bin").write_bytes(src.read_bytes())
                src.write_bytes(b"BBBB")
                return 0

            with patch.object(iq_pack, "Model", return_value=types.SimpleNamespace(paths=[src])), \
                    patch.object(iq_pack, "build_pack", side_effect=build):
                with self.assertRaisesRegex(ValueError, "source changed during conversion"):
                    iq_pack.transactional_pack(args, dst)
            self.assertEqual(args.out, str(dst))
            self.assertFalse(dst.exists())
            self.assertFalse(list(root.glob(".stage-pack-*")))

            def failed(a):
                Path(a.out).mkdir()
                (Path(a.out) / "dense.bin").write_bytes(b"partial")
                return 1

            with patch.object(iq_pack, "Model", return_value=types.SimpleNamespace(paths=[src])), \
                    patch.object(iq_pack, "build_pack", side_effect=failed):
                self.assertEqual(iq_pack.transactional_pack(args, dst), 1)
            self.assertFalse(dst.exists())
            self.assertFalse(list(root.glob(".stage-pack-*")))

    def test_real_conversion_interruption_never_exposes_partial_pack(self):
        import contextlib
        import io
        import sys
        from test_iq_pack import split_model

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = split_model(root)
            dst = root / "published"
            argv = ["iq_pack.py", "--gguf", str(shards[0]), "--out", str(dst)]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(iq_pack, "write_index", side_effect=KeyboardInterrupt("interrupted")):
                with self.assertRaises(KeyboardInterrupt):
                    iq_pack.main()
            self.assertFalse(dst.exists())
            self.assertFalse(list(root.glob(".stage-published-*")))

    def test_base_dense_hardlink_does_not_count_as_source_content_change(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source"
            src.write_bytes(b"source")
            base = root / "base"
            base.mkdir()
            (base / "dense.bin").write_bytes(b"shared dense weights")
            dst = root / "pack"
            args = argparse.Namespace(gguf=str(src), base=str(base), out=str(dst),
                                      compat_bf16=False, experts_bin=False, skip_experts=False)

            def build(a):
                stage = Path(a.out)
                stage.mkdir()
                os.link(base / "dense.bin", stage / "dense.bin")
                return 0

            with patch.object(iq_pack, "Model", return_value=types.SimpleNamespace(paths=[src])), \
                    patch.object(iq_pack, "build_pack", side_effect=build):
                self.assertEqual(iq_pack.transactional_pack(args, dst), 0)
            self.assertEqual((dst / "dense.bin").read_bytes(), (base / "dense.bin").read_bytes())
            self.assertEqual((dst / "dense.bin").stat().st_ino, (base / "dense.bin").stat().st_ino)

    def test_reuse_rejects_source_mutation_after_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'source'
            src.write_bytes(b'AAAA')
            dst = root / 'pack'
            args = argparse.Namespace(gguf=str(src), base=None, out=str(dst),
                                      compat_bf16=False, experts_bin=False, skip_experts=False)

            def build(a):
                Path(a.out).mkdir()
                (Path(a.out) / 'dense.bin').write_bytes(b'AAAA')
                return 0

            with patch.object(iq_pack, 'Model', return_value=types.SimpleNamespace(paths=[src])), \
                    patch.object(iq_pack, 'build_pack', side_effect=build):
                self.assertEqual(iq_pack.transactional_pack(args, dst), 0)
                before = {p.name: p.read_bytes() for p in dst.iterdir()}
                real_identity = iq_pack.tree_identity

                def mutate_after_hash(path):
                    result = real_identity(path)
                    if path == dst:
                        src.write_bytes(b'BBBB')
                    return result

                with patch.object(iq_pack, 'tree_identity', side_effect=mutate_after_hash):
                    with self.assertRaisesRegex(ValueError, 'source changed during verification'):
                        iq_pack.transactional_pack(args, dst)
                self.assertEqual(before, {p.name: p.read_bytes() for p in dst.iterdir()})

    def test_setup_low_ram_publishes_final_options_once_and_preserves_old_pack(self):
        import contextlib
        import io
        import sys
        import setup
        from test_iq_pack import split_model, expected_experts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = split_model(root)
            commands = []

            def export(cmd, **kwargs):
                tokenizer = Path(cmd[cmd.index('--out') + 1]) / 'tokenizer'
                tokenizer.mkdir()
                (tokenizer / 'vocab.json').write_text('{}')
                (tokenizer / 'chat_template.jinja').write_text('fixture')

            def convert(cmd, **kwargs):
                commands.append(cmd)
                with patch.object(sys, 'argv', ['iq_pack.py', *cmd[2:]]):
                    self.assertEqual(iq_pack.main(), 0)

            with patch.object(setup, 'run', side_effect=convert), \
                    patch.object(iq_pack.subprocess, 'run', side_effect=export), \
                    contextlib.redirect_stdout(io.StringIO()):
                fresh = setup.prepare_native_pack(root / 'fresh', shards[0], [], True, {})
                self.assertEqual(len(commands), 1)
                self.assertTrue(json.loads((fresh / 'PACK.json').read_text())['identity']['experts_bin'])
                self.assertEqual((fresh / 'experts.bin').read_bytes(), expected_experts(iq_pack.Model(shards[0])))
                old = setup.prepare_native_pack(root / 'text', shards[0], [], False, {})
                before = {p.relative_to(old): p.read_bytes() for p in old.rglob('*') if p.is_file()}
                upgraded = setup.prepare_native_pack(old, shards[0], [], True, {})
                self.assertNotEqual(upgraded, old)
                self.assertEqual((upgraded / 'experts.bin').read_bytes(), (fresh / 'experts.bin').read_bytes())
                self.assertEqual(before, {p.relative_to(old): p.read_bytes() for p in old.rglob('*') if p.is_file()})
