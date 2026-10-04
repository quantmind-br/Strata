"""Managed launcher configuration, independent of GPU/model assets."""
import unittest
from pathlib import Path

from serve.server import engine_args, managed_config


class ManagedConfig(unittest.TestCase):
    def test_context_override_preserves_pack_split_and_original(self):
        cfg = {"cwd": "/models", "gpu": [0, 1], "layer_split": "24",
               "args": ["--pack", "pack", "--native", "target.gguf",
                        "--max-context", "4096", "--max-context", "8192"]}
        result = managed_config(cfg, "/models/target.gguf", 32768)
        args = engine_args(result)
        self.assertEqual(args.count("--max-context"), 1)
        self.assertEqual(args[args.index("--max-context") + 1], "32768")
        self.assertEqual(args[-2:], ["--layer-split", "24"])
        self.assertEqual(cfg["args"][-1], "8192")
        self.assertEqual(result["args"][:4], cfg["args"][:4])

    def test_model_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            managed_config({"args": ["--native", "one.gguf"]}, "two.gguf", None)

    def test_missing_or_duplicate_model_is_rejected(self):
        for args in ([], ["--native"], ["--native", "a", "--native", "a"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                managed_config({"args": args}, "a", None)

    def test_context_validation(self):
        for context in (0, -1):
            with self.subTest(context=context), self.assertRaises(ValueError):
                managed_config({"args": []}, None, context)
        with self.assertRaisesRegex(ValueError, "missing its value"):
            managed_config({"args": ["--max-context", "--kv", "int8"]}, None, 4096)

    def test_omitted_context_uses_config(self):
        args = ["--native", str(Path("model.gguf").resolve()), "--max-context", "4096"]
        self.assertEqual(managed_config({"args": args}, "model.gguf", None)["args"], args)
