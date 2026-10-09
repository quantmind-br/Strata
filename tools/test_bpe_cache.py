import unittest
from concurrent.futures import ThreadPoolExecutor
import random
from strata_tokenizer import Tokenizer, BYTE_TO_UNICODE


class BPECache(unittest.TestCase):
    def test_exactness_and_tokenizer_isolation(self):
        vocab=list(BYTE_TO_UNICODE.values())+['ab','abc']
        a=Tokenizer(vocab,['a b','ab c']);b=Tokenizer(vocab,[])
        for text in ['abc '*500,'ação 日本語 🙂\n'*50, 'a'*300, '<|not_special|>\t']:
            expected=[]
            for piece in a._re.findall(text):
                mapped=''.join(BYTE_TO_UNICODE[n] for n in piece.encode())
                expected.extend(a.ids[t] for t in a._bpe(mapped))
            self.assertEqual(a.encode(text),expected)
            self.assertEqual(a.encode(text),expected)
        self.assertNotEqual(a.encode('abc'),b.encode('abc'))

    def test_parallel_encoding_matches_uncached_high_entropy(self):
        vocab=list(BYTE_TO_UNICODE.values())+['ab','abc','<|im_end|>']
        cached=Tokenizer(vocab,['a b','ab c'])
        oracle=Tokenizer(vocab,['a b','ab c'])
        oracle.PIECE_CACHE_MAX = 0
        rng=random.Random(0)
        corpus=[''.join(rng.choices('abcXYZ0123456789!? ação🙂日本語',k=200)) for _ in range(100)]
        corpus+=['abc<|im_end|>ação','a'*300]
        expected=[oracle.encode(text,parse_special=True) for text in corpus]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual=list(pool.map(lambda text:cached.encode(text,parse_special=True),corpus))
        self.assertEqual(actual,expected)


    def test_cached_special_token_boundaries_preserve_ids_and_decode(self):
        vocab = list(BYTE_TO_UNICODE.values()) + ['ab', '<|im_end|>', '<tool_call>']
        types = [1] * (len(vocab) - 2) + [3, 4]
        tokenizer = Tokenizer(vocab, ['a b'], types)
        text = 'ab<|im_end|>ab<tool_call>ab'
        expected = [tokenizer.ids['ab'], tokenizer.ids['<|im_end|>'],
                    tokenizer.ids['ab'], tokenizer.ids['<tool_call>'], tokenizer.ids['ab']]
        for _ in range(2):
            self.assertEqual(tokenizer.encode(text, parse_special=True), expected)
            self.assertEqual(tokenizer.decode(expected), text)
            ordinary = tokenizer.encode(text, parse_special=False)
            self.assertNotIn(tokenizer.ids['<|im_end|>'], ordinary)
            self.assertIn(tokenizer.ids['<tool_call>'], ordinary)
            self.assertEqual(tokenizer.decode(ordinary), text)
