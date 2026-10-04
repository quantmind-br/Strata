"""Behavioral regressions for truthful controls and bounded cancellation; no GPU."""
import json
from pathlib import Path
import socket
import threading
import time
import unittest
import urllib.request
import urllib.error

from serve.controls import StopFilter, validate_request, validate_sampling
from serve.server import (ByteTokenizer, ChatTemplate, MockEngine, Service, StrataEngine, serve,
                          anthropic_collect, anthropic_events, clean_shared_defaults, sampling_defaults_from_config)

ROOT = Path(__file__).resolve().parents[1]


class Controls(unittest.TestCase):
    def test_stop_every_boundary_and_unicode(self):
        text = 'ação before<END>must not leak'
        for split in range(len(text)+1):
            filt = StopFilter(['<END>', 'unused'])
            actual = filt.feed(text[:split]) + filt.feed(text[split:]) + filt.finish()
            self.assertEqual(actual, 'ação before')
        filt = StopFilter(['xyxyz'])
        self.assertEqual(''.join(filt.feed(t) for t in 'axyxyxyzafter') + filt.finish(), 'axy')
        filt = StopFilter(['xyz'])
        self.assertEqual(filt.feed('abxy') + filt.finish(), 'abxy')

    def test_controls_are_explicit(self):
        base = {'messages': [{'role':'user','content':'hi'}]}
        for control in [{'tool_choice':'required'},
                        {'seed':True}, {'seed':-1}, {'seed':2**64}, {'top_k':0}, {'top_k':65},
                        {'temperature':float('nan')}, {'temperature':'1'}, {'stop':['']},
                        {'max_tokens':2.5}, {'n':2}, {'seed':10**1000}, {'temperature':10**1000},
                        {'strata_tune':{'pcie_frac':10**1000}}]:
            with self.subTest(control=control), self.assertRaises(ValueError):
                validate_request({**base, **control})
        validate_request({**base, 'seed':0, 'temperature':0, 'top_k':64, 'stop':['done']})
        self.assertIn(' seed=0', StrataEngine.sampling_keys({'seed':0}))

    def test_http_validation_precedes_stream(self):
        tok=ByteTokenizer(); engine=MockEngine(tok,'hello')
        svc=Service(engine,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
        httpd=serve(svc,port=0)
        try:
            url=f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions'
            for stream in [False,True]:
                req={'messages':[{'role':'user','content':'hi'}], 'stream':stream,'tool_choice':'required'}
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(req).encode()),timeout=3)
                with caught.exception as response:
                    self.assertEqual(response.code,400)
                    self.assertEqual(json.load(response)['error']['param'],'tool_choice')
            self.assertEqual(svc.totals['requests'],0)
        finally:
            httpd.shutdown();httpd.server_close()

    def test_stop_nonstream(self):
        tok=ByteTokenizer(); svc=Service(MockEngine(tok,'before<END>after'),tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
        result=list(svc.run([1],False,None,100,{'stop':'<END>'},threading.Event()))
        self.assertEqual(''.join(ev.text for kind,ev in result if kind=='event'),'before')
        self.assertEqual(result[-1][1]['finish'],'stop')
        self.assertEqual(result[-1][1]['stop_sequence'], '<END>')

    def test_default_admission_rejects_unsupported_sampling_bounds(self):
        bad = {'temperature': 3, 'presence_penalty': 3, 'frequency_penalty': 3,
               'repetition_penalty': 1e6 + 1, 'penalty_last_n': 2**31, 'seed': 2**64,
               'top_k': 65}
        for key, value in bad.items():
            with self.subTest(key=key), self.assertRaises(SystemExit):
                sampling_defaults_from_config({'sampling': {key: value}})
        for key, value in {'temperature': 3, 'max_tokens': 2**31, 'seed': 2**64, 'top_k': 65}.items():
            with self.subTest(shared=key), self.assertRaises(ValueError):
                clean_shared_defaults({key: value})
        valid = {'temperature': 2, 'presence_penalty': 2, 'frequency_penalty': 2,
                 'repetition_penalty': 1e6, 'penalty_last_n': 2**31 - 1, 'seed': 2**64 - 1,
                 'top_k': 64}
        admitted = sampling_defaults_from_config({'sampling': valid})
        self.assertEqual(admitted, valid)
        validate_sampling(admitted)
        shared = clean_shared_defaults({'seed': 0, 'max_tokens': 2**31 - 1, 'temperature': 2})
        self.assertEqual(shared['seed'], 0)
        validate_sampling(shared)

    def test_anthropic_stop_metadata_survives_stream_and_collect(self):
        for stops, reason, delimiter, text in [(['<END>'], 'stop_sequence', '<END>', 'before'),
                                               (['absent'], 'end_turn', None, 'before<END>after')]:
            tok = ByteTokenizer()
            svc = Service(MockEngine(tok, 'before<END>after'), tok, ChatTemplate(ROOT/'serve/chat_template.jinja'))
            req = {'stop_sequences': stops}
            events = list(anthropic_events(svc, req, [1], False, None, 100, threading.Event()))
            delta = next(e['delta'] for name, e in events if name == 'message_delta')
            self.assertEqual(delta, {'stop_reason': reason, 'stop_sequence': delimiter})
            out = anthropic_collect(iter(events))
            self.assertEqual(out['stop_reason'], reason)
            self.assertEqual(out['stop_sequence'], delimiter)
            self.assertEqual(''.join(b.get('text', '') for b in out['content']), text)

    def test_cancel_while_queued_does_not_start_engine(self):
        tok=ByteTokenizer(); svc=Service(MockEngine(tok,'hello'),tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
        cancel=threading.Event();svc.fifo.acquire()
        thread=threading.Thread(target=lambda:list(svc.run([1],False,None,100,{},cancel)))
        thread.start()
        time.sleep(.05);cancel.set();thread.join(1)
        try:
            self.assertFalse(thread.is_alive());self.assertEqual(svc.status['queued'],0)
            self.assertEqual(svc.totals['requests'],0)
        finally:svc.fifo.release()

    def test_nonstream_disconnect_cancels(self):
        tok=ByteTokenizer(); engine=MockEngine(tok,'x'*2000,delay_s=.01)
        svc=Service(engine,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
        httpd=serve(svc,port=0)
        try:
            req=json.dumps({'messages':[{'role':'user','content':'hi'}],'max_tokens':1000}).encode()
            sock=socket.create_connection(httpd.server_address)
            sock.sendall(b'POST /v1/chat/completions HTTP/1.0\r\nContent-Length: '+str(len(req)).encode()+b'\r\n\r\n'+req)
            deadline=time.monotonic()+2
            while not svc.status['busy'] and time.monotonic()<deadline:time.sleep(.01)
            sock.close()
            deadline=time.monotonic()+2
            while svc.status['busy'] and time.monotonic()<deadline:time.sleep(.01)
            self.assertFalse(svc.status['busy']);self.assertLess(svc.totals['output_tokens'],100)
        finally:httpd.shutdown();httpd.server_close()

