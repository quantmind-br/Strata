"""Behavioral regressions for truthful controls and bounded cancellation; no GPU."""
import json
from pathlib import Path
import queue
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from unittest.mock import patch

from serve.controls import StopFilter, validate_request
from serve.server import ByteTokenizer, ChatTemplate, MockEngine, Service, StrataEngine, serve

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
        for control in [{'tool_choice':'required'}, {'response_format':{'type':'json_object'}},
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

    def test_silent_child_is_reaped_before_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            child=Path(tmp)/'child.py'
            child.write_text('import sys,time\nprint("READY 4096 stop",flush=True)\nfor line in sys.stdin:\n if line.startswith("GEN"): print("T 97",flush=True)\n')
            # Python -c makes --serve an argv item, matching the native protocol.
            wrapper=Path(tmp)/'child';wrapper.write_text('#!/bin/sh\nexec '+sys.executable+' '+str(child)+'\n');wrapper.chmod(0o755)
            engine=StrataEngine(str(wrapper),[])
            try:
                gen=engine.generate([1],10,{},threading.Event());self.assertEqual(next(gen),97)
                clock=time.monotonic
                with patch('serve.server.time.monotonic',side_effect=[clock(),clock()+11]):
                    gen.close()
                self.assertFalse(engine.alive());self.assertIsNotNone(engine.proc.returncode)
                engine.restart();self.assertTrue(engine.alive())
            finally:engine._terminate()
