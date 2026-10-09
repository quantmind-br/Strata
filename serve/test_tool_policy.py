"""Forced tool choice and the single-call limit: honored exactly or refused, never silently ignored; no GPU."""
import json
from pathlib import Path
import unittest
import urllib.error
import urllib.request

from serve.controls import single_call, validate_request
from serve.frontend import CALL_START, forced_call
from serve.test_server import CallingEngine
from serve.server import ByteTokenizer, ChatTemplate, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
TOOL = {'type': 'function', 'function': {'name': 'execute', 'description': 'Run a command',
        'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}
OTHER = {'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object', 'properties': {}}}}
BODY = '<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>'
TWO_CALLS = ('<tool_call>\n<function=execute>\n' + BODY + '\n<tool_call>\n<function=execute>\n'
             '<parameter=command>\nrm -rf x\n</parameter>\n</function>\n</tool_call>\ntrailing text')
CALL_PREFIX = CALL_START + '\n<function='


def request(**extra):
    return {'messages': [{'role': 'user', 'content': 'list files'}], 'reasoning_effort': 'none', **extra}


class Policy(unittest.TestCase):
    def test_validation_accepts_only_honored_choices(self):
        ok = [request(tools=[TOOL], tool_choice='required'),
              request(tools=[TOOL], tool_choice={'type': 'function', 'function': {'name': 'execute'}}),
              request(tools=[TOOL], parallel_tool_calls=False),
              request(tools=[TOOL], tool_choice='none')]
        for req in ok:
            with self.subTest(req=req):
                validate_request(req)
        anthropic_ok = [request(tools=[TOOL['function']], tool_choice={'type': 'any'}),
                        request(tools=[TOOL['function']], tool_choice={'type': 'tool', 'name': 'execute'}),
                        request(tools=[TOOL['function']], tool_choice={'type': 'auto', 'disable_parallel_tool_use': True})]
        for req in anthropic_ok:
            with self.subTest(req=req):
                validate_request(req, 'anthropic')
        bad = [request(tool_choice='required'),
               request(tools=[TOOL], tool_choice={'type': 'function', 'function': {'name': 'missing'}}),
               request(tools=[TOOL], tool_choice={'type': 'allowed_tools'}),
               request(tools=[TOOL], tool_choice='required', strata_mcp=True),
               request(tools=[TOOL], parallel_tool_calls=False, strata_mcp=True)]
        for req in bad:
            with self.subTest(req=req), self.assertRaises(ValueError):
                validate_request(req)
        validate_request(request(tools=[TOOL['function']], tool_choice={'type': 'none'}), 'anthropic')

    def test_policy(self):
        tools = [TOOL['function'], OTHER['function']]
        self.assertIsNone(forced_call('auto', tools))
        self.assertEqual(forced_call('required', tools[:1]), CALL_PREFIX + 'execute>\n')
        self.assertEqual(forced_call('required', tools), CALL_PREFIX)
        self.assertEqual(forced_call({'type': 'any'}, tools[:1]), CALL_PREFIX + 'execute>\n')
        self.assertTrue(single_call(request(parallel_tool_calls=False)))
        self.assertTrue(single_call(request(tool_choice={'type': 'tool', 'name': 'execute',
                                                         'disable_parallel_tool_use': True}), 'anthropic'))


class Http(unittest.TestCase):
    def start(self, script):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, script)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        return tok

    def post(self, path, req):
        url = f'http://127.0.0.1:{self.httpd.server_address[1]}{path}'
        with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(req).encode()), timeout=5) as r:
            return json.load(r)

    def stream(self, req):
        url = f'http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions'
        chunks = []
        with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps({**req, 'stream': True}).encode()),
                                    timeout=5) as r:
            for line in r:
                if line.startswith(b'data: {'):
                    chunks.append(json.loads(line[6:]))
        return chunks

    def assert_refused(self, path, req, param=None):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post(path, req)
        with caught.exception as response:
            self.assertEqual(response.code, 400)
            if param:
                self.assertEqual(json.load(response)['error']['param'], param)

    def test_named_choice_forces_the_call(self):
        tok = self.start(BODY)
        named = {'type': 'function', 'function': {'name': 'execute'}}
        out = self.post('/v1/chat/completions', request(tools=[TOOL], tool_choice=named))
        self.assertTrue(tok.decode(self.engine.last_prompt).endswith('</think>\n\n' + CALL_PREFIX + 'execute>\n'))
        choice = out['choices'][0]
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        self.assertEqual([c['function']['name'] for c in choice['message']['tool_calls']], ['execute'])
        self.assertEqual(json.loads(choice['message']['tool_calls'][0]['function']['arguments']), {'command': 'ls -la'})
        self.assertIsNone(choice['message']['content'])

    def test_required_choice_with_one_tool_names_it(self):
        tok = self.start(BODY)
        out = self.post('/v1/chat/completions', request(tools=[TOOL], tool_choice='required'))
        self.assertTrue(tok.decode(self.engine.last_prompt).endswith(CALL_PREFIX + 'execute>\n'))
        self.assertEqual(out['choices'][0]['message']['tool_calls'][0]['function']['name'], 'execute')

    def test_required_choice_lets_the_model_name_an_offered_tool(self):
        self.start('execute>\n' + BODY)
        out = self.post('/v1/chat/completions', request(tools=[TOOL, OTHER], tool_choice='required'))
        self.assertEqual(out['choices'][0]['message']['tool_calls'][0]['function']['name'], 'execute')

    def test_forced_choice_refuses_unknown_name_and_incomplete_call(self):
        self.start('shutdown>\n' + BODY)
        self.assert_refused('/v1/chat/completions', request(tools=[TOOL, OTHER], tool_choice='required'))
        self.start('no call here')
        self.assert_refused('/v1/chat/completions',
                            request(tools=[TOOL], tool_choice={'type': 'function', 'function': {'name': 'execute'}}))

    def test_forced_choice_after_thinking(self):
        tok = self.start(BODY)
        self.svc.engine = self.engine = CallingEngine(tok)
        tool = {'type': 'function', 'function': {'name': 'search', 'parameters': {
            'type': 'object', 'properties': {'q': {'type': 'string'}}}}}
        out = self.post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': '2+2?'}],
                                               'tools': [tool], 'tool_choice': 'required', 'max_tokens': 400})
        call = out['choices'][0]['message']['tool_calls'][0]['function']
        self.assertEqual((call['name'], json.loads(call['arguments'])), ('search', {'q': '2+2'}))
        self.assertEqual(len(self.engine.prompts), 2)

    def test_single_call_limit_returns_only_the_first_complete_call(self):
        self.start(TWO_CALLS)
        req = request(tools=[TOOL], parallel_tool_calls=False)
        out = self.post('/v1/chat/completions', req)
        msg = out['choices'][0]['message']
        self.assertEqual(out['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(len(msg['tool_calls']), 1)
        self.assertEqual(json.loads(msg['tool_calls'][0]['function']['arguments']), {'command': 'ls -la'})
        self.assertIsNone(msg['content'])
        self.assertLess(out['usage']['completion_tokens'], len(TWO_CALLS.encode()))
        chunks = self.stream(req)
        streamed = [c for ch in chunks for c in ch['choices'][0]['delta'].get('tool_calls', [])]
        indices = {c['index'] for c in streamed}
        ids = {c['id'] for c in streamed if c.get('id')}
        names = [c['function']['name'] for c in streamed if c.get('function', {}).get('name')]
        arguments = ''.join(c.get('function', {}).get('arguments', '') for c in streamed)
        self.assertEqual(indices, {0})
        self.assertEqual(len(ids), 1)
        self.assertEqual(names, ['execute'])
        self.assertEqual(json.loads(arguments), {'command': 'ls -la'})
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertFalse(any('rm -rf' in json.dumps(ch) or 'trailing' in json.dumps(ch) for ch in chunks))

    def test_without_limit_both_calls_are_returned(self):
        self.start(TWO_CALLS)
        out = self.post('/v1/chat/completions', request(tools=[TOOL]))
        self.assertEqual(len(out['choices'][0]['message']['tool_calls']), 2)

    def test_double_encoded_tools_survive_admission(self):
        self.start(BODY)
        out = self.post('/v1/chat/completions', request(tools=json.dumps([TOOL]), tool_choice='required'))
        self.assertEqual(out['choices'][0]['message']['tool_calls'][0]['function']['name'], 'execute')

    def test_single_call_limit_applies_to_reasoning_rescue(self):
        self.start(TWO_CALLS.split('\ntrailing text')[0])
        out = self.post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'list files'}],
                                               'tools': [TOOL], 'parallel_tool_calls': False, 'max_tokens': 1000})
        self.assertEqual(len(out['choices'][0]['message']['tool_calls']), 1)
        self.assertEqual(out['choices'][0]['finish_reason'], 'tool_calls')

    def test_forced_recovery_does_not_downgrade_bad_arguments(self):
        tool = {'type': 'function', 'function': {'name': 'count', 'parameters': {
            'type': 'object', 'properties': {'value': {'type': 'integer'}}}}}
        self.start('<parameter=value>oops</parameter>\n</function>\n</tool_call>')
        self.svc.tool_call_recovery = True
        self.assert_refused('/v1/chat/completions', request(tools=[tool], tool_choice='required'))

    def test_malformed_named_choices_return_param_errors_before_stream(self):
        self.start(BODY)
        for function in ('execute', ['execute'], 7):
            for stream in (False, True):
                with self.subTest(function=function, stream=stream):
                    self.assert_refused('/v1/chat/completions', request(
                        tools=[TOOL], stream=stream,
                        tool_choice={'type': 'function', 'function': function}), 'tool_choice')
        self.assertEqual(self.svc.totals['requests'], 0)

    def test_recovered_call_keeps_arguments_and_validates_declared_type(self):
        tool = {'type': 'function', 'function': {'name': 'count', 'parameters': {
            'type': 'object', 'properties': {'value': {'type': 'integer'}}}}}
        for value in ('42', 'oops'):
            self.start('<parameter=value>\n' + value + '\n</parameter>\n</function>')
            req = request(tools=[tool], tool_choice='required')
            if value == 'oops':
                self.assert_refused('/v1/chat/completions', req)
                chunks = self.stream(req)
                errors = [c['error']['message'] for c in chunks if 'error' in c]
                self.assertEqual(errors, ['tool count: parameter value does not match declared type integer'])
                self.assertFalse(any(c.get('choices', [{}])[0].get('finish_reason') == 'tool_calls' for c in chunks))
            else:
                out = self.post('/v1/chat/completions', req)
                call = out['choices'][0]['message']['tool_calls'][0]
                self.assertEqual(json.loads(call['function']['arguments']), {'value': 42})
                self.assertEqual(out['choices'][0]['finish_reason'], 'tool_calls')
                chunks = self.stream(req)
                pieces = [c for ch in chunks for c in ch['choices'][0]['delta'].get('tool_calls', [])]
                self.assertEqual(json.loads(''.join(c.get('function', {}).get('arguments', '') for c in pieces)),
                                 {'value': 42})
                self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'tool_calls')

    def test_anthropic_forced_tool_and_single_call(self):
        self.start(BODY + '\n<tool_call>\n<function=execute>\n' + BODY)
        req = {'model': 'm', 'max_tokens': 200, 'messages': [{'role': 'user', 'content': 'list files'}],
               'thinking': {'type': 'disabled'}, 'tools': [{**TOOL['function'], 'input_schema': TOOL['function']['parameters']}],
               'tool_choice': {'type': 'tool', 'name': 'execute', 'disable_parallel_tool_use': True}}
        out = self.post('/v1/messages', req)
        uses = [b for b in out['content'] if b['type'] == 'tool_use']
        self.assertEqual(len(uses), 1)
        self.assertEqual(uses[0]['input'], {'command': 'ls -la'})
        self.assertEqual(out['stop_reason'], 'tool_use')
        self.svc.engine = self.engine = CallingEngine(self.svc.tok)
        req['tools'] = [{'name': 'search', 'input_schema': {'type': 'object',
                         'properties': {'q': {'type': 'string'}}}}]
        req['tool_choice'] = {'type': 'tool', 'name': 'search', 'disable_parallel_tool_use': True}
        req['thinking'] = {'type': 'enabled', 'budget_tokens': 4096}
        req['max_tokens'] = 400
        out = self.post('/v1/messages', req)
        uses = [b for b in out['content'] if b['type'] == 'tool_use']
        self.assertEqual([(b['name'], b['input']) for b in uses], [('search', {'q': '2+2'})])


if __name__ == '__main__':
    unittest.main()
