"""The call result written by Claude (call_result.ClaudeSummarizer), through the real SDK over a
fake HTTP connection."""
import asyncio
import json
import unittest

import anthropic
import httpx2

from call_result import (
    MEETING_SUMMARY_INSTRUCTIONS, RESULT_SCHEMA, SUMMARY_INSTRUCTIONS, ClaudeSummarizer,
    SummaryUnavailable, meeting_facts,
)

TRANSCRIPT = [{'speaker': 'agent', 'text': 'Grüezi, ich rufe im Auftrag von Robin an.'},
              {'speaker': 'other', 'text': 'Ja, 19 Uhr geht.'}]
BRIEF = {'objective': 'Einen Tisch für vier um 19 Uhr reservieren', 'onBehalfOf': 'Robin'}
RESULT = {'outcome': 'achieved', 'summary': 'Der Tisch ist für 19 Uhr reserviert.',
          'details': [{'label': 'Reservation', 'value': 'Hunziker, 19 Uhr'}, {'label': '', 'value': 'x'}],
          'decisions': ['19 Uhr'], 'actionItems': [], 'openQuestions': [], 'doNotCall': False}


def message(text, *, stop_reason='end_turn', usage=None):
    return {
        'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': 'claude-haiku-5-5',
        'content': [{'type': 'thinking', 'thinking': '', 'signature': 'sig'},
                    {'type': 'text', 'text': text}],
        'stop_reason': stop_reason, 'stop_sequence': None,
        'usage': usage or {'input_tokens': 300, 'output_tokens': 120,
                           'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0},
    }


class ClaudeSummaryTests(unittest.TestCase):
    def summarizer(self, *responses):
        self.bodies = []
        responses = list(responses)

        def handler(request):
            self.bodies.append(json.loads(request.content))
            status, body = responses.pop(0)
            return httpx2.Response(status, json=body)

        client = anthropic.AsyncAnthropic(
            api_key='test-key', max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
        return ClaudeSummarizer(client=client)

    def summarize(self, summarizer, brief=BRIEF, transcript=TRANSCRIPT, **kwargs):
        return asyncio.run(summarizer.summarize(brief, transcript, **kwargs))

    def test_claude_writes_the_result_as_json_that_fits_the_schema(self):
        usage = {'input_tokens': 300, 'output_tokens': 120, 'cache_read_input_tokens': 50,
                 'cache_creation_input_tokens': 10}
        result = self.summarize(self.summarizer((200, message(json.dumps(RESULT), usage=usage))),
                                duration_seconds=65)
        body = self.bodies[0]
        self.assertEqual(body['model'], 'claude-haiku-5-5')
        self.assertEqual(body['output_config'], {
            'effort': 'medium', 'format': {'type': 'json_schema', 'schema': RESULT_SCHEMA}})
        self.assertTrue(body['system'].startswith(SUMMARY_INSTRUCTIONS))
        self.assertIn('in the language the brief is written in', body['system'])
        self.assertEqual([turn['role'] for turn in body['messages']], ['user'])
        text = body['messages'][0]['content']
        self.assertIn('Objective: Einen Tisch für vier um 19 Uhr reservieren', text)
        self.assertIn('Other party: Ja, 19 Uhr geht.', text)
        self.assertNotIn('Facts from Smitline', text)
        self.assertEqual(result['outcome'], 'achieved')
        self.assertEqual(result['summary'], 'Der Tisch ist für 19 Uhr reserviert.')
        self.assertEqual(result['details'], [{'label': 'Reservation', 'value': 'Hunziker, 19 Uhr'}])
        self.assertEqual(result['durationSeconds'], 65)
        self.assertEqual(result['source'], 'summary_model')
        self.assertEqual(len(result['transcript']), 2)
        # Cache reads and writes are input too.
        self.assertEqual(result['summaryTokens'], {'input': 360, 'cached': 50, 'output': 120})
        self.assertEqual(result['summaryModel'], 'claude-haiku-5-5')

    def test_a_meeting_gets_the_meeting_rules_and_the_facts(self):
        facts = meeting_facts(room_created=True, participants={'count': 2, 'names': ['Ana', 'Ben']})
        result = self.summarize(
            self.summarizer((200, message(json.dumps(dict(RESULT, doNotCall=True))))),
            brief={'channel': 'meeting', 'objective': 'Fragen beantworten', 'onBehalfOf': 'Robin'},
            facts=facts)
        body = self.bodies[0]
        self.assertTrue(body['system'].startswith(MEETING_SUMMARY_INSTRUCTIONS))
        text = body['messages'][0]['content']
        self.assertIn('Attending on behalf of: Robin', text)
        self.assertIn('Facts from Smitline:\n- The assistant set up this meeting room itself', text)
        self.assertIn('Assistant: Grüezi', text)
        self.assertNotIn('doNotCall', result)  # nobody can ask a meeting not to call again

    def test_a_summary_that_cannot_be_written_is_unavailable(self):
        cases = [
            ((200, message('', stop_reason='refusal')), 'Claude declined'),
            ((200, message('{"outcome": "achi', stop_reason='max_tokens')), 'cut off'),
            ((200, message('nope')), 'invalid JSON'),
            ((200, message('[]')), 'invalid JSON'),
            ((401, {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'bad key'}}),
             'summary request failed (claude_auth)'),
            ((529, {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'busy'}}),
             'summary request failed (claude_http_529)'),
        ]
        for response, why in cases:
            with self.subTest(why=why):
                with self.assertRaises(SummaryUnavailable) as caught:
                    self.summarize(self.summarizer(response))
                self.assertIn(why, str(caught.exception))

    def test_without_a_key_there_is_no_summarizer(self):
        for key in (None, '', '  '):
            with self.assertRaises(SummaryUnavailable) as caught:
                ClaudeSummarizer(key)
            self.assertIn('ANTHROPIC_API_KEY', str(caught.exception))
        self.assertEqual(ClaudeSummarizer(' sk-ant-test ').client.api_key, 'sk-ant-test')


if __name__ == '__main__':
    unittest.main()
