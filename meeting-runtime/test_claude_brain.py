"""Claude as the brain of a call, against a fake Anthropic client: no key, no network."""
import asyncio
import json
import unittest
from types import SimpleNamespace

import anthropic
import httpx2

from call_brief import CallBrief
from call_locale import STANDARD_GERMAN, SWISS_GERMAN
from claude_brain import (
    END_CALL_TOOL, END_REASONS, MODEL, OPENING_TURN, QUIET_TURN, ClaudeBrain, SentenceSplitter,
    failure_code, refusal_line, trouble_line,
)
from claude_prompts import system_prompt
from phone_prompts import discloses
from voice_chain import VoiceBrain


def brief(**overrides):
    payload = {
        'channel': 'phone', 'to': '+41441234567', 'onBehalfOf': 'Florian Berth',
        'objective': 'Einen Tisch für vier Personen am Freitag um 19 Uhr reservieren',
    }
    payload.update(overrides)
    return CallBrief.from_dict(payload)


def text(value):
    return SimpleNamespace(type='text', text=value)


def thinking():
    return SimpleNamespace(type='thinking', thinking='', signature='sig')


def end_call(reason, tool_id='toolu_1'):
    return SimpleNamespace(type='tool_use', id=tool_id, name='end_call', input={'reason': reason})


def message(*content, stop_reason='end_turn', input_tokens=100, output_tokens=20,
            cache_read=0):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                            cache_read_input_tokens=cache_read, cache_creation_input_tokens=0)
    return SimpleNamespace(content=list(content), stop_reason=stop_reason, usage=usage)


def connection_error():
    return anthropic.APIConnectionError(
        request=httpx2.Request('POST', 'https://api.anthropic.com/v1/messages'))


def status_error(cls, status):
    request = httpx2.Request('POST', 'https://api.anthropic.com/v1/messages')
    return cls('error', response=httpx2.Response(status, request=request), body=None)


class FakeStream:
    """Streams DELTAS as text events, then hands back FINAL; ERROR is raised on entering."""

    def __init__(self, deltas=(), final=None, *, error=None):
        self.deltas = list(deltas)
        self.final = final if final is not None else message(text(''.join(self.deltas)))
        self.error = error
        self.sent = 0
        self.closed = False

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def __aiter__(self):
        yield SimpleNamespace(type='message_start')
        for delta in self.deltas:
            self.sent += 1
            yield SimpleNamespace(type='text', text=delta)
            await asyncio.sleep(0)

    async def get_final_message(self):
        return self.final


class FakeClient:
    """Stands in for anthropic.AsyncAnthropic: one FakeStream per request, requests kept."""

    def __init__(self, *streams):
        self.streams = list(streams)
        self.requests = []
        self.messages = self

    def stream(self, **request):
        self.requests.append({**request, 'messages': [dict(m) for m in request['messages']]})
        return self.streams.pop(0)


def run(coro):
    return asyncio.run(coro)


async def collect(iterator):
    return [event async for event in iterator]


def said(events):
    return [event.text for event in events if event.kind == 'say']


class SentenceSplitterTests(unittest.TestCase):
    def test_sentences_come_out_as_soon_as_they_are_complete(self):
        splitter = SentenceSplitter()
        self.assertEqual(splitter.add('Grüezi Herr Hunziker'), [])
        self.assertEqual(splitter.add('.'), [])  # Could still be "...".
        self.assertEqual(splitter.add(' Ich rufe'), ['Grüezi Herr Hunziker. '])
        self.assertEqual(splitter.add(' an'), [])
        self.assertEqual(splitter.add('! Passt das? Ja'), ['Ich rufe an! ', 'Passt das? '])
        self.assertEqual(splitter.rest(), 'Ja')
        self.assertEqual(splitter.rest(), '')

    def test_a_full_stop_after_a_number_does_not_end_the_sentence(self):
        splitter = SentenceSplitter()
        self.assertEqual(splitter.add('Am 1. Mai passt es. Danke. '),
                         ['Am 1. Mai passt es. ', 'Danke. '])

    def test_a_line_break_ends_a_sentence(self):
        self.assertEqual(SentenceSplitter().add('Erstens\nZweitens'), ['Erstens\n'])


class RequestTests(unittest.TestCase):
    def test_a_turn_asks_haiku_at_low_effort_with_the_german_prompt_and_end_call(self):
        client = FakeClient(FakeStream(['Grüezi.']))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client)
        run(collect(brain.reply(None)))
        request = client.requests[0]
        self.assertEqual(request['model'], MODEL)
        self.assertEqual(MODEL, 'claude-haiku-5-5')
        self.assertEqual(request['system'], system_prompt(brief(), SWISS_GERMAN))
        self.assertEqual(request['tools'], [END_CALL_TOOL])
        self.assertEqual(request['thinking'], {'type': 'adaptive'})
        self.assertEqual(request['output_config'], {'effort': 'low'})
        self.assertEqual(request['cache_control'], {'type': 'ephemeral'})
        for removed in ('temperature', 'top_p', 'top_k'):
            self.assertNotIn(removed, request)

    def test_end_call_takes_only_the_known_reasons(self):
        schema = END_CALL_TOOL['input_schema']
        self.assertEqual(schema['properties']['reason']['enum'], list(END_REASONS))
        self.assertFalse(schema['additionalProperties'])
        self.assertTrue(END_CALL_TOOL['strict'])

    def test_the_contact_name_reaches_the_prompt(self):
        client = FakeClient(FakeStream(['Grüezi.']))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client, contact={'name': 'Frau Meier'})
        self.assertIn('Frau Meier', brain.system)

    def test_the_opening_turn_says_nobody_has_spoken_yet(self):
        client = FakeClient(FakeStream(['Grüezi.']))
        run(collect(ClaudeBrain(brief(), SWISS_GERMAN, client=client).reply(None)))
        self.assertEqual(client.requests[0]['messages'],
                         [{'role': 'user', 'content': [{'type': 'text', 'text': OPENING_TURN}]}])

    def test_without_a_client_it_needs_an_api_key(self):
        with self.assertRaises(ValueError):
            ClaudeBrain(brief(), SWISS_GERMAN)
        with self.assertRaises(ValueError):
            ClaudeBrain(brief(), SWISS_GERMAN, api_key='  ')
        brain = ClaudeBrain(brief(), SWISS_GERMAN, api_key='test-key')
        self.assertIsInstance(brain.client, anthropic.AsyncAnthropic)

    def test_it_fits_the_voice_brain_interface(self):
        self.assertIsInstance(ClaudeBrain(brief(), SWISS_GERMAN, client=FakeClient()), VoiceBrain)


class ReplyTests(unittest.TestCase):
    def test_the_first_sentence_is_ready_before_claude_has_finished(self):
        stream = FakeStream(['Grüezi Herr Hunziker, ich rufe im Auftrag ',
                             'von Florian Berth an. Ich bin ', 'sein KI-Assistent. Es geht ',
                             'um eine Reservation.'])
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=FakeClient(stream))

        async def scenario():
            replies = brain.reply(None)
            first = await anext(replies)
            sent_by_then = stream.sent
            rest = await collect(replies)
            return first, sent_by_then, rest

        first, sent_by_then, rest = run(scenario())
        self.assertEqual(first.text, 'Grüezi Herr Hunziker, ich rufe im Auftrag von Florian Berth an.')
        self.assertLess(sent_by_then, 4)
        self.assertEqual(said(rest), ['Ich bin sein KI-Assistent.', 'Es geht um eine Reservation.'])
        self.assertEqual(rest[-1].kind, 'done')

    def test_text_is_cleaned_for_speaking_with_swiss_spelling(self):
        stream = FakeStream(['**Grüße** an Sie 😊. Schönen ', 'Tag noch'])
        events = run(collect(ClaudeBrain(brief(), SWISS_GERMAN, client=FakeClient(stream))
                             .reply('Danke.')))
        self.assertEqual(said(events), ['Grüsse an Sie.', 'Schönen Tag noch'])

    def test_german_calls_keep_the_sharp_s(self):
        stream = FakeStream(['Grüße an Sie.'])
        events = run(collect(ClaudeBrain(brief(to='+4930123456'), STANDARD_GERMAN,
                                         client=FakeClient(stream)).reply('Danke.')))
        self.assertEqual(said(events), ['Grüße an Sie.'])

    def test_end_call_comes_after_the_goodbye(self):
        final = message(thinking(), text('Merci, auf Wiederhören.'), end_call('completed'))
        stream = FakeStream(['Merci, auf Wiederhören.'], final)
        events = run(collect(ClaudeBrain(brief(), SWISS_GERMAN, client=FakeClient(stream))
                             .reply('Gut, bis Freitag.')))
        self.assertEqual([event.kind for event in events], ['say', 'tool', 'done'])
        self.assertEqual((events[1].name, events[1].arguments), ('end_call', {'reason': 'completed'}))

    def test_an_unknown_end_reason_becomes_other(self):
        final = message(text('Adieu.'), end_call('fertig'))
        events = run(collect(ClaudeBrain(brief(), SWISS_GERMAN,
                                         client=FakeClient(FakeStream(['Adieu.'], final)))
                             .reply('Tschüss.')))
        self.assertEqual(events[1].arguments, {'reason': 'other'})

    def test_tokens_add_up_over_turns(self):
        client = FakeClient(
            FakeStream(['Grüezi.'], message(text('Grüezi.'), input_tokens=1200, output_tokens=30)),
            FakeStream(['Gut.'], message(text('Gut.'), input_tokens=40, output_tokens=10,
                                         cache_read=1200)))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client)

        async def scenario():
            await collect(brain.reply(None))
            await collect(brain.reply('Ja?'))

        run(scenario())
        self.assertEqual(brain.tokens, {'input_tokens': 1240, 'output_tokens': 40,
                                        'cache_read_input_tokens': 1200,
                                        'cache_creation_input_tokens': 0})


class ConversationTests(unittest.TestCase):
    def test_each_turn_sends_the_whole_conversation_with_claudes_turns_as_they_came(self):
        first = message(thinking(), text('Grüezi, hier ist der KI-Assistent von Florian Berth.'))
        client = FakeClient(FakeStream(['Grüezi, hier ist der KI-Assistent von Florian Berth.'],
                                       first),
                            FakeStream(['Gerne.']))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client)

        async def scenario():
            await collect(brain.reply(None))
            await collect(brain.reply('Ja, um was geht es?'))

        run(scenario())
        sent = client.requests[1]['messages']
        self.assertEqual([turn['role'] for turn in sent], ['user', 'assistant', 'user'])
        # Thinking blocks go back unchanged.
        self.assertIs(sent[1]['content'], first.content)
        self.assertEqual(sent[2]['content'], [{'type': 'text', 'text': 'Ja, um was geht es?'}])

    def test_a_turn_after_end_call_answers_the_tool_first(self):
        final = message(text('Adieu.'), end_call('completed', tool_id='toolu_9'))
        client = FakeClient(FakeStream(['Adieu.'], final), FakeStream(['Ja?']))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client)

        async def scenario():
            await collect(brain.reply('Tschüss.'))
            await collect(brain.reply('Moment, noch etwas!'))

        run(scenario())
        content = client.requests[1]['messages'][-1]['content']
        self.assertEqual(content[0], {'type': 'tool_result', 'tool_use_id': 'toolu_9',
                                      'content': 'ok'})
        self.assertEqual(content[1], {'type': 'text', 'text': 'Moment, noch etwas!'})

    def test_an_interruption_is_told_in_a_system_message_without_rewriting_the_last_turn(self):
        first = message(text('Grüezi, ich rufe im Auftrag von Florian Berth an, es geht um '
                             'eine Reservation.'))
        client = FakeClient(FakeStream(['Grüezi, ich rufe im Auftrag von Florian Berth an, '
                                        'es geht um eine Reservation.'], first),
                            FakeStream(['Entschuldigung.']))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client)

        async def scenario():
            await collect(brain.reply(None))
            brain.interrupted('Grüezi, ich rufe im Auf')
            await collect(brain.reply('Wer sind Sie?'))

        run(scenario())
        sent = client.requests[1]['messages']
        self.assertEqual([turn['role'] for turn in sent], ['user', 'assistant', 'user', 'system'])
        self.assertIs(sent[1]['content'], first.content)
        self.assertIn('"Grüezi, ich rufe im Auf"', sent[3]['content'])
        self.assertIn('unterbrochen', sent[3]['content'])

    def test_a_reply_cut_off_while_streaming_keeps_only_what_was_written(self):
        stream = FakeStream(['Grüezi Herr Hunziker. ', 'Ich rufe im Auftrag ', 'von Florian an.'],
                            message(thinking(), text('nie gesendet')))
        client = FakeClient(stream, FakeStream(['Gerne.']))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client)

        async def scenario():
            replies = brain.reply(None)
            first = await anext(replies)
            await replies.aclose()
            brain.interrupted('Grüezi Herr')
            await collect(brain.reply('Wer ist da?'))
            return first

        self.assertEqual(run(scenario()).text, 'Grüezi Herr Hunziker.')
        self.assertTrue(stream.closed)
        sent = client.requests[1]['messages']
        self.assertEqual(sent[1], {'role': 'assistant', 'content': 'Grüezi Herr Hunziker. '})
        self.assertEqual(sent[3]['role'], 'system')

    def test_a_note_without_claudes_turn_after_it_moves_behind_the_next_user_turn(self):
        # The turn after the note failed before Claude wrote a word, so no assistant turn follows
        # the system message. The API would refuse that, so the note moves behind the next turn.
        client = FakeClient(FakeStream(['Grüezi.']), FakeStream(error=connection_error()),
                            FakeStream(['Gut.']))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client)

        async def scenario():
            await collect(brain.reply(None))
            brain.interrupted('Grü')
            await collect(brain.reply('Hallo?'))
            await collect(brain.reply('Hallo, wer ist da?'))

        run(scenario())
        sent = client.requests[-1]['messages']
        self.assertEqual([turn['role'] for turn in sent],
                         ['user', 'assistant', 'user', 'user', 'system'])
        self.assertIn('"Grü"', sent[-1]['content'])


    def test_a_turn_the_call_asks_for_has_a_quiet_user_turn_and_the_note(self):
        client = FakeClient(FakeStream(['Grüezi.']), FakeStream(['Sind Sie noch da?']))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client)

        async def scenario():
            await collect(brain.reply(None))
            await collect(brain.reply(None, note='Frag, ob die Person noch da ist.'))

        run(scenario())
        sent = client.requests[1]['messages']
        self.assertEqual([turn['role'] for turn in sent], ['user', 'assistant', 'user', 'system'])
        self.assertEqual(sent[2]['content'], [{'type': 'text', 'text': QUIET_TURN}])
        self.assertEqual(sent[3]['content'], 'Frag, ob die Person noch da ist.')

    def test_the_opening_can_carry_a_note(self):
        client = FakeClient(FakeStream(['Hoi Florian.']))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client)
        run(collect(brain.reply(None, note='Du rufst Florian Berth selbst an.')))
        sent = client.requests[0]['messages']
        self.assertEqual(sent[0]['content'], [{'type': 'text', 'text': OPENING_TURN}])
        self.assertEqual(sent[1], {'role': 'system', 'content': 'Du rufst Florian Berth selbst an.'})

    def test_quiet_notes_and_an_interruption_come_with_the_next_turn(self):
        client = FakeClient(FakeStream(['Grüezi.']), FakeStream(['Gut.']))
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=client)

        async def scenario():
            await collect(brain.reply(None))
            brain.note('Vielleicht hat eine Mailbox abgenommen.')
            brain.interrupted('Grü')
            await collect(brain.reply('Hallo?', note='Die Zeit ist fast um.'))

        run(scenario())
        sent = client.requests[1]['messages']
        self.assertEqual([turn['role'] for turn in sent], ['user', 'assistant', 'user', 'system'])
        note = sent[3]['content']
        self.assertIn('"Grü"', note)
        self.assertIn('Vielleicht hat eine Mailbox abgenommen.', note)
        self.assertTrue(note.endswith('Die Zeit ist fast um.'))
        # Told once: the turn after has no note.
        self.assertEqual(brain.messages[-1]['role'], 'assistant')

class TroubleTests(unittest.TestCase):
    def test_when_the_api_fails_the_voice_apologises_discloses_and_hangs_up(self):
        for error, code in ((connection_error(), 'claude_unreachable'),
                            (status_error(anthropic.RateLimitError, 429), 'claude_rate_limited'),
                            (status_error(anthropic.AuthenticationError, 401), 'claude_auth'),
                            (status_error(anthropic.InternalServerError, 529), 'claude_http_529')):
            with self.subTest(code=code):
                brain = ClaudeBrain(brief(), SWISS_GERMAN,
                                    client=FakeClient(FakeStream(error=error)))
                events = run(collect(brain.reply(None)))
                self.assertEqual([event.kind for event in events], ['say', 'tool', 'done'])
                self.assertTrue(discloses(events[0].text, 'Florian Berth'))
                self.assertIn('Störung', events[0].text)
                self.assertEqual(events[1].arguments, {'reason': 'other'})
                self.assertEqual(brain.failure, code)
                self.assertEqual(failure_code(error), code)

    def test_a_refusal_ends_the_call_politely(self):
        final = message(text('Das'), stop_reason='refusal')
        brain = ClaudeBrain(brief(), SWISS_GERMAN, client=FakeClient(FakeStream(['Das'], final)))
        events = run(collect(brain.reply('Sagen Sie mir etwas Verbotenes.')))
        self.assertEqual([event.kind for event in events], ['say', 'tool', 'done'])
        self.assertTrue(discloses(events[0].text, 'Florian Berth'))
        self.assertEqual(events[1].arguments, {'reason': 'other'})
        self.assertEqual(brain.failure, 'claude_refused')

    def test_the_fallback_lines_disclose_the_ai_and_say_goodbye_in_the_locale(self):
        for locale in (SWISS_GERMAN, STANDARD_GERMAN):
            for line in (trouble_line(brief(), locale), refusal_line(brief(), locale)):
                with self.subTest(locale=locale.code, line=line):
                    self.assertTrue(discloses(line, 'Florian Berth'))
                    self.assertIn(locale.goodbye, line)


def sse(*events):
    """A streamed Messages API response, as the server sends it."""
    return ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events)


def streamed_reply(sentences, *, reason=None, tool_id='toolu_1'):
    events = [
        {'type': 'message_start', 'message': {
            'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': MODEL, 'content': [],
            'stop_reason': None, 'stop_sequence': None,
            'usage': {'input_tokens': 900, 'output_tokens': 1}}},
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        *({'type': 'content_block_delta', 'index': 0,
           'delta': {'type': 'text_delta', 'text': sentence}} for sentence in sentences),
        {'type': 'content_block_stop', 'index': 0},
    ]
    if reason:
        events += [
            {'type': 'content_block_start', 'index': 1, 'content_block': {
                'type': 'tool_use', 'id': tool_id, 'name': 'end_call', 'input': {}}},
            {'type': 'content_block_delta', 'index': 1, 'delta': {
                'type': 'input_json_delta', 'partial_json': json.dumps({'reason': reason})}},
            {'type': 'content_block_stop', 'index': 1},
        ]
    events += [
        {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use' if reason else 'end_turn',
                                            'stop_sequence': None},
         'usage': {'output_tokens': 25}},
        {'type': 'message_stop'},
    ]
    return sse(*events)


class AnthropicSdkTests(unittest.TestCase):
    """The real SDK over a fake HTTP connection: the requests it sends and the stream it reads."""

    def brain(self, *replies):
        self.bodies = []
        replies = list(replies)

        def handler(request):
            self.bodies.append(json.loads(request.content))
            return httpx2.Response(200, text=replies.pop(0),
                                   headers={'content-type': 'text/event-stream'})

        client = anthropic.AsyncAnthropic(
            api_key='test-key', max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
        return ClaudeBrain(brief(), SWISS_GERMAN, client=client)

    def test_a_call_runs_through_the_sdk_from_opening_to_end_call(self):
        brain = self.brain(
            streamed_reply(['Grüezi, hier ist der KI-Assistent ', 'von Florian Berth. Es geht ',
                            'um eine Reservation.']),
            streamed_reply(['Merci, auf Wiederhören.'], reason='completed', tool_id='toolu_7'),
            streamed_reply(['Ja, bitte?']))

        async def scenario():
            opening = await collect(brain.reply(None))
            brain.interrupted('Grüezi, hier ist der KI-Assistent von Florian Berth.')
            goodbye = await collect(brain.reply('Nein danke, kein Interesse.'))
            after = await collect(brain.reply('Halt, doch noch etwas.'))
            return opening, goodbye, after

        opening, goodbye, after = run(scenario())
        self.assertEqual(said(opening), ['Grüezi, hier ist der KI-Assistent von Florian Berth.',
                                         'Es geht um eine Reservation.'])
        self.assertEqual([(event.kind, event.name, event.arguments) for event in goodbye[1:]],
                         [('tool', 'end_call', {'reason': 'completed'}), ('done', '', {})])
        self.assertEqual(said(after), ['Ja, bitte?'])

        first = self.bodies[0]
        self.assertEqual(first['model'], 'claude-haiku-5-5')
        self.assertEqual(first['thinking'], {'type': 'adaptive'})
        self.assertEqual(first['output_config'], {'effort': 'low'})
        self.assertEqual(first['cache_control'], {'type': 'ephemeral'})
        self.assertTrue(first['stream'])
        self.assertEqual(first['tools'][0]['name'], 'end_call')
        self.assertEqual([turn['role'] for turn in self.bodies[1]['messages']],
                         ['user', 'assistant', 'user', 'system'])
        third = self.bodies[2]['messages']
        self.assertEqual(third[4]['content'][1],
                         {'type': 'tool_use', 'id': 'toolu_7', 'name': 'end_call',
                          'input': {'reason': 'completed'}})
        self.assertEqual(third[5]['content'][0],
                         {'type': 'tool_result', 'tool_use_id': 'toolu_7', 'content': 'ok'})
        self.assertEqual(brain.tokens['input_tokens'], 2700)


if __name__ == '__main__':
    unittest.main()
