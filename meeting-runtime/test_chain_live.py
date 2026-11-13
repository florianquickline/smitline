"""The voice chain in GPT-Live's place: when it answers, when it stops, and what the call sees."""
import asyncio
import base64
import unittest

from call_locale import STANDARD_GERMAN, SWISS_GERMAN
from chain_live import ChainLive, Playback, call_keywords, function_call_event, usage_event
from call_brief import CallBrief
from voice_chain import BrainEvent, Heard
from voice_core import backend_usage_from, function_call_from
from voice_fakes import FakeBrain, FakeRecognizer, FakeSynthesizer


OPENING = [BrainEvent.say('Grüezi, ich rufe im Auftrag von Florian an.'),
           BrainEvent.say('Ich bin sein KI-Assistent.')]


class FakePlayback:
    """The line: PLAYING while the caller has not heard all of the agent's speech."""

    def __init__(self):
        self.playing = False
        self.unplayed = 0
        self.cleared = 0

    def pending(self):
        return self.playing

    def unplayed_ms(self):
        return self.unplayed

    async def clear(self):
        self.cleared += 1
        self.playing = False
        self.unplayed = 0


async def until(predicate, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError('condition not met in time')
        await asyncio.sleep(0.005)


async def settle():
    await asyncio.sleep(0.15)


class Call:
    """A ChainLive on fakes, with its events read as a phone call reads them."""

    def __init__(self, replies, *, locale=SWISS_GERMAN, brain=None, synthesizer=None):
        self.recognizer = FakeRecognizer(keep_open=True)
        self.brain = brain or FakeBrain(replies)
        self.synthesizer = synthesizer or FakeSynthesizer(bytes_per_char=8, fill=b'\x2a')
        self.playback = FakePlayback()
        self.live = ChainLive(self.recognizer, self.brain, self.synthesizer, locale=locale,
                              keywords=('Florian', 'Hunziker'), playback=self.playback)
        self.events = []

    async def __aenter__(self):
        await self.live.__aenter__()
        self.reader = asyncio.create_task(self._read())
        return self

    async def _read(self):
        async for event in self.live.events():
            self.events.append(event)

    async def __aexit__(self, *exc):
        await self.live.close()
        await asyncio.wait_for(self.reader, 2)
        await self.live.__aexit__(None, None, None)

    async def open(self, note=''):
        await self.live.append('session.commentary.append', note)

    def hear(self, event):
        self.recognizer.hear(event)

    def kinds(self):
        return [event['type'] for event in self.events]

    def said(self):
        return ''.join(event['delta'] for event in self.events
                       if event['type'] == 'session.output_transcript.delta').strip()

    def heard(self):
        return ''.join(event['delta'] for event in self.events
                       if event['type'] == 'session.input_transcript.delta').strip()

    def audio(self):
        return b''.join(base64.b64decode(event['delta']) for event in self.events
                        if event['type'] == 'session.output_audio.delta')

    def calls(self):
        return [call for call in map(function_call_from, self.events) if call]


def run(scenario):
    return asyncio.run(asyncio.wait_for(scenario(), 10))


class SessionTests(unittest.TestCase):
    def test_it_listens_and_speaks_in_the_language_of_the_call(self):
        async def scenario():
            for locale, language in ((SWISS_GERMAN, 'de-CH'), (STANDARD_GERMAN, 'de-DE')):
                async with Call([], locale=locale) as call:
                    self.assertTrue(call.live.started.is_set())
                    self.assertEqual(call.recognizer.language, language)
                    self.assertEqual(call.recognizer.keywords, ('Florian', 'Hunziker'))
                    self.assertTrue(call.synthesizer.started)
                self.assertTrue(call.recognizer.closed and call.synthesizer.closed)
        run(scenario)

    def test_the_callers_audio_reaches_the_recognizer(self):
        async def scenario():
            async with Call([]) as call:
                payload = base64.b64encode(b'\x01\x02' * 80).decode('ascii')
                self.assertTrue(await call.live.send_audio(payload))
                self.assertTrue(await call.live.send_audio(b'\x03' * 160))
                self.assertFalse(await call.live.send_audio(''))
                self.assertEqual(call.recognizer.audio, b'\x01\x02' * 80 + b'\x03' * 160)
        run(scenario)

    def test_close_ends_the_events_with_session_closed(self):
        async def scenario():
            call = Call([])
            async with call:
                pass
            self.assertEqual(call.events[-1], {'type': 'session.closed',
                                               'reason': 'close_requested'})
            self.assertTrue(call.live.closed.is_set())
        run(scenario)

    def test_it_fits_where_the_call_expects_gpt_live(self):
        self.assertIsInstance(FakePlayback(), Playback)
        self.assertEqual(function_call_from(function_call_event('end_call', {'reason': 'other'}))
                         ['arguments'], {'reason': 'other'})
        tokens = {'input_tokens': 20, 'cache_read_input_tokens': 1000,
                  'cache_creation_input_tokens': 5, 'output_tokens': 30}
        self.assertEqual(backend_usage_from(usage_event(tokens)),
                         {'input': 1025, 'cached': 1000, 'output': 30, 'webSearches': 0})

    def test_the_recognizer_expects_the_names_on_the_call(self):
        brief = CallBrief.from_dict({'channel': 'phone', 'to': '+41441234567',
                                     'onBehalfOf': 'Florian Berth', 'objective': 'Reservieren'})
        self.assertEqual(call_keywords(brief, {'name': 'Frau Hunziker'}),
                         ['Florian Berth', 'Frau Hunziker'])
        self.assertEqual(call_keywords(brief), ['Florian Berth'])


class TurnTests(unittest.TestCase):
    def test_nothing_is_said_before_the_call_opens(self):
        async def scenario():
            async with Call([OPENING]) as call:
                call.hear(Heard.final('Hunziker?', end_of_turn=True))
                await settle()
                self.assertEqual(call.brain.turns, [])
                self.assertEqual(call.heard(), 'Hunziker?')
                await call.open()
                await until(lambda: call.said())
                # What they said before the opening goes with it.
                self.assertEqual(call.brain.turns, ['Hunziker?'])
                self.assertEqual(call.brain.cues, [None])
                self.assertEqual(call.said(), 'Grüezi, ich rufe im Auftrag von Florian an. Ich '
                                              'bin sein KI-Assistent.')
        run(scenario)

    def test_an_opening_with_nobody_speaking_has_no_words(self):
        async def scenario():
            async with Call([OPENING]) as call:
                await call.open('Du rufst Florian selbst an.')
                await until(lambda: call.said())
                self.assertEqual(call.brain.turns, [None])
                self.assertEqual(call.brain.cues, ['Du rufst Florian selbst an.'])
                text = 'Grüezi, ich rufe im Auftrag von Florian an. Ich bin sein KI-Assistent.'
                self.assertEqual(len(call.audio()), len(text) * 8)
        run(scenario)

    def test_their_turn_is_answered_once_it_ends(self):
        async def scenario():
            async with Call([OPENING, [BrainEvent.say('Kein Problem.')]]) as call:
                await call.open()
                await until(lambda: call.said())
                call.hear(Heard.partial('Ich hab'))
                call.hear(Heard.final('Ich habe morgen'))
                await settle()
                self.assertEqual(len(call.brain.turns), 1)
                call.hear(Heard.final('Ich habe morgen keine Zeit.', end_of_turn=True))
                await until(lambda: len(call.brain.turns) == 2)
                self.assertEqual(call.brain.turns[1], 'Ich habe morgen keine Zeit.')
                self.assertEqual(call.heard(), 'Ich habe morgen keine Zeit.')
                await until(lambda: call.said().endswith('Kein Problem.'))
        run(scenario)

    def test_a_turn_the_recognizer_never_ended_ends_after_a_pause(self):
        async def scenario():
            async with Call([OPENING, [BrainEvent.say('Gut.')]]) as call:
                await call.open()
                await until(lambda: call.said())
                call.hear(Heard.partial('Ja das passt'))
                await until(lambda: len(call.brain.turns) == 2, timeout=4)
                self.assertEqual(call.brain.turns[1], 'Ja das passt')
                self.assertEqual(call.heard(), 'Ja das passt')
        run(scenario)

    def test_the_reply_is_written_down_once_it_was_heard(self):
        async def scenario():
            async with Call([OPENING]) as call:
                call.playback.playing = True
                await call.open()
                await until(lambda: call.audio())
                await settle()
                self.assertEqual(call.said(), '')
                call.playback.playing = False
                await until(lambda: call.said())
                await asyncio.wait_for(call.live.all_heard(), 1)
        run(scenario)

    def test_end_call_comes_after_the_goodbye_audio(self):
        async def scenario():
            goodbye = [BrainEvent.say('Merci, auf Wiederhören.'),
                       BrainEvent.tool('end_call', {'reason': 'completed'})]
            async with Call([OPENING, goodbye]) as call:
                await call.open()
                await until(lambda: call.said())
                call.hear(Heard.final('Nein, das wär alles.', end_of_turn=True))
                await until(lambda: call.calls())
                self.assertEqual([(c['name'], c['arguments']) for c in call.calls()],
                                 [('end_call', {'reason': 'completed'})])
                kinds = call.kinds()
                last_audio = max(i for i, kind in enumerate(kinds)
                                 if kind == 'session.output_audio.delta')
                self.assertGreater(kinds.index('response.event'), last_audio)
        run(scenario)

    def test_a_cue_asks_for_a_turn_with_its_note(self):
        async def scenario():
            async with Call([OPENING, [BrainEvent.say('Sind Sie noch da?')]]) as call:
                await call.open()
                await until(lambda: call.said())
                await call.live.append('session.commentary.append', 'Frag, ob jemand da ist.')
                await until(lambda: len(call.brain.turns) == 2)
                self.assertEqual(call.brain.turns[1], None)
                self.assertEqual(call.brain.cues[1], 'Frag, ob jemand da ist.')
        run(scenario)

    def test_a_cue_waits_for_them_to_finish(self):
        async def scenario():
            async with Call([OPENING, [BrainEvent.say('Gut, dann zum Schluss.')]]) as call:
                await call.open()
                await until(lambda: call.said())
                call.hear(Heard.partial('Also ich'))
                await call.live.append('session.commentary.append', 'Die Zeit ist fast um.')
                await settle()
                self.assertEqual(len(call.brain.turns), 1)
                call.hear(Heard.final('Also ich weiss nicht.', end_of_turn=True))
                await until(lambda: len(call.brain.turns) == 2)
                self.assertEqual((call.brain.turns[1], call.brain.cues[1]),
                                 ('Also ich weiss nicht.', 'Die Zeit ist fast um.'))
        run(scenario)

    def test_a_quiet_note_waits_for_the_next_turn(self):
        async def scenario():
            async with Call([OPENING]) as call:
                await call.open()
                await until(lambda: call.said())
                self.assertTrue(await call.live.append('session.thinking.append',
                                                       'Vielleicht eine Mailbox.'))
                await settle()
                self.assertEqual(call.brain.notes, ['Vielleicht eine Mailbox.'])
                self.assertEqual(len(call.brain.turns), 1)
        run(scenario)

    def test_the_next_reply_waits_until_the_last_one_was_heard(self):
        async def scenario():
            async with Call([OPENING, [BrainEvent.say('Gut.')]]) as call:
                call.playback.playing = True
                await call.open()
                await until(lambda: call.audio())
                call.hear(Heard.final('Mhm.', end_of_turn=True))
                await settle()
                self.assertEqual(len(call.brain.turns), 1)
                self.assertEqual(call.playback.cleared, 0)
                call.playback.playing = False
                await until(lambda: len(call.brain.turns) == 2)
                self.assertEqual(call.brain.turns[1], 'Mhm.')
        run(scenario)

    def test_a_closed_voice_opens_again_before_the_next_reply(self):
        async def scenario():
            async with Call([OPENING, [BrainEvent.say('Gut.')]]) as call:
                await call.open()
                await until(lambda: call.said())
                call.synthesizer.is_open = False
                call.hear(Heard.final('Ja.', end_of_turn=True))
                await until(lambda: call.said().endswith('Gut.'))
                self.assertEqual(call.synthesizer.starts, 2)
        run(scenario)


class InterruptionTests(unittest.TestCase):
    def test_talking_over_the_reply_stops_it_and_keeps_only_what_was_heard(self):
        async def scenario():
            async with Call([OPENING, [BrainEvent.say('Entschuldigung.')]]) as call:
                call.playback.playing = True
                await call.open()
                await until(lambda: call.live._reply and call.live._reply.delivered.is_set())
                whole = 'Grüezi, ich rufe im Auftrag von Florian an. Ich bin sein KI-Assistent.'
                # 8 bytes a character: 1 ms each. "Grüezi, ich rufe" has played.
                call.playback.unplayed = len(whole) - len('Grüezi, ich rufe')
                call.hear(Heard.partial('Moment mal'))
                await until(lambda: 'session.output_audio.interrupted' in call.kinds())
                self.assertEqual(call.playback.cleared, 1)
                self.assertTrue(call.synthesizer.canceled)
                self.assertEqual(call.brain.interruptions, ['Grüezi, ich rufe'])
                self.assertEqual(call.said(), 'Grüezi, ich rufe')
                call.hear(Heard.final('Moment mal, wer ist da?', end_of_turn=True))
                await until(lambda: len(call.brain.turns) == 2)
                self.assertEqual(call.brain.turns[1], 'Moment mal, wer ist da?')
        run(scenario)

    def test_mhm_and_their_own_voice_coming_back_do_not_stop_it(self):
        async def scenario():
            async with Call([OPENING]) as call:
                call.playback.playing = True
                await call.open()
                await until(lambda: call.audio())
                call.hear(Heard.partial('mhm'))
                call.hear(Heard.partial('ja genau'))
                call.hear(Heard.partial('ich rufe im Auftrag'))  # an echo of its own words
                await settle()
                self.assertEqual(call.playback.cleared, 0)
                self.assertNotIn('session.output_audio.interrupted', call.kinds())
        run(scenario)

    def test_their_words_while_claude_is_still_writing_cancel_the_reply(self):
        async def scenario():
            hold = asyncio.Event()
            brain = FakeBrain([OPENING, [BrainEvent.say('Wann denn?')],
                               [BrainEvent.say('Ah, schade.')]], hold=hold)
            async with Call([], brain=brain) as call:
                await call.open()
                hold.set()
                await until(lambda: call.said())
                hold.clear()
                call.hear(Heard.final('Ich habe', end_of_turn=True))
                await until(lambda: len(brain.turns) == 2)
                call.hear(Heard.partial('morgen keine'))
                await until(lambda: 'session.output_audio.interrupted' in call.kinds())
                # Claude had not written a word: nothing to tell it.
                self.assertEqual(brain.interruptions, [])
                hold.set()
                call.hear(Heard.final('morgen keine Zeit.', end_of_turn=True))
                await until(lambda: call.said().endswith('Ah, schade.'))
                self.assertNotIn('Wann denn?', call.said())
                self.assertEqual(brain.turns[2], 'morgen keine Zeit.')
                self.assertEqual(call.heard(), 'Ich habe morgen keine Zeit.')
        run(scenario)


class TroubleTests(unittest.TestCase):
    def test_a_lost_recognizer_ends_the_session(self):
        async def scenario():
            async with Call([]) as call:
                call.recognizer.drop()
                await until(lambda: 'session.closed' in call.kinds())
                self.assertEqual(call.events[-2], {'type': 'error', 'error': {
                    'code': 'speech_recognition_lost'}})
                self.assertEqual(call.events[-1], {'type': 'session.closed',
                                                   'reason': 'connection_lost'})
        run(scenario)

    def test_a_voice_that_fails_ends_the_session(self):
        class Broken(Exception):
            code = 'deepgram_closed'

        class BrokenSynthesizer(FakeSynthesizer):
            async def speak(self, text, *, language):
                raise Broken('closed')
                yield b''

        async def scenario():
            async with Call([OPENING], synthesizer=BrokenSynthesizer()) as call:
                await call.open()
                await until(lambda: 'session.closed' in call.kinds())
                self.assertIn({'type': 'error', 'error': {'code': 'deepgram_closed'}},
                              call.events)
                self.assertEqual(call.events[-1]['reason'], 'connection_lost')
        run(scenario)

    def test_claude_trouble_and_tokens_are_reported(self):
        class CountingBrain(FakeBrain):
            model = 'claude-haiku-5-5'

            def __init__(self, replies):
                super().__init__(replies)
                self.tokens = {'input_tokens': 0, 'output_tokens': 0,
                               'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}
                self.failure = None

            async def reply(self, heard, *, note=None):
                async for event in super().reply(heard, note=note):
                    yield event
                self.tokens['input_tokens'] += 100
                self.tokens['output_tokens'] += 10
                if len(self.turns) == 2:
                    self.failure = 'claude_rate_limited'

        async def scenario():
            brain = CountingBrain([OPENING, [BrainEvent.say('Entschuldigung.')]])
            async with Call([], brain=brain) as call:
                self.assertEqual(call.live.model, 'claude-haiku-5-5')
                await call.open()
                await until(lambda: call.said())
                call.hear(Heard.final('Hallo?', end_of_turn=True))
                await until(lambda: call.said().endswith('Entschuldigung.'))
                usage = [u for u in map(backend_usage_from, call.events) if u]
                self.assertEqual(usage, [{'input': 100, 'cached': 0, 'output': 10,
                                          'webSearches': 0}] * 2)
                self.assertIn({'type': 'error', 'error': {'code': 'claude_rate_limited'}},
                              call.events)
        run(scenario)


if __name__ == '__main__':
    unittest.main()
