"""Whole phone calls on Claude (SMITLINE_PHONE_ENGINE=claude): Twilio on one side, the voice chain
with fake parts on the other, through the call service as a real call goes."""
import asyncio
import base64
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from call_hooks import DefaultCallHooks, phone_engine
from call_locale import STANDARD_GERMAN, SWISS_GERMAN
from call_service import CallError, CallService
from call_store import CallStore
from chain_live import ChainLive, call_keywords
from phone_line import ChainPhoneSession, PhoneLine, PhoneSession
from test_phone import FakeLive, FakeSummarizer, FakeTwilio, FakeTwilioSocket, until
from voice_chain import BrainEvent, Heard
from voice_fakes import FakeBrain, FakeRecognizer, FakeSynthesizer


PUBLIC = 'https://abc-123.trycloudflare.com'
SWISS_NUMBER = '+41446681800'
GERMAN_NUMBER = '+493012345678'
ENV = {'ANTHROPIC_API_KEY': 'sk-ant-test', 'DEEPGRAM_API_KEY': 'dg-test',
       'TWILIO_ACCOUNT_SID': 'AC1', 'TWILIO_AUTH_TOKEN': 'tw-secret',
       'TWILIO_FROM_NUMBER': '+41445550000', 'SMITLINE_PHONE_ENGINE': 'claude'}
OPENING = [BrainEvent.say('Grüezi Herr Hunziker, ich rufe im Auftrag von Robin an, es geht um '
                          'einen Tisch.'),
           BrainEvent.say('Ich bin der KI-Assistent von Robin.')]
GOODBYE = [BrainEvent.say('Merci vielmal, auf Wiederhören.'),
           BrainEvent.tool('end_call', {'reason': 'completed'})]


def brief(**overrides):
    payload = {'channel': 'phone', 'to': SWISS_NUMBER, 'onBehalfOf': 'Robin',
               'objective': 'Einen Tisch für vier Personen um 19 Uhr reservieren',
               'contact': {'name': 'Herr Hunziker'}}
    payload.update(overrides)
    return payload


class Brain(FakeBrain):
    """A scripted brain that counts tokens and names its model, as ClaudeBrain does."""

    model = 'claude-haiku-5-5'

    def __init__(self, replies):
        super().__init__(replies)
        self.tokens = {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0,
                       'cache_creation_input_tokens': 0}

    async def reply(self, heard, *, note=None):
        async for event in super().reply(heard, note=note):
            yield event
        self.tokens['input_tokens'] += 1000
        self.tokens['cache_read_input_tokens'] += 800
        self.tokens['output_tokens'] += 40


class Chains:
    """The chain_factory: a ChainLive on fakes for each call, with what it was given."""

    def __init__(self):
        self.replies = []
        self.made = []

    def __call__(self, keys, *, brief, locale, contact, playback):
        self.recognizer = FakeRecognizer(keep_open=True)
        self.brain = Brain(self.replies)
        self.synthesizer = FakeSynthesizer(bytes_per_char=8, fill=b'\x2a')
        self.live = ChainLive(self.recognizer, self.brain, self.synthesizer, locale=locale,
                              keywords=call_keywords(brief, contact), playback=playback)
        self.made.append({'keys': keys, 'brief': brief, 'locale': locale, 'contact': contact,
                          'playback': playback})
        return self.live


class Harness:
    def __init__(self, temp, env=None):
        self.env = dict(ENV if env is None else env)
        self.twilio = FakeTwilio()
        self.store = CallStore(Path(temp) / 'calls')
        # 11 AM in Zurich and Berlin: inside calling hours.
        self.hooks = DefaultCallHooks(
            environ=self.env, store=self.store,
            clock=lambda: datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc))
        self.chains = Chains()

        async def public_url():
            return PUBLIC
        self.line = PhoneLine(public_url=public_url, environ=lambda: self.env,
                              twilio_factory=lambda creds: self.twilio, live_factory=FakeLive,
                              chain_factory=self.chains, status_grace=0.2)
        self.service = CallService(self.store, hooks=self.hooks, lines={'phone': self.line},
                                   summarizer_factory=lambda owner: FakeSummarizer(), environ={},
                                   shutdown_grace=0.2)

    async def dial(self, **overrides):
        record = await self.service.create(brief(**overrides))
        await until(lambda: self.twilio.created and self.line.session(record['id']))
        return record, self.line.session(record['id'])

    async def connect(self, record, session):
        ws = FakeTwilioSocket()
        assert self.line.claim(record['id'], session.token) is session
        task = asyncio.create_task(session.run(ws, {'streamSid': 'MZ1', 'callSid': 'CA123'}))
        await until(lambda: self.chains.made and session.live is self.chains.live)
        return ws, task

    def heard(self, text):
        self.chains.recognizer.hear(Heard.final(text, end_of_turn=True))


def agent_audio(ws):
    return b''.join(base64.b64decode(sent['media']['payload']) for sent in ws.sent
                    if sent['event'] == 'media')


class ClaudeCallTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        FakeLive.instances = []
        self.temp = tempfile.TemporaryDirectory()
        self.h = Harness(self.temp.name)

    async def asyncTearDown(self):
        await self.h.service.shutdown()
        self.temp.cleanup()

    async def finish(self, record, task):
        await asyncio.wait_for(task, 5)
        self.h.line.on_status(record['id'], {'CallStatus': 'completed', 'CallDuration': '40'})
        return await self.h.service.wait(record['id'], timeout=5, owner='local')

    async def test_a_swiss_call_runs_on_deepgram_and_claude_from_hello_to_goodbye(self):
        self.h.chains.replies = [OPENING, GOODBYE]
        record, session = await self.h.dial()
        self.assertIsInstance(session, ChainPhoneSession)
        ws, task = await self.h.connect(record, session)
        made = self.h.chains.made[0]
        self.assertEqual(made['keys'], {'anthropic': 'sk-ant-test', 'deepgram': 'dg-test'})
        self.assertIs(made['locale'], SWISS_GERMAN)
        self.assertEqual(self.h.chains.recognizer.language, 'de-CH')
        self.assertEqual(self.h.chains.recognizer.keywords, ('Robin', 'Herr Hunziker'))
        self.assertEqual(FakeLive.instances, [])  # GPT-Live is not used

        ws.push({'event': 'media', 'media': {'track': 'inbound',
                                             'payload': base64.b64encode(b'\x01' * 160).decode()}})
        await until(lambda: self.h.chains.recognizer.audio == b'\x01' * 160)
        self.h.heard('Hunziker.')
        await until(lambda: self.h.chains.brain.turns)
        # The opening answers their hello; its rules are in the system prompt, so no note.
        self.assertEqual((self.h.chains.brain.turns, self.h.chains.brain.cues),
                         (['Hunziker.'], [None]))
        opening = ' '.join(event.text for event in OPENING)
        await until(lambda: len(agent_audio(ws)) >= len(opening) * 8)
        self.assertEqual(agent_audio(ws).count(0x2a), len(opening) * 8)  # the rest pads frames
        await until(lambda: session.disclosed.is_set())  # heard, so verified

        self.h.heard('Ja, das passt.')
        await until(lambda: ws.closed)
        done = await self.finish(record, task)
        self.assertEqual(done['status'], 'completed')
        self.assertEqual(done['endReason'], 'hangup')
        self.assertIs(done['result']['disclosureVerified'], True)
        self.assertEqual([(line['speaker'], line['text']) for line in done['result']['transcript']],
                         [('other', 'Hunziker.'), ('agent', opening),
                          ('other', 'Ja, das passt.'),
                          ('agent', 'Merci vielmal, auf Wiederhören.')])
        usage = done['usage']
        self.assertEqual(usage['backendModel'], 'claude-haiku-5-5')
        self.assertEqual(usage['backendTokens'], {'input': 3600, 'cached': 1600, 'output': 80})
        self.assertEqual(usage['voiceSeconds'], 0)
        events = self.h.service.events(record['id'])
        ended = [e['data'] for e in events if e['type'] == 'call.end_requested']
        self.assertEqual(ended, [{'reason': 'completed'}])
        self.assertIn(('CA123', {'status': 'completed'}), self.h.twilio.updates)

    async def test_a_german_number_runs_in_standard_german(self):
        record, session = await self.h.dial(to=GERMAN_NUMBER)
        ws, task = await self.h.connect(record, session)
        self.assertIs(self.h.chains.made[0]['locale'], STANDARD_GERMAN)
        self.assertEqual(self.h.chains.recognizer.language, 'de-DE')
        ws.push({'event': 'stop'})
        await self.finish(record, task)

    async def test_the_call_asks_claude_in_german(self):
        self.h.chains.replies = [OPENING, [BrainEvent.say('Gerne.')],
                                 [BrainEvent.say('Ich muss leider Schluss machen. Adieu.'),
                                  BrainEvent.tool('end_call', {'reason': 'completed'})]]
        record, session = await self.h.dial()
        ws, task = await self.h.connect(record, session)
        self.h.heard('Grüezi?')
        await until(lambda: session.disclosed.is_set())
        self.h.heard('Um was geht es genau?')
        await until(lambda: len(self.h.chains.brain.turns) == 2)
        await asyncio.wait_for(session._let_speech_finish(), 3)
        await self.h.service.end(record['id'], owner='local')
        await until(lambda: len(self.h.chains.brain.turns) == 3)
        self.assertEqual(self.h.chains.brain.cues[2], 'Du musst das Gespräch jetzt beenden. '
                                                      'Bedank dich und verabschiede dich.')
        await until(lambda: ws.closed)
        await self.finish(record, task)

    async def test_talking_over_the_agent_drops_what_they_have_not_heard(self):
        long = [BrainEvent.say('Grüezi, ich rufe im Auftrag von Robin an. ' * 8),
                BrainEvent.say('Ich bin der KI-Assistent von Robin.')]
        self.h.chains.replies = [long, [BrainEvent.say('Entschuldigung, bitte.')]]
        record, session = await self.h.dial()
        ws, task = await self.h.connect(record, session)
        self.h.heard('Ja?')
        await until(lambda: len(agent_audio(ws)) > 0)
        spoke_before = session._other_spoke_at
        self.h.chains.recognizer.hear(Heard.partial('Moment, wer'))
        await until(lambda: any(sent['event'] == 'clear' for sent in ws.sent))
        await until(lambda: session.stats['interruptionsFollowed'] == 1)
        # Their words count before their transcript comes: a goodbye's hang-up waits for them.
        self.assertGreater(session._other_spoke_at, spoke_before)
        brain = self.h.chains.brain
        self.assertEqual(len(brain.interruptions), 1)
        self.assertLess(len(brain.interruptions[0]), 40)
        # Nothing it said was heard to the end, so the disclosure is still owed.
        self.assertFalse(session.disclosed.is_set())
        self.h.heard('Moment, wer sind Sie?')
        await until(lambda: len(brain.turns) == 2)
        self.assertEqual(brain.turns[1], 'Moment, wer sind Sie?')
        ws.push({'event': 'stop'})
        done = await self.finish(record, task)
        self.assertEqual(done['usage']['audio']['interruptionsFollowed'], 1)
        self.assertIn('call.interrupted', [e['type'] for e in self.h.service.events(record['id'])])


class ClaudeReadyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        await self.h.service.shutdown()
        self.temp.cleanup()

    async def refused(self, env=None, **overrides):
        self.h = Harness(self.temp.name, env)
        with self.assertRaises(CallError) as refused:
            await self.h.service.create(brief(**overrides))
        self.assertEqual(self.h.twilio.created, [])
        return refused.exception

    async def test_calls_on_claude_need_both_keys_not_openai(self):
        for missing in ('ANTHROPIC_API_KEY', 'DEEPGRAM_API_KEY'):
            env = {key: value for key, value in ENV.items() if key != missing}
            error = await self.refused(env)
            self.assertEqual((error.status, error.code), (503, 'not_configured'))
            self.assertEqual(error.details['missing'], [missing])
            await self.h.service.shutdown()

    async def test_calls_on_claude_go_to_switzerland_and_germany_only(self):
        error = await self.refused(to='+33142685300')
        self.assertEqual(error.code, 'unsupported_country')

    async def test_calls_on_claude_need_relayed_audio(self):
        error = await self.refused(dict(ENV, SMITLINE_PHONE_AUDIO='sip',
                                        SMITLINE_SIP_TRUNK_URL='sips:trunk.example.com',
                                        SMITLINE_SIP_USERNAME='user', SMITLINE_SIP_PASSWORD='pw'))
        self.assertEqual(error.code, 'engine_needs_relay')

    async def test_calls_on_claude_cannot_press_keys_yet(self):
        error = await self.refused(phoneMenus=True)
        self.assertEqual(error.code, 'phone_menus_unavailable')

    async def test_gpt_live_stays_the_default(self):
        self.h = Harness(self.temp.name, dict(ENV, SMITLINE_PHONE_ENGINE='',
                                              OPENAI_API_KEY='sk-test'))
        self.assertEqual(phone_engine(self.h.env), 'gpt-live')
        self.assertEqual(phone_engine({'SMITLINE_PHONE_ENGINE': ' Claude '}), 'claude')
        record = await self.h.service.create(brief())
        await until(lambda: self.h.line.session(record['id']))
        session = self.h.line.session(record['id'])
        self.assertIs(type(session), PhoneSession)
        await self.h.service.end(record['id'], owner='local')


if __name__ == '__main__':
    unittest.main()
