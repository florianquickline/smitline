"""Deepgram as the call's ears and voice, against a fake Deepgram on this computer: no key, no
network. The fake speaks the same WebSocket protocol as Nova-3 listening and Flux speaking."""
import asyncio
import json
import unittest
from unittest import mock

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer

import deepgram_voice
from deepgram_voice import (
    DeepgramError, DeepgramRecognizer, DeepgramSynthesizer, heard_part, keyterms, listen_url,
    voice_for,
)
from voice_chain import MULAW_FRAME_BYTES, Heard, SpeechRecognizer, SpeechSynthesizer


KEY = 'dg-test-key'


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 10))


async def serve(path, handler):
    app = web.Application()
    app.router.add_get(path, handler)
    server = TestServer(app)
    await server.start_server()
    return server


def results(transcript, *, is_final=False, speech_final=False):
    return {'type': 'Results', 'is_final': is_final, 'speech_final': speech_final,
            'channel': {'alternatives': [{'transcript': transcript, 'confidence': 0.9}]}}


class FakeListen:
    """Nova-3 listening: plays SCRIPT once audio arrives, then ends or waits for CloseStream."""

    def __init__(self, script=(), *, status=None, end_after_script=True):
        self.script = list(script)
        self.status = status
        self.end_after_script = end_after_script
        self.query = []
        self.auth = None
        self.audio = b''
        self.control = []

    async def handler(self, request):
        if self.status:
            return web.Response(status=self.status)
        self.query = list(request.query.items())
        self.auth = request.headers.get('Authorization')
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({'type': 'Metadata', 'request_id': 'req-1'})
        async for message in ws:
            if message.type == WSMsgType.BINARY:
                first = not self.audio
                self.audio += message.data
                if first:
                    for event in self.script:
                        await ws.send_json(event)
                    if self.end_after_script:
                        await ws.close()
            elif message.type == WSMsgType.TEXT:
                kind = json.loads(message.data)['type']
                self.control.append(kind)
                if kind == 'CloseStream':
                    await ws.close()
        return ws


class FakeFlux:
    """Flux speaking: BYTES_PER_CHAR bytes of audio per character, in CHUNK-byte pieces, one
    every DELAY seconds; Interrupt splits the text at the played offset."""

    def __init__(self, *, bytes_per_char=40, chunk=500, delay=0.0, hello=None, status=None):
        self.bytes_per_char = bytes_per_char
        self.chunk = chunk
        self.delay = delay
        self.hello = hello or {'type': 'Connected', 'request_id': 'req-2',
                               'model_name': 'flux-leonie-de'}
        self.status = status
        self.query = []
        self.auth = None
        self.received = []
        self.closed = False

    async def handler(self, request):
        if self.status:
            return web.Response(status=self.status)
        self.query = list(request.query.items())
        self.auth = request.headers.get('Authorization')
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json(self.hello)
        if self.hello.get('type') != 'Connected':
            await ws.close()
            return ws
        turn = {'id': None, 'text': '', 'start': 0, 'task': None}
        session = {'ms': 0, 'turns': 0}

        async def generate(current):
            audio = b'\x7f' * (len(current['text']) * self.bytes_per_char)
            for start in range(0, len(audio), self.chunk):
                await ws.send_bytes(audio[start:start + self.chunk])
                await asyncio.sleep(self.delay)
            await ws.send_json({'type': 'Flushed', 'speech_id': current['id']})
            duration = len(audio) // 8
            session['ms'] += duration
            await ws.send_json({'type': 'SpeechMetadata', 'speech_id': current['id'],
                                'audio_duration_ms': duration,
                                'input_character_count': len(current['text'])})
            turn.update(id=None, text='', task=None)

        async for message in ws:
            if message.type != WSMsgType.TEXT:
                continue
            data = json.loads(message.data)
            self.received.append(data)
            kind = data['type']
            if kind == 'Speak':
                if turn['id'] is None:
                    session['turns'] += 1
                    turn.update(id=f'dg_sp_{session["turns"]}', start=session['ms'])
                    await ws.send_json({'type': 'SpeechStarted', 'speech_id': turn['id']})
                turn['text'] += data['text']
            elif kind == 'Flush':
                if turn['id'] is None:
                    await ws.send_json({'type': 'Warning', 'code': 'NO_ACTIVE_SPEECH',
                                        'description': 'Nothing to flush.'})
                else:
                    turn['task'] = asyncio.create_task(generate(dict(turn)))
            elif kind == 'Interrupt':
                if turn['task'] is not None:
                    turn['task'].cancel()
                offset = (data.get('playback_offset') or {}).get('value')
                answer = {'type': 'SpeechInterrupted',
                          'metadata': {'speech_id': turn['id']}}
                if offset is not None:
                    chars = (offset - turn['start']) * 8 // self.bytes_per_char
                    answer.update(audio_played_ms=offset, text_spoken=turn['text'][:chars],
                                  text_remaining=turn['text'][chars:])
                    session['ms'] = offset
                await ws.send_json(answer)
                turn.update(id=None, text='', task=None)
            elif kind == 'Close':
                await ws.send_json({'type': 'SessionMetadata', 'total_audio_duration_ms':
                                    session['ms']})
                self.closed = True
                await ws.close()
        return ws


async def sentences(*pieces):
    for piece in pieces:
        yield piece
        await asyncio.sleep(0)


class SetupTests(unittest.TestCase):
    def test_both_fit_their_interfaces(self):
        self.assertIsInstance(DeepgramRecognizer(KEY), SpeechRecognizer)
        self.assertIsInstance(DeepgramSynthesizer(KEY, voice='flux-leonie-de'), SpeechSynthesizer)

    def test_both_need_a_key(self):
        for blank in (None, '', '  '):
            with self.assertRaises(ValueError):
                DeepgramRecognizer(blank)
            with self.assertRaises(ValueError):
                DeepgramSynthesizer(blank, voice='flux-leonie-de')

    def test_swiss_calls_get_a_swiss_voice_and_german_calls_a_german_one(self):
        self.assertEqual(voice_for('de-CH'), 'flux-leonie-de')
        self.assertEqual(voice_for('de-DE'), 'flux-janine-de')
        with self.assertRaises(DeepgramError):
            voice_for('en-US')

    def test_keyterms_drop_blanks_and_repeats_and_are_capped(self):
        self.assertEqual(keyterms(['Hunziker', ' hunziker ', '', None, 'Bahnhof  Enge']),
                         ['Hunziker', 'Bahnhof Enge'])
        self.assertEqual(len(keyterms([f'Name{i}' for i in range(80)])),
                         deepgram_voice.MAX_KEYTERMS)

    def test_other_languages_are_refused_before_connecting(self):
        with self.assertRaises(DeepgramError) as caught:
            listen_url('fr-FR')
        self.assertEqual(caught.exception.code, 'unsupported_language')

    def test_the_heard_part_is_cut_at_a_word(self):
        text = 'Grüezi Herr Hunziker, ich rufe an.'
        self.assertEqual(heard_part(text, 0, 1000), '')
        self.assertEqual(heard_part(text, 500, 1000), 'Grüezi Herr')
        self.assertEqual(heard_part(text, 1000, 1000), text)
        self.assertEqual(heard_part(text, 5000, 1000), text)
        self.assertEqual(heard_part(text, 300, None), '')


class RecognizerTests(unittest.TestCase):
    def listen(self, fake, scenario):
        async def main():
            server = await serve('/v1/listen', fake.handler)
            try:
                recognizer = DeepgramRecognizer(KEY, url=str(server.make_url('/v1/listen')))
                return await scenario(recognizer)
            finally:
                await server.close()
        return run(main())

    def test_a_swiss_call_streams_telephone_audio_to_nova_3_in_de_ch(self):
        fake = FakeListen(end_after_script=False)

        async def scenario(recognizer):
            await recognizer.start(language='de-CH', keywords=['Hunziker', 'Bahnhof Enge'])
            await recognizer.send_audio(b'\xff' * 160)
            await recognizer.send_audio(b'\x7f' * 160)
            await recognizer.close()

        self.listen(fake, scenario)
        query = fake.query
        for pair in (('model', 'nova-3'), ('language', 'de-CH'), ('encoding', 'mulaw'),
                     ('sample_rate', '8000'), ('channels', '1'), ('interim_results', 'true'),
                     ('endpointing', '300'), ('utterance_end_ms', '1000'),
                     ('vad_events', 'true'), ('keyterm', 'Hunziker'),
                     ('keyterm', 'Bahnhof Enge')):
            self.assertIn(pair, query)
        self.assertEqual(fake.auth, f'Token {KEY}')
        self.assertEqual(fake.audio, b'\xff' * 160 + b'\x7f' * 160)

    def test_a_german_call_listens_in_de(self):
        fake = FakeListen()

        async def scenario(recognizer):
            await recognizer.start(language='de-DE')
            await recognizer.send_audio(b'\xff' * 160)
            [event async for event in recognizer.events()]
            await recognizer.close()

        self.listen(fake, scenario)
        self.assertIn(('language', 'de'), fake.query)

    def test_what_they_say_comes_as_it_is_heard_and_the_turn_ends_with_all_of_it(self):
        fake = FakeListen([
            {'type': 'SpeechStarted', 'channel': [0], 'timestamp': 0.4},
            results('I ha'),
            results('Ich habe morgen', is_final=True),
            results('keine'),
            results('keine Zeit.', is_final=True, speech_final=True),
            results('', is_final=True, speech_final=True),
            results('Und', is_final=True),
            {'type': 'UtteranceEnd', 'channel': [0], 'last_word_end': 3.1},
            {'type': 'UtteranceEnd', 'channel': [0], 'last_word_end': 3.1},
        ])

        async def scenario(recognizer):
            await recognizer.start(language='de-CH')
            await recognizer.send_audio(b'\xff' * 160)
            heard = [event async for event in recognizer.events()]
            await recognizer.close()
            return heard, recognizer.request_id

        heard, request_id = self.listen(fake, scenario)
        self.assertEqual(heard, [
            Heard.speech_started(),
            Heard.partial('I ha'),
            Heard.final('Ich habe morgen'),
            Heard.partial('Ich habe morgen keine'),
            Heard.final('Ich habe morgen keine Zeit.', end_of_turn=True),
            Heard.final('Und'),
            Heard.final('Und', end_of_turn=True),
        ])
        self.assertEqual(request_id, 'req-1')

    def test_close_asks_deepgram_to_finish_the_stream(self):
        fake = FakeListen(end_after_script=False)

        async def scenario(recognizer):
            await recognizer.start(language='de-CH')
            await recognizer.send_audio(b'\xff' * 160)
            await recognizer.close()
            await recognizer.send_audio(b'\xff' * 160)  # After close: dropped, no error.

        self.listen(fake, scenario)
        self.assertEqual(fake.control, ['CloseStream'])
        self.assertEqual(fake.audio, b'\xff' * 160)

    def test_a_quiet_line_is_kept_open(self):
        fake = FakeListen(end_after_script=False)

        async def scenario(recognizer):
            await recognizer.start(language='de-CH')
            await asyncio.sleep(0.2)
            await recognizer.close()

        with mock.patch.object(deepgram_voice, 'KEEPALIVE_SECONDS', 0.05):
            self.listen(fake, scenario)
        self.assertIn('KeepAlive', fake.control)

    def test_a_refused_key_is_a_deepgram_auth_error_without_the_key(self):
        fake = FakeListen(status=401)

        async def scenario(recognizer):
            with self.assertRaises(DeepgramError) as caught:
                await recognizer.start(language='de-CH')
            return caught.exception

        error = self.listen(fake, scenario)
        self.assertEqual(error.code, 'deepgram_auth')
        self.assertNotIn(KEY, str(error))

    def test_an_unreachable_deepgram_is_reported(self):
        async def main():
            server = await serve('/v1/listen', FakeListen().handler)
            url = str(server.make_url('/v1/listen'))
            await server.close()
            with self.assertRaises(DeepgramError) as caught:
                await DeepgramRecognizer(KEY, url=url).start(language='de-CH')
            return caught.exception.code

        self.assertEqual(run(main()), 'deepgram_unreachable')


class SynthesizerTests(unittest.TestCase):
    def speak(self, fake, scenario, voice='flux-leonie-de'):
        async def main():
            server = await serve('/v2/speak', fake.handler)
            try:
                synthesizer = DeepgramSynthesizer(KEY, voice=voice,
                                                  url=str(server.make_url('/v2/speak')))
                return await scenario(synthesizer)
            finally:
                await server.close()
        return run(main())

    def test_a_reply_comes_back_as_telephone_audio_in_20_ms_frames(self):
        fake = FakeFlux(bytes_per_char=40, chunk=333)
        text = 'Grüezi Herr Hunziker.'

        async def scenario(synthesizer):
            await synthesizer.start()
            audio = [frame async for frame in synthesizer.speak(text, language='de-CH')]
            await synthesizer.close()
            return audio

        audio = self.speak(fake, scenario)
        self.assertEqual(sum(len(frame) for frame in audio), len(text) * 40)
        self.assertTrue(all(len(frame) == MULAW_FRAME_BYTES for frame in audio[:-1]))
        self.assertIn(('model', 'flux-leonie-de'), fake.query)
        self.assertIn(('encoding', 'mulaw'), fake.query)
        self.assertIn(('sample_rate', '8000'), fake.query)
        self.assertEqual(fake.auth, f'Token {KEY}')
        self.assertEqual([m['type'] for m in fake.received], ['Speak', 'Flush', 'Close'])
        self.assertTrue(fake.closed)

    def test_sentences_of_one_reply_go_out_as_they_come_in_one_turn(self):
        fake = FakeFlux(bytes_per_char=10)

        async def scenario(synthesizer):
            await synthesizer.start()
            audio = [frame async for frame in synthesizer.speak(
                sentences('Grüezi.', 'Hier ist der KI-Assistent von Florian.'), language='de-CH')]
            await synthesizer.close()
            return audio

        audio = self.speak(fake, scenario)
        whole = 'Grüezi. Hier ist der KI-Assistent von Florian.'
        self.assertEqual(sum(len(frame) for frame in audio), len(whole) * 10)
        self.assertEqual(fake.received[:3], [
            {'type': 'Speak', 'text': 'Grüezi.'},
            {'type': 'Speak', 'text': ' Hier ist der KI-Assistent von Florian.'},
            {'type': 'Flush'}])

    def test_replies_follow_each_other_on_the_same_stream(self):
        fake = FakeFlux(bytes_per_char=8)

        async def scenario(synthesizer):
            await synthesizer.start()
            first = [f async for f in synthesizer.speak('Grüezi.', language='de-CH')]
            second = [f async for f in synthesizer.speak('Merci vielmal.', language='de-CH')]
            nothing = [f async for f in synthesizer.speak(sentences(' ', ''), language='de-CH')]
            await synthesizer.close()
            return first, second, nothing

        first, second, nothing = self.speak(fake, scenario)
        self.assertEqual(sum(map(len, first)), len('Grüezi.') * 8)
        self.assertEqual(sum(map(len, second)), len('Merci vielmal.') * 8)
        self.assertEqual(nothing, [])
        self.assertEqual([m['type'] for m in fake.received],
                         ['Speak', 'Flush', 'Speak', 'Flush', 'Close'])

    def test_cancel_while_flux_speaks_stops_the_audio_and_says_what_was_heard(self):
        # 40 bytes a character is 5 ms a character; the audio comes slower than it is read.
        fake = FakeFlux(bytes_per_char=40, chunk=400, delay=0.02)
        first_text = 'Grüezi.'
        text = 'Grüezi Herr Hunziker, ich rufe im Auftrag von Florian an.'

        async def scenario(synthesizer):
            await synthesizer.start()
            [f async for f in synthesizer.speak(first_text, language='de-CH')]
            received = []
            async for frame in synthesizer.speak(text, language='de-CH'):
                received.append(frame)
                if len(received) == 6:
                    heard = await synthesizer.cancel(heard_ms=60)
            after = [f async for f in synthesizer.speak('Wie bitte?', language='de-CH')]
            await synthesizer.close()
            return received, heard, after

        received, heard, after = self.speak(fake, scenario)
        self.assertEqual(len(received), 6)
        self.assertEqual(heard, text[:60 * 8 // 40].strip())  # Flux's split, tidied.
        interrupt = next(m for m in fake.received if m['type'] == 'Interrupt')
        # The offset counts from the start of the call's audio: the first reply came before.
        self.assertEqual(interrupt['playback_offset'],
                         {'type': 'time_ms', 'value': len(first_text) * 40 // 8 + 60})
        # The canceled reply's audio does not leak into the next one.
        self.assertEqual(sum(map(len, after)), len('Wie bitte?') * 40)

    def test_cancel_after_flux_finished_estimates_what_was_heard_without_asking(self):
        fake = FakeFlux(bytes_per_char=8)
        text = 'Grüezi Herr Hunziker, ich rufe an.'

        async def scenario(synthesizer):
            await synthesizer.start()
            audio = [f async for f in synthesizer.speak(text, language='de-CH')]
            heard = await synthesizer.cancel(heard_ms=len(text) // 2)
            again = await synthesizer.cancel(heard_ms=10)
            await synthesizer.close()
            return audio, heard, again

        audio, heard, again = self.speak(fake, scenario)
        self.assertEqual(sum(map(len, audio)), len(text) * 8)
        self.assertEqual(heard, heard_part(text, len(text) // 2, len(text)))
        self.assertIsNone(again)
        self.assertNotIn('Interrupt', [m['type'] for m in fake.received])

    def test_cancel_without_a_reply_or_a_played_time_says_nothing(self):
        fake = FakeFlux(bytes_per_char=40, chunk=400, delay=0.02)

        async def scenario(synthesizer):
            await synthesizer.start()
            nothing = await synthesizer.cancel(heard_ms=100)
            received = []
            async for frame in synthesizer.speak('Grüezi Herr Hunziker.', language='de-CH'):
                received.append(frame)
                heard = await synthesizer.cancel()
            await synthesizer.close()
            return nothing, received, heard

        nothing, received, heard = self.speak(fake, scenario)
        self.assertIsNone(nothing)
        self.assertEqual(len(received), 1)
        self.assertIsNone(heard)
        interrupt = next(m for m in fake.received if m['type'] == 'Interrupt')
        self.assertNotIn('playback_offset', interrupt)

    def test_a_refused_key_or_a_failed_hello_is_a_deepgram_error(self):
        for fake, code in ((FakeFlux(status=401), 'deepgram_auth'),
                           (FakeFlux(status=500), 'deepgram_http_500'),
                           (FakeFlux(hello={'type': 'Error', 'code': 'INVALID_MODEL',
                                            'description': 'Unknown voice.'}), 'INVALID_MODEL')):
            with self.subTest(code=code):
                async def scenario(synthesizer):
                    with self.assertRaises(DeepgramError) as caught:
                        await synthesizer.start()
                    return caught.exception

                error = self.speak(fake, scenario)
                self.assertEqual(error.code, code)
                self.assertNotIn(KEY, str(error))


if __name__ == '__main__':
    unittest.main()
