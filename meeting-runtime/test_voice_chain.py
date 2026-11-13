"""The three parts that replace GPT-Live: recognizer, brain, synthesizer, and their fakes."""
import asyncio
import unittest

from voice_chain import (
    BrainEvent, Heard, SpeechRecognizer, SpeechSynthesizer, VoiceBrain, MULAW_FRAME_BYTES,
    frames, heard_part,
)
from voice_fakes import FakeBrain, FakeRecognizer, FakeSynthesizer


def run(coro):
    return asyncio.run(coro)


async def collect(iterator):
    return [item async for item in iterator]


class InterfaceTests(unittest.TestCase):
    def test_the_heard_part_is_cut_at_a_word(self):
        self.assertEqual(heard_part('Grüezi Herr Hunziker', 500, 1000), 'Grüezi')
        self.assertEqual(heard_part('Grüezi', 1000, 1000), 'Grüezi')
        self.assertEqual(heard_part('Grüezi', 0, None), '')

    def test_the_fakes_fit_the_interfaces(self):
        self.assertIsInstance(FakeRecognizer([]), SpeechRecognizer)
        self.assertIsInstance(FakeBrain([]), VoiceBrain)
        self.assertIsInstance(FakeSynthesizer(), SpeechSynthesizer)

    def test_heard_events_say_what_they_are(self):
        self.assertTrue(Heard.speech_started().started)
        partial = Heard.partial('I ha morn')
        self.assertFalse(partial.is_final)
        self.assertFalse(partial.end_of_turn)
        final = Heard.final('Ich habe morgen keine Zeit.', end_of_turn=True)
        self.assertTrue(final.is_final)
        self.assertTrue(final.end_of_turn)

    def test_brain_events_say_what_they_are(self):
        self.assertEqual(BrainEvent.say('Grüezi.').text, 'Grüezi.')
        tool = BrainEvent.tool('end_call', {'reason': 'completed'})
        self.assertEqual((tool.kind, tool.name, tool.arguments),
                         ('tool', 'end_call', {'reason': 'completed'}))
        self.assertEqual(BrainEvent.done().kind, 'done')

    def test_frames_cut_telephone_audio_into_20_ms_pieces(self):
        self.assertEqual(MULAW_FRAME_BYTES, 160)
        audio = bytes(range(256)) * 2
        pieces = list(frames(audio))
        self.assertTrue(all(len(piece) == 160 for piece in pieces[:-1]))
        self.assertEqual(b''.join(pieces), audio)


class FakeRecognizerTests(unittest.TestCase):
    def test_it_plays_its_script_and_keeps_the_audio(self):
        script = [Heard.speech_started(), Heard.partial('I ha'),
                  Heard.final('Ich habe morgen keine Zeit.', end_of_turn=True)]
        recognizer = FakeRecognizer(script)

        async def scenario():
            await recognizer.start(language='de-CH', keywords=('Hunziker',))
            await recognizer.send_audio(b'\xff' * 160)
            events = await collect(recognizer.events())
            await recognizer.close()
            return events

        self.assertEqual(run(scenario()), script)
        self.assertEqual(recognizer.language, 'de-CH')
        self.assertEqual(recognizer.keywords, ('Hunziker',))
        self.assertEqual(recognizer.audio, b'\xff' * 160)
        self.assertTrue(recognizer.closed)

    def test_kept_open_it_hears_what_comes_next_until_dropped(self):
        recognizer = FakeRecognizer([Heard.partial('Grüe')], keep_open=True)

        async def scenario():
            events = []

            async def listen():
                async for event in recognizer.events():
                    events.append(event)
            listening = asyncio.create_task(listen())
            await asyncio.sleep(0.01)
            recognizer.hear(Heard.final('Grüezi.', end_of_turn=True))
            await asyncio.sleep(0.01)
            still_open = not listening.done()
            recognizer.drop()
            await asyncio.wait_for(listening, 1)
            return events, still_open

        events, still_open = run(scenario())
        self.assertTrue(still_open)
        self.assertEqual([event.text for event in events], ['Grüe', 'Grüezi.'])


class FakeBrainTests(unittest.TestCase):
    def test_it_answers_each_turn_from_its_script(self):
        brain = FakeBrain([
            [BrainEvent.say('Grüezi, ich rufe im Auftrag von Florian an.'),
             BrainEvent.say('Ich bin sein KI-Assistent.')],
            [BrainEvent.say('Merci, auf Wiederhören.'),
             BrainEvent.tool('end_call', {'reason': 'completed'})],
        ])

        async def scenario():
            first = await collect(brain.reply(None))
            second = await collect(brain.reply('Ja, das passt.'))
            return first, second

        first, second = run(scenario())
        self.assertEqual([event.text for event in first[:-1]],
                         ['Grüezi, ich rufe im Auftrag von Florian an.', 'Ich bin sein KI-Assistent.'])
        self.assertEqual(first[-1].kind, 'done')
        self.assertEqual(second[1].name, 'end_call')
        self.assertEqual(brain.turns, [None, 'Ja, das passt.'])

    def test_it_keeps_the_notes_of_each_turn_and_the_quiet_ones(self):
        brain = FakeBrain([[BrainEvent.say('Sind Sie noch da?')]])
        brain.note('Vielleicht hat eine Mailbox abgenommen.')
        events = run(collect(brain.reply(None, note='Frag, ob die Person noch da ist.')))
        self.assertEqual(events[0].text, 'Sind Sie noch da?')
        self.assertEqual(brain.turns, [None])
        self.assertEqual(brain.cues, ['Frag, ob die Person noch da ist.'])
        self.assertEqual(brain.notes, ['Vielleicht hat eine Mailbox abgenommen.'])

    def test_a_held_turn_says_nothing_until_released(self):
        async def scenario():
            hold = asyncio.Event()
            brain = FakeBrain([[BrainEvent.say('Grüezi.')]], hold=hold)
            turn = asyncio.ensure_future(collect(brain.reply(None)))
            await asyncio.sleep(0.01)
            waiting = not turn.done()
            hold.set()
            return waiting, await turn

        waiting, events = run(scenario())
        self.assertTrue(waiting)
        self.assertEqual(events[0].text, 'Grüezi.')

    def test_it_remembers_what_was_heard_before_an_interruption(self):
        brain = FakeBrain([])
        brain.interrupted('Grüezi, ich rufe im Auf')
        self.assertEqual(brain.interruptions, ['Grüezi, ich rufe im Auf'])

    def test_a_turn_past_the_script_only_ends(self):
        brain = FakeBrain([])
        events = run(collect(brain.reply('Hallo?')))
        self.assertEqual([event.kind for event in events], ['done'])


class FakeSynthesizerTests(unittest.TestCase):
    def test_it_returns_telephone_audio_frames_for_each_text(self):
        synthesizer = FakeSynthesizer(bytes_per_char=40)

        async def scenario():
            return await collect(synthesizer.speak('Grüezi', language='de-CH'))

        chunks = run(scenario())
        self.assertEqual(sum(len(chunk) for chunk in chunks), 6 * 40)
        self.assertTrue(all(len(chunk) <= MULAW_FRAME_BYTES for chunk in chunks))
        self.assertEqual(synthesizer.spoken, [('Grüezi', 'de-CH')])

    def test_cancel_stops_the_audio(self):
        synthesizer = FakeSynthesizer(bytes_per_char=400)

        async def scenario():
            received = []
            async for chunk in synthesizer.speak('Ein langer Satz', language='de-DE'):
                received.append(chunk)
                if len(received) == 2:
                    await synthesizer.cancel()
            return received

        self.assertEqual(len(run(scenario())), 2)
        self.assertTrue(synthesizer.canceled)

    def test_a_reply_can_come_as_sentences_and_cancel_says_what_was_heard(self):
        synthesizer = FakeSynthesizer(bytes_per_char=8)

        async def written():
            for sentence in ('Grüezi.', 'Ich bin der KI-Assistent.'):
                yield sentence

        async def scenario():
            await synthesizer.start()
            received = []
            async for chunk in synthesizer.speak(written(), language='de-CH'):
                received.append(chunk)
            heard = await synthesizer.cancel(heard_ms=len('Grüezi. Ich') * 8 // 8)
            await synthesizer.close()
            return received, heard

        received, heard = run(scenario())
        whole = 'Grüezi. Ich bin der KI-Assistent.'
        self.assertEqual(sum(len(chunk) for chunk in received), len(whole) * 8)
        self.assertEqual(synthesizer.spoken, [(whole, 'de-CH')])
        self.assertEqual(heard, 'Grüezi. Ich')
        self.assertTrue(synthesizer.started and synthesizer.closed)
        self.assertFalse(synthesizer.is_open)
        self.assertIsNone(run(FakeSynthesizer().cancel()))

    def test_it_can_sound_like_speech(self):
        synthesizer = FakeSynthesizer(bytes_per_char=20, fill=b'\x2a')
        chunks = run(collect(synthesizer.speak('Ja', language='de-CH')))
        self.assertEqual(b''.join(chunks), b'\x2a' * 40)


if __name__ == '__main__':
    unittest.main()
