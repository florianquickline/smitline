"""Fakes of the voice chain's three parts, for tests: scripted, instant, and keyless."""
import asyncio

from voice_chain import BYTES_PER_MS, MULAW_FRAME_BYTES, BrainEvent, frames, texts


class FakeRecognizer:
    """Plays back a script of Heard events and keeps the audio it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.language = None
        self.keywords = ()
        self.audio = b''
        self.closed = False

    async def start(self, *, language, keywords=()):
        self.language = language
        self.keywords = tuple(keywords)

    async def send_audio(self, chunk):
        self.audio += chunk

    async def events(self):
        for event in self.script:
            yield event
            await asyncio.sleep(0)

    async def close(self):
        self.closed = True


class FakeBrain:
    """Answers each turn with the next list of BrainEvents from its script, then `done`."""

    def __init__(self, replies):
        self.replies = [list(reply) for reply in replies]
        self.turns = []
        self.interruptions = []

    async def reply(self, heard):
        self.turns.append(heard)
        events = self.replies.pop(0) if self.replies else []
        for event in events:
            yield event
            await asyncio.sleep(0)
        yield BrainEvent.done()

    def interrupted(self, spoken):
        self.interruptions.append(spoken)


class FakeSynthesizer:
    """Makes BYTES_PER_CHAR bytes of silence per character, in 20 ms frames, until canceled.

    A reply given as sentences is spoken as they come, joined by spaces, as a provider would.
    """

    SILENCE = b'\xff'

    def __init__(self, bytes_per_char=8):
        self.bytes_per_char = bytes_per_char
        self.spoken = []
        self.canceled = False
        self.started = False
        self.closed = False
        self._stop = None
        self._text = ''

    async def start(self):
        self.started = True

    async def speak(self, text, *, language):
        self._stop = stop = asyncio.Event()
        self._text = ''
        try:
            async for piece in texts(text):
                piece = f' {piece}' if self._text and not piece.startswith(' ') else piece
                self._text += piece
                for frame in frames(self.SILENCE * (len(piece) * self.bytes_per_char),
                                    MULAW_FRAME_BYTES):
                    if stop.is_set():
                        return
                    yield frame
                    await asyncio.sleep(0)
        finally:
            self.spoken.append((self._text, language))

    async def cancel(self, *, heard_ms=None):
        self.canceled = True
        if self._stop is not None:
            self._stop.set()
        if heard_ms is None:
            return None
        return self._text[:heard_ms * BYTES_PER_MS // self.bytes_per_char]

    async def close(self):
        self.closed = True
