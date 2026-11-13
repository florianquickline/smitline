"""Fakes of the voice chain's three parts, for tests: scripted, instant, and keyless."""
import asyncio

from voice_chain import BYTES_PER_MS, MULAW_FRAME_BYTES, BrainEvent, frames, texts


class FakeRecognizer:
    """Plays back a script of Heard events and keeps the audio it was sent.

    With KEEP_OPEN it then stays open like a real stream: `hear` adds what is heard next and
    `drop` ends the stream as if the connection were lost.
    """

    def __init__(self, script=(), *, keep_open=False):
        self.script = list(script)
        self.keep_open = keep_open
        self.language = None
        self.keywords = ()
        self.audio = b''
        self.closed = False
        self._next = asyncio.Queue() if keep_open else None

    async def start(self, *, language, keywords=()):
        self.language = language
        self.keywords = tuple(keywords)

    async def send_audio(self, chunk):
        self.audio += chunk

    def hear(self, event):
        self._next.put_nowait(event)

    def drop(self):
        self._next.put_nowait(None)

    async def events(self):
        for event in self.script:
            yield event
            await asyncio.sleep(0)
        while self._next is not None:
            event = await self._next.get()
            if event is None:
                return
            yield event

    async def close(self):
        self.closed = True
        if self._next is not None:
            self._next.put_nowait(None)


class FakeBrain:
    """Answers each turn with the next list of BrainEvents from its script, then `done`.

    With HOLD (an asyncio.Event), each turn waits for it before saying anything, as Claude takes
    a moment to start.
    """

    def __init__(self, replies, *, hold=None):
        self.replies = [list(reply) for reply in replies]
        self.hold = hold
        self.turns = []
        self.cues = []
        self.notes = []
        self.interruptions = []

    async def reply(self, heard, *, note=None):
        self.turns.append(heard)
        self.cues.append(note)
        events = self.replies.pop(0) if self.replies else []
        if self.hold is not None:
            await self.hold.wait()
        for event in events:
            yield event
            await asyncio.sleep(0)
        yield BrainEvent.done()

    def note(self, text):
        self.notes.append(text)

    def interrupted(self, spoken):
        self.interruptions.append(spoken)


class FakeSynthesizer:
    """Makes BYTES_PER_CHAR bytes of audio per character, in 20 ms frames, until canceled.

    The audio is silence unless FILL gives another byte (b'\\x2a' sounds like speech to the
    line). A reply given as sentences is spoken as they come, joined by spaces, as a provider
    would. `is_open` can be set False to act as a stream the provider closed.
    """

    SILENCE = b'\xff'

    def __init__(self, bytes_per_char=8, *, fill=SILENCE):
        self.bytes_per_char = bytes_per_char
        self.fill = fill
        self.spoken = []
        self.canceled = False
        self.started = False
        self.starts = 0
        self.is_open = False
        self.closed = False
        self._stop = None
        self._text = ''

    async def start(self):
        self.started = True
        self.starts += 1
        self.is_open = True

    async def speak(self, text, *, language):
        self._stop = stop = asyncio.Event()
        self._text = ''
        try:
            async for piece in texts(text):
                piece = f' {piece}' if self._text and not piece.startswith(' ') else piece
                self._text += piece
                for frame in frames(self.fill * (len(piece) * self.bytes_per_char),
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
        self.is_open = False
