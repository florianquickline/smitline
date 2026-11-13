"""The three parts that hear and speak on a phone call, in place of one GPT-Live session.

    caller audio -> SpeechRecognizer -> text -> VoiceBrain -> text -> SpeechSynthesizer -> audio

Each part is an interface, so a provider can be swapped without touching the call: Deepgram for
recognition and speech, Claude for the brain, and the fakes in voice_fakes.py for tests.

Audio on the phone line is G.711 mu-law at 8 kHz, as Twilio's media stream carries it: one byte
per sample, 160 bytes per 20 ms frame.
"""
from dataclasses import dataclass, field
from typing import (
    AsyncIterable, AsyncIterator, Optional, Protocol, Sequence, Union, runtime_checkable,
)


SAMPLE_RATE = 8000
ENCODING = 'mulaw'
MULAW_FRAME_BYTES = SAMPLE_RATE * 20 // 1000
BYTES_PER_MS = SAMPLE_RATE // 1000


def frames(audio, size=MULAW_FRAME_BYTES):
    """Cut audio into frames of SIZE bytes; the last one may be shorter."""
    for start in range(0, len(audio), size):
        yield audio[start:start + size]


async def texts(text):
    """TEXT as pieces: a whole reply is one piece, a stream of sentences is passed through."""
    if isinstance(text, str):
        yield text
        return
    async for piece in text:
        yield piece


@dataclass(frozen=True)
class Heard:
    """What the recognizer reports while the other person speaks.

    `started`: they began to speak (used to stop the voice when they talk over it).
    `text`: what they said so far, in Standard German; `is_final` once it will not change.
    `end_of_turn`: they have finished, so the brain may answer.
    """
    text: str = ''
    is_final: bool = False
    end_of_turn: bool = False
    started: bool = False

    @classmethod
    def speech_started(cls):
        return cls(started=True)

    @classmethod
    def partial(cls, text):
        return cls(text=text)

    @classmethod
    def final(cls, text, *, end_of_turn=False):
        return cls(text=text, is_final=True, end_of_turn=end_of_turn)


@dataclass(frozen=True)
class BrainEvent:
    """What the brain produces for one turn, in order.

    `say`: a sentence ready to speak, sent as soon as it is complete so speech can start early.
    `tool`: a tool call such as end_call, with its arguments.
    `done`: the turn is over.
    """
    kind: str
    text: str = ''
    name: str = ''
    arguments: dict = field(default_factory=dict)

    @classmethod
    def say(cls, text):
        return cls('say', text=text)

    @classmethod
    def tool(cls, name, arguments=None):
        return cls('tool', name=name, arguments=dict(arguments or {}))

    @classmethod
    def done(cls):
        return cls('done')


@runtime_checkable
class SpeechRecognizer(Protocol):
    """Turns the other person's audio into Standard German text, with turn ends."""

    async def start(self, *, language: str, keywords: Sequence[str] = ()) -> None:
        """Open a stream for LANGUAGE (de-CH or de-DE); KEYWORDS are names and places to favour."""

    async def send_audio(self, chunk: bytes) -> None:
        """Pass on mu-law 8 kHz audio from the phone line."""

    def events(self) -> AsyncIterator[Heard]:
        """What was heard, as it is heard."""

    async def close(self) -> None:
        """End the stream."""


@runtime_checkable
class VoiceBrain(Protocol):
    """Decides what to say: the brief, the conversation so far, and tools such as end_call."""

    def reply(self, heard: Optional[str]) -> AsyncIterator[BrainEvent]:
        """Answer one turn. HEARD is None for the opening, when nobody has spoken yet."""

    def interrupted(self, spoken: str) -> None:
        """They talked over the last reply: only SPOKEN reached them, the rest was dropped."""


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Turns text into mu-law 8 kHz audio for the phone line, one reply at a time."""

    async def start(self) -> None:
        """Get ready to speak (open the provider's stream for the call)."""

    def speak(self, text: Union[str, AsyncIterable[str]], *,
              language: str) -> AsyncIterator[bytes]:
        """Audio for one reply, frame by frame, as it is made. TEXT is the whole reply, or its
        sentences as the brain writes them, so speech can start on the first one."""

    async def cancel(self, *, heard_ms: Optional[int] = None) -> Optional[str]:
        """Stop the reply in progress; no more audio comes from it. Given HEARD_MS, how many
        milliseconds of the reply's audio were played, returns the text that was heard, or None
        when that cannot be told."""

    async def close(self) -> None:
        """End the stream."""
