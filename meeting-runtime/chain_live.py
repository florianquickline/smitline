"""The voice chain in GPT-Live's place on a phone call.

A phone call (phone_line.PhoneSession) runs on a voice session: it streams the caller's audio in,
adds notes as the call goes ("time is almost up"), and reads events back: speech to play,
transcripts, tool calls, and the end of the session. ChainLive is such a session made of the
three parts in voice_chain.py: the recognizer hears the caller, the brain (Claude) answers, and
the synthesizer speaks. It sends the events GPT-Live sends, so what a call does around its voice
(pacing, the AI disclosure check, voicemail, hanging up) works the same on both.

GPT-Live decides when to speak and when to stop; the chain decides it here:

- It answers once the caller's turn has ended, or when the call asks for a turn with a note and
  the caller is quiet. The first note opens the call, with what the caller said before it.
- One reply at a time: the next turn starts once the last reply has been heard.
- It stops when the caller talks over it with more than "ja" or "mhm" (and not its own voice
  coming back): the reply is canceled, the speech the caller has not heard is dropped from the
  line (Playback.clear), and the brain learns which words were heard.
- The agent's words go into the transcript once they were heard: when their audio has played,
  or the part that played before an interruption. The AI disclosure check reads that transcript.
"""
import asyncio
import base64
import json
import re
from typing import Protocol, runtime_checkable
from uuid import uuid4

from voice_chain import BYTES_PER_MS, heard_part


# Words that say "I'm listening"; with nothing else, they do not stop the voice.
BACKCHANNELS = frozenset((
    'ja', 'jaja', 'jo', 'joo', 'jä', 'jep', 'mhm', 'mhmm', 'mm', 'hm', 'hmm', 'aha', 'ah', 'oh',
    'okay', 'ok', 'oke', 'genau', 'gut', 'eben', 'klar', 'stimmt', 'richtig', 'super', 'achso',
    'gell', 'ne', 'nä',
))
# Their words stopped this long ago and the recognizer never said the turn ended: it has.
TURN_FALLBACK_SECONDS = 2.5
POLL_SECONDS = 0.05
SPEAKING_KINDS = ('session.commentary.append', 'session.instructions.append')
_WORD = re.compile(r"[\w']+")


def _words(text):
    return _WORD.findall(str(text or '').casefold())


@runtime_checkable
class Playback(Protocol):
    """The line's side of the agent's speech: what the caller has not heard yet."""

    def pending(self) -> bool:
        """Speech is still to be heard: queued for the line or playing."""

    def unplayed_ms(self) -> int:
        """Milliseconds of speech sent out that the caller has not heard yet."""

    async def clear(self) -> None:
        """Drop the speech the caller has not heard."""


class NoPlayback:
    """No line to play on: speech counts as heard once it is made."""

    def pending(self):
        return False

    def unplayed_ms(self):
        return 0

    async def clear(self):
        return None


class _Reply:
    """One answer: its text, its audio as the call took it, and whether it was heard."""

    def __init__(self):
        self.text = ''
        self.task = None
        self.audio_bytes = 0  # audio the call has taken from events()
        self.canceled = False
        self.done = False
        self.delivered = asyncio.Event()  # all its audio was taken
        self.written = asyncio.Event()  # its transcript was taken

    @property
    def audio_ms(self):
        return self.audio_bytes // BYTES_PER_MS


_DELIVERED = object()
_WRITTEN = object()


def function_call_event(name, arguments):
    """A tool call in the shape GPT-Live's backend sends it (voice_core.function_call_from)."""
    return {'type': 'response.event', 'delegation_id': None, 'event': {
        'type': 'response.output_item.done',
        'item': {'type': 'function_call', 'call_id': f'call_{uuid4().hex[:16]}', 'name': name,
                 'arguments': json.dumps(arguments)}}}


def usage_event(tokens):
    """One turn's tokens in the shape GPT-Live's backend reports them (backend_usage_from)."""
    cached = tokens.get('cache_read_input_tokens', 0)
    total = tokens.get('input_tokens', 0) + cached + tokens.get('cache_creation_input_tokens', 0)
    return {'type': 'response.event', 'event': {'type': 'response.completed', 'response': {
        'usage': {'input_tokens': total, 'input_tokens_details': {'cached_tokens': cached},
                  'output_tokens': tokens.get('output_tokens', 0)},
        'output': []}}}


class ChainLive:
    """A voice session from a recognizer, a brain and a synthesizer, used like voice_core's
    LiveSession: an async context manager with send_audio, append, events and close.

    PLAYBACK tells the chain what of its speech the caller has heard (phone_line.PacerPlayback).
    """

    def __init__(self, recognizer, brain, synthesizer, *, locale, keywords=(), playback=None):
        self.recognizer = recognizer
        self.brain = brain
        self.synthesizer = synthesizer
        self.locale = locale
        self.keywords = tuple(keywords)
        self.playback = playback or NoPlayback()
        self.model = getattr(brain, 'model', None)
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.session_id = None
        self.usage_seconds = 0  # nothing is billed by the second, as GPT-Live is
        self.close_reason = None
        self._queue = asyncio.Queue()
        self._tasks = []
        self._helpers = set()
        self._wake = asyncio.Event()
        self._closing = False
        self._opened = False
        self._cue_due = False
        self._notes = []
        self._turns = []  # the caller's finished turns not yet answered
        self._turn_text = ''  # their turn in progress, as heard so far
        self._turn_written = ''  # how much of it is in the transcript
        self._words_at = None
        self._reply = None  # the reply being made or played
        self._last = None
        self._tokens = dict(getattr(brain, 'tokens', None) or {})
        self._failure = None

    # The session ----------------------------------------------------------

    async def __aenter__(self):
        await self.recognizer.start(language=self.locale.recognizer_language,
                                    keywords=self.keywords)
        try:
            await self.synthesizer.start()
        except BaseException:
            await self.recognizer.close()
            raise
        self._tasks = [asyncio.create_task(self._listen(), name='chain-listen'),
                       asyncio.create_task(self._converse(), name='chain-converse')]
        self.started.set()
        return self

    async def __aexit__(self, *exc):
        await self.shutdown()

    async def send_audio(self, audio):
        """The caller's audio: mu-law bytes, or base64 as Twilio sends it."""
        if not self.started.is_set() or self._closing:
            return False
        try:
            chunk = base64.b64decode(audio) if isinstance(audio, str) else bytes(audio)
        except ValueError:
            return False
        if not chunk:
            return False
        try:
            await self.recognizer.send_audio(chunk)
        except Exception:
            return False  # the listener notices a lost stream and ends the session
        return True

    async def append(self, kind, content, delegation_id=None):
        """A note from the call. Commentary and instructions want an answer now (the first one
        opens the call); thinking is context for the next turn."""
        del delegation_id
        text = ' '.join(str(content or '').split())
        if kind == 'session.thinking.append':
            if not text:
                return False
            self.brain.note(text)
            return True
        if kind not in SPEAKING_KINDS:
            raise ValueError(f'unsupported append kind {kind}')
        self._opened = True
        if text:
            self._notes.append(text)
        self._cue_due = True
        self._wake.set()
        return True

    async def submit_function_output(self, call_id, output):
        """The chain's only tool, end_call, needs no result."""
        return None

    async def close(self):
        """Begin graceful shutdown; session.closed arrives through events()."""
        self._end('close_requested')

    async def shutdown(self):
        self._closing = True
        self.closed.set()
        reply = self._reply
        tasks = [*self._tasks, *self._helpers]
        if reply is not None and reply.task is not None:
            tasks.append(reply.task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for part in (self.recognizer, self.synthesizer):
            try:
                await part.close()
            except Exception:
                pass
        self._queue.put_nowait(None)

    async def events(self):
        """What GPT-Live would send: audio, transcripts, tool calls, errors, session.closed."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            event, reply, size = item
            if event is _DELIVERED:
                reply.delivered.set()
                continue
            if event is _WRITTEN:
                reply.written.set()
                continue
            if event['type'] == 'session.output_audio.delta':
                if reply.canceled:
                    continue  # made before they interrupted: nobody hears it
                reply.audio_bytes += size
            yield event
            if event['type'] == 'session.closed':
                self.closed.set()
                return

    async def all_heard(self):
        """Wait until the agent's latest reply was heard and its words are in the transcript."""
        if self._last is not None:
            await self._last.written.wait()

    # Listening ------------------------------------------------------------

    async def _listen(self):
        try:
            async for heard in self.recognizer.events():
                await self._hear(heard)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = getattr(error, 'code', None) or 'speech_recognition_failed'
        else:
            code = 'speech_recognition_lost'
        if not self._closing:
            self._error(code)
            self._end('connection_lost')

    async def _hear(self, heard):
        text = ' '.join(str(heard.text or '').split())
        if not text:
            return  # speech started: words decide, not noise
        self._words_at = asyncio.get_running_loop().time()
        reply = self._reply
        if reply is not None and self._stops_for(text, reply):
            await self._interrupt(reply)
        if heard.end_of_turn:
            self._end_their_turn(text)
        else:
            if heard.is_final:
                self._write_their(text)
            self._turn_text = text
        self._wake.set()

    def _write_their(self, text):
        written = self._turn_written
        if written and text.startswith(written):
            delta = text[len(written):]
        else:
            delta = f' {text}'
        if delta.strip():
            self._emit({'type': 'session.input_transcript.delta', 'delta': delta})
        self._turn_written = text

    def _end_their_turn(self, text):
        self._write_their(text)
        self._turns.append(text)
        self._turn_text = self._turn_written = ''

    def _stops_for(self, text, reply):
        """Whether these words from the caller stop the reply."""
        if reply.done or reply.canceled:
            return False
        words = _words(text)
        if not words or all(word in BACKCHANNELS for word in words):
            return False
        if self.playback.pending() and set(words) <= set(_words(reply.text)):
            return False  # its own words coming back down the line
        return True

    async def _interrupt(self, reply):
        reply.canceled = True
        if reply.task is not None and not reply.task.done():
            reply.task.cancel()
            await asyncio.gather(reply.task, return_exceptions=True)
        heard_ms = max(0, reply.audio_ms - self.playback.unplayed_ms())
        try:
            spoken = await self.synthesizer.cancel(heard_ms=heard_ms)
        except Exception:
            spoken = None
        if spoken is None:
            spoken = heard_part(reply.text, heard_ms, reply.audio_ms)
        await self.playback.clear()
        if reply.text:
            self.brain.interrupted(spoken)
        self._emit({'type': 'session.output_audio.interrupted', 'heard_ms': heard_ms})
        self._finish(reply, spoken)

    # Answering ------------------------------------------------------------

    async def _converse(self):
        loop = asyncio.get_running_loop()
        while not self._closing:
            turn = self._next_turn(loop.time())
            if turn is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), POLL_SECONDS)
                except asyncio.TimeoutError:
                    pass
                continue
            reply = await self._respond(*turn)
            await reply.written.wait()

    def _next_turn(self, now):
        """(heard, note) when a turn is due: theirs ended, or the call asked for one."""
        if not self._opened:
            return None
        if self._turn_text and now - self._words_at >= TURN_FALLBACK_SECONDS:
            self._end_their_turn(self._turn_text)
        if self._turn_text or not (self._turns or self._cue_due):
            return None  # they are talking, or nothing to answer
        heard = ' '.join(self._turns) or None
        note = '\n\n'.join(self._notes) or None
        self._turns, self._notes, self._cue_due = [], [], False
        return heard, note

    async def _respond(self, heard, note):
        reply = self._reply = self._last = _Reply()
        reply.task = asyncio.create_task(self._speak(reply, heard, note))
        await asyncio.wait({reply.task})
        if reply.canceled:
            return reply  # _interrupt finishes it
        if reply.task.cancelled():
            self._finish(reply, '')
            return reply
        error = reply.task.exception()
        if error is not None:
            self._error(getattr(error, 'code', None) or 'speech_failed')
            self._finish(reply, '')
            self._end('connection_lost')
            return reply
        helper = asyncio.create_task(self._until_heard(reply))
        self._helpers.add(helper)
        helper.add_done_callback(self._helpers.discard)
        return reply

    async def _speak(self, reply, heard, note):
        if getattr(self.synthesizer, 'is_open', True) is False:
            await self.synthesizer.start()  # the provider closed it while the line was quiet
        tools = []

        async def sentences():
            async for event in self.brain.reply(heard, note=note):
                if event.kind == 'say' and event.text.strip():
                    reply.text = f'{reply.text} {event.text.strip()}'.strip()
                    yield event.text
                elif event.kind == 'tool':
                    tools.append(event)

        async for chunk in self.synthesizer.speak(sentences(), language=self.locale.code):
            if chunk:
                self._emit({'type': 'session.output_audio.delta',
                            'delta': base64.b64encode(chunk).decode('ascii')}, reply, len(chunk))
        self._emit(_DELIVERED, reply)
        self._report_turn()
        # After the audio, so the call has all of it before it starts to hang up.
        for tool in tools:
            self._emit(function_call_event(tool.name, tool.arguments))

    async def _until_heard(self, reply):
        await reply.delivered.wait()
        while not reply.canceled and self.playback.pending():
            await asyncio.sleep(POLL_SECONDS)
        if not reply.canceled:
            self._finish(reply, reply.text)

    def _finish(self, reply, heard_text):
        """The reply is over: what was heard of it goes into the transcript."""
        if reply.done:
            return
        reply.done = True
        if heard_text:
            self._emit({'type': 'session.output_transcript.delta', 'delta': f' {heard_text}'})
        self._emit(_WRITTEN, reply)
        if self._reply is reply:
            self._reply = None

    def _report_turn(self):
        tokens = getattr(self.brain, 'tokens', None)
        if tokens:
            turn = {key: value - self._tokens.get(key, 0) for key, value in tokens.items()}
            self._tokens = dict(tokens)
            if any(turn.values()):
                self._emit(usage_event(turn))
        failure = getattr(self.brain, 'failure', None)
        if failure and failure != self._failure:
            self._failure = failure
            self._error(failure)

    # Events -----------------------------------------------------------------

    def _emit(self, event, reply=None, size=0):
        self._queue.put_nowait((event, reply, size))

    def _error(self, code):
        self._emit({'type': 'error', 'error': {'code': code}})

    def _end(self, reason):
        if self.close_reason is not None:
            return
        self._closing = True
        self.close_reason = reason
        self._emit({'type': 'session.closed', 'reason': reason})


def call_keywords(brief, contact=None):
    """Names the recognizer should expect: who the call is for, and who is called."""
    names = [brief.on_behalf_of]
    if contact and contact.get('name'):
        names.append(contact['name'])
    return names


def open_chain(keys, *, brief, locale, contact=None, playback=None):
    """The voice chain for one call: Deepgram hears and speaks, Claude answers.

    KEYS: {'anthropic': ..., 'deepgram': ...}. The providers' modules load only here, so an
    installation on GPT-Live alone needs none of them.
    """
    from claude_brain import ClaudeBrain
    from deepgram_voice import DeepgramRecognizer, DeepgramSynthesizer, voice_for
    return ChainLive(
        DeepgramRecognizer(keys['deepgram']),
        ClaudeBrain(brief, locale, api_key=keys['anthropic'], contact=contact),
        DeepgramSynthesizer(keys['deepgram'], voice=voice_for(locale.code)),
        locale=locale, keywords=call_keywords(brief, contact), playback=playback)
