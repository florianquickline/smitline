"""Deepgram as the call's ears and voice, each one WebSocket for the whole call.

Nova-3 hears the other person: Swiss German (de-CH) on +41 calls, Standard German (de) on +49
calls, written out in Standard German. Flux speaks the replies in mu-law 8 kHz, ready for the
phone line, with a Swiss voice on +41 calls and a German one on +49 calls.

The API key goes only into the Authorization header, never into a URL, a log, or an error.
"""
import asyncio
import collections
import json
from urllib.parse import urlencode

import aiohttp

from voice_chain import (
    BYTES_PER_MS, ENCODING, MULAW_FRAME_BYTES, SAMPLE_RATE, Heard, frames, heard_part, texts,
)


LISTEN_URL = 'wss://api.deepgram.com/v1/listen'
SPEAK_URL = 'wss://api.deepgram.com/v2/speak'
LISTEN_MODEL = 'nova-3'
# Deepgram's codes: Swiss German has its own, Standard German is plain "de".
LANGUAGES = {'de-CH': 'de-CH', 'de-DE': 'de'}
# Flux voices: Leonie is one of its two Swiss voices, Janine a Standard German one.
VOICES = {'de-CH': 'flux-leonie-de', 'de-DE': 'flux-janine-de'}

# A pause this long ends what they said; people pause mid-thought on the phone.
ENDPOINTING_MS = 300
# A gap this long between words ends their turn even when line noise hides the pause.
UTTERANCE_END_MS = 1000
# Deepgram closes a listening stream after about 10 s without audio.
KEEPALIVE_SECONDS = 5
# Names and places to favour; Nova-3 takes a limited prompt.
MAX_KEYTERMS = 50
CONNECT_SECONDS = 10
INTERRUPT_ANSWER_SECONDS = 1.0
CLOSE_SECONDS = 2.0
# Warnings that answer an Interrupt instead of SpeechInterrupted.
INTERRUPT_WARNINGS = frozenset(('NO_AUDIO_GENERATED', 'INTERRUPT_IN_PROGRESS',
                                'INVALID_INTERRUPT_OFFSET'))


class DeepgramError(Exception):
    """Deepgram could not be reached or refused the stream; `code` says which."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def keyterms(keywords):
    """KEYWORDS without blanks or repeats, at most MAX_KEYTERMS of them."""
    terms, seen = [], set()
    for keyword in keywords or ():
        term = ' '.join(str(keyword or '').split())
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)
    return terms[:MAX_KEYTERMS]


def listen_url(language, keywords=(), *, base=LISTEN_URL, model=LISTEN_MODEL):
    """The Nova-3 stream for LANGUAGE (de-CH or de-DE) on telephone audio."""
    if language not in LANGUAGES:
        raise DeepgramError('unsupported_language',
                            f'Deepgram listens in {", ".join(LANGUAGES)}, not {language}.')
    params = [
        ('model', model), ('language', LANGUAGES[language]), ('encoding', ENCODING),
        ('sample_rate', SAMPLE_RATE), ('channels', 1), ('interim_results', 'true'),
        ('endpointing', ENDPOINTING_MS), ('utterance_end_ms', UTTERANCE_END_MS),
        ('vad_events', 'true'), ('smart_format', 'true'),
    ]
    params += [('keyterm', term) for term in keyterms(keywords)]
    return f'{base}?{urlencode(params)}'


def speak_url(voice, *, base=SPEAK_URL):
    """The Flux stream for VOICE, in telephone audio."""
    return f'{base}?{urlencode([("model", voice), ("encoding", ENCODING), ("sample_rate", SAMPLE_RATE)])}'


def voice_for(language):
    """The default voice for a call in LANGUAGE (de-CH or de-DE)."""
    if language not in VOICES:
        raise DeepgramError('unsupported_language',
                            f'Deepgram speaks {", ".join(VOICES)} here, not {language}.')
    return VOICES[language]


async def _open(url, api_key, session):
    """A WebSocket to Deepgram; the session is ours to close when none was given."""
    own = session is None
    if own:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=CONNECT_SECONDS))
    try:
        ws = await session.ws_connect(url, headers={'Authorization': f'Token {api_key}'},
                                      heartbeat=20)
    except aiohttp.WSServerHandshakeError as error:
        if own:
            await session.close()
        code = 'deepgram_auth' if error.status in (401, 403) else f'deepgram_http_{error.status}'
        raise DeepgramError(code, f'Deepgram refused the stream ({error.status}).') from None
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        if own:
            await session.close()
        raise DeepgramError('deepgram_unreachable', 'Deepgram could not be reached.') from None
    return session, own, ws


def _message(message):
    if message.type != aiohttp.WSMsgType.TEXT:
        return {}
    try:
        data = json.loads(message.data)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _key(api_key):
    key = (api_key or '').strip()
    if not key:
        raise ValueError('Deepgram needs an API key (DEEPGRAM_API_KEY).')
    return key


class DeepgramRecognizer:
    """The SpeechRecognizer on Nova-3: what they say, as they say it, and when their turn ends.

    A turn ends at a pause (speech_final) or, when noise hides the pause, after a gap between
    words (UtteranceEnd). Every finished piece of the turn is kept, so `Heard.final(...,
    end_of_turn=True)` carries all of it.
    """

    def __init__(self, api_key, *, url=LISTEN_URL, model=LISTEN_MODEL, session=None):
        self._key = _key(api_key)
        self._url = url
        self._model = model
        self._session = session
        self._own_session = False
        self._ws = None
        self._keepalive = None
        self._last_audio = 0.0
        self._said = []
        self.language = None
        self.request_id = None

    async def start(self, *, language, keywords=()):
        url = listen_url(language, keywords, base=self._url, model=self._model)
        self._session, self._own_session, self._ws = await _open(url, self._key, self._session)
        self.language = language
        self._last_audio = asyncio.get_running_loop().time()
        self._keepalive = asyncio.create_task(self._keep_alive())

    async def send_audio(self, chunk):
        if self._ws is None or self._ws.closed or not chunk:
            return
        await self._ws.send_bytes(chunk)
        self._last_audio = asyncio.get_running_loop().time()

    async def events(self):
        if self._ws is None:
            return
        async for message in self._ws:
            for heard in self._heard(_message(message)):
                yield heard

    async def close(self):
        if self._keepalive is not None:
            self._keepalive.cancel()
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.send_str(json.dumps({'type': 'CloseStream'}))
                await asyncio.wait_for(self._ws.close(), CLOSE_SECONDS)
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, RuntimeError):
                pass
        if self._own_session and self._session is not None:
            await self._session.close()

    async def _keep_alive(self):
        loop = asyncio.get_running_loop()
        while self._ws is not None and not self._ws.closed:
            await asyncio.sleep(KEEPALIVE_SECONDS)
            if loop.time() - self._last_audio >= KEEPALIVE_SECONDS and not self._ws.closed:
                try:
                    await self._ws.send_str(json.dumps({'type': 'KeepAlive'}))
                except (aiohttp.ClientError, ConnectionError, RuntimeError):
                    return

    def _heard(self, data):
        kind = data.get('type')
        if kind == 'SpeechStarted':
            return [Heard.speech_started()]
        if kind == 'UtteranceEnd':
            return self._end_of_turn()
        if kind == 'Metadata':
            self.request_id = data.get('request_id')
            return []
        if kind != 'Results':
            return []
        alternatives = (data.get('channel') or {}).get('alternatives') or [{}]
        transcript = ' '.join(str(alternatives[0].get('transcript') or '').split())
        if not data.get('is_final'):
            return [Heard.partial(' '.join([*self._said, transcript]))] if transcript else []
        if transcript:
            self._said.append(transcript)
        if data.get('speech_final'):
            return self._end_of_turn()
        return [Heard.final(' '.join(self._said))] if transcript else []

    def _end_of_turn(self):
        if not self._said:
            return []
        said, self._said = ' '.join(self._said), []
        return [Heard.final(said, end_of_turn=True)]


class _Reply:
    """One reply on the Flux stream: its text, its audio as it comes, and whether it ended."""

    def __init__(self):
        self.audio = asyncio.Queue()
        self.text = ''
        self.sent = False
        self.flushed = False
        self.speech_id = None
        self.session_start_ms = 0
        self.generated_bytes = 0
        self.duration_ms = None
        self.canceled = False
        self.ended = False

    def end(self):
        if not self.ended:
            self.ended = True
            self.audio.put_nowait(None)


class DeepgramSynthesizer:
    """The SpeechSynthesizer on Flux: one stream per call, one turn per reply.

    The reply's sentences go out as they come and one Flush ends the turn, as Flux expects, so
    there is no gap between sentences. On cancel, Flux is told how much was played and says which
    words were heard; when it has already finished the turn, the words are estimated from the
    share of the audio that was played.
    """

    def __init__(self, api_key, *, voice, url=SPEAK_URL, session=None):
        self._key = _key(api_key)
        self.voice = voice
        self._url = url
        self._session = session
        self._own_session = False
        self._ws = None
        self._reader = None
        self._send_lock = asyncio.Lock()
        self._reply = None
        self._waiting = collections.deque()
        self._replies = {}
        self._generating = None
        self._session_ms = 0
        self._interrupt_answer = None
        self.request_id = None
        self.failure = None

    @property
    def is_open(self):
        """Whether a reply can go out now. Flux closes a stream that has been idle a while."""
        return self._ws is not None and not self._ws.closed

    async def start(self):
        """Open the stream, or open it again once Flux has closed it (see is_open)."""
        if self._own_session and self._session is not None and self._session.closed:
            self._session, self._own_session = None, False
        self._session, own, self._ws = await _open(
            speak_url(self.voice, base=self._url), self._key, self._session)
        self._own_session = self._own_session or own
        # Replies and playback offsets belong to one stream.
        self._reply = self._generating = self._interrupt_answer = None
        self._waiting.clear()
        self._replies.clear()
        self._session_ms = 0
        self.failure = None
        try:
            hello = _message(await asyncio.wait_for(self._ws.receive(), CONNECT_SECONDS))
        except asyncio.TimeoutError:
            hello = {}
        if hello.get('type') != 'Connected':
            await self.close()
            raise DeepgramError(hello.get('code') or 'deepgram_unreachable',
                                hello.get('description') or 'Deepgram did not open the voice.')
        self.request_id = hello.get('request_id')
        self._reader = asyncio.create_task(self._read())

    def speak(self, text, *, language):
        """Audio for one reply. The voice was chosen for the call; LANGUAGE fits the interface."""
        del language
        return self._speak(text)

    async def cancel(self, *, heard_ms=None):
        reply = self._reply
        if reply is None or reply.canceled:
            return None
        reply.canceled = True
        reply.end()
        generating = reply.sent and reply.duration_ms is None
        if not generating:
            return None if heard_ms is None else heard_part(reply.text, heard_ms,
                                                              reply.duration_ms)
        loop = asyncio.get_running_loop()
        self._interrupt_answer = answer = loop.create_future()
        interrupt = {'type': 'Interrupt'}
        if heard_ms is not None:
            offset = reply.session_start_ms + max(heard_ms, 0)
            interrupt['playback_offset'] = {'type': 'time_ms', 'value': offset}
        await self._send(interrupt)
        if not reply.flushed:
            # A turn Flux could not interrupt (no audio yet) must still end, or the next reply's
            # text would be added to it. Its audio is dropped.
            reply.flushed = True
            await self._send({'type': 'Flush'})
        try:
            spoken = await asyncio.wait_for(answer, INTERRUPT_ANSWER_SECONDS)
        except asyncio.TimeoutError:
            spoken = None
        if spoken and heard_ms is not None and 'text_spoken' in spoken:
            return ' '.join(str(spoken['text_spoken']).split())
        return None

    async def close(self):
        if self._reply is not None:
            self._reply.end()
        if self._ws is not None and not self._ws.closed:
            try:
                await self._send({'type': 'Close'})
                if self._reader is not None:
                    await asyncio.wait_for(asyncio.shield(self._reader), CLOSE_SECONDS)
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, RuntimeError,
                    DeepgramError):
                pass
            await self._ws.close()
        if self._reader is not None:
            self._reader.cancel()
        if self._own_session and self._session is not None:
            await self._session.close()

    async def _speak(self, text):
        if self._reply is not None and not self._reply.ended:
            raise RuntimeError('Flux speaks one reply at a time; cancel the last one first.')
        reply = self._reply = _Reply()
        feeder = asyncio.create_task(self._feed(text, reply))
        pending = b''
        try:
            while (chunk := await reply.audio.get()) is not None:
                pending += chunk
                whole = len(pending) - len(pending) % MULAW_FRAME_BYTES
                for frame in frames(pending[:whole]):
                    if reply.canceled:
                        return
                    yield frame
                pending = pending[whole:]
            if pending and not reply.canceled:
                yield pending
            if feeder.done() and not feeder.cancelled() and feeder.exception() is not None:
                raise feeder.exception()
        finally:
            if not feeder.done():
                feeder.cancel()
            reply.end()

    async def _feed(self, text, reply):
        try:
            await self._feed_pieces(text, reply)
        except BaseException:
            reply.end()  # The listener must not wait for audio that will not come.
            raise

    async def _feed_pieces(self, text, reply):
        async for piece in texts(text):
            if reply.ended:
                return
            if not piece.strip():
                continue
            if reply.text and not piece[0].isspace():
                piece = f' {piece}'  # Flux does not add a space between pieces.
            if not reply.sent:
                reply.sent = True
                self._waiting.append(reply)
            reply.text += piece
            await self._send({'type': 'Speak', 'text': piece})
        if not reply.sent:
            reply.end()  # Nothing to say.
        elif not reply.ended:
            reply.flushed = True
            await self._send({'type': 'Flush'})

    async def _send(self, message):
        if self._ws is None or self._ws.closed:
            raise DeepgramError('deepgram_closed', 'The Deepgram voice stream is closed.')
        async with self._send_lock:
            await self._ws.send_str(json.dumps(message))

    async def _read(self):
        try:
            async for message in self._ws:
                if message.type == aiohttp.WSMsgType.BINARY:
                    reply = self._generating
                    if reply is not None:
                        reply.generated_bytes += len(message.data)
                        if not reply.ended:
                            reply.audio.put_nowait(message.data)
                else:
                    self._event(_message(message))
        finally:
            for reply in (self._reply, *self._waiting, *self._replies.values()):
                if reply is not None:
                    reply.end()
            self._answer_interrupt(None)

    def _event(self, data):
        kind = data.get('type')
        if kind == 'SpeechStarted':
            reply = self._waiting.popleft() if self._waiting else None
            if reply is not None:
                reply.speech_id = data.get('speech_id')
                reply.session_start_ms = self._session_ms
                self._replies[reply.speech_id] = reply
            self._generating = reply
        elif kind == 'SpeechMetadata':
            reply = self._finish(data.get('speech_id'))
            duration = int(data.get('audio_duration_ms') or 0)
            self._session_ms += duration
            if reply is not None:
                reply.duration_ms = duration
                reply.end()
        elif kind == 'SpeechInterrupted':
            metadata = data.get('metadata') or {}
            reply = self._finish(metadata.get('speech_id'))
            played = data.get('audio_played_ms')
            if isinstance(played, (int, float)) and played >= self._session_ms:
                self._session_ms = int(played)
            elif reply is not None:
                self._session_ms += reply.generated_bytes // BYTES_PER_MS
            self._answer_interrupt(data)
        elif kind == 'Warning' and data.get('code') in INTERRUPT_WARNINGS:
            self._answer_interrupt(None)
        elif kind == 'Error':
            self.failure = data.get('code') or 'deepgram_error'

    def _finish(self, speech_id):
        reply = self._replies.pop(speech_id, None)
        if reply is None and self._generating is not None and self._generating.speech_id is None:
            reply = self._generating
        if reply is None and self._waiting and self._waiting[0].canceled:
            reply = self._waiting.popleft()  # Interrupted before it started.
        if reply is not None and reply is self._generating:
            self._generating = None
        return reply

    def _answer_interrupt(self, data):
        answer, self._interrupt_answer = self._interrupt_answer, None
        if answer is not None and not answer.done():
            answer.set_result(data)
