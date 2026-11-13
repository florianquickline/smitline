"""Claude as the brain of a phone call: it reads what was heard and answers in sentences to speak.

Each turn streams from the Anthropic API, and every sentence goes to the voice as soon as it is
complete, so speech starts while Claude is still writing. The conversation only ever grows: an
interruption is told to Claude in a system message instead of rewriting its last answer, which
keeps the cached prompt and Claude's earlier thinking valid.
"""
import re

import anthropic

from claude_prompts import spoken_text, system_prompt
from voice_chain import BrainEvent


MODEL = 'claude-haiku-5-5'
# Phone answers are short and must start fast: low effort keeps Claude's thinking brief or skips it.
EFFORT = 'low'
# Thinking counts toward the cap too; a spoken answer needs far less.
MAX_TOKENS = 4096
# A caller cannot wait on a silent line: give up on a stalled connection after 10 s, retry once.
TIMEOUT_SECONDS = 10.0
MAX_RETRIES = 1

END_REASONS = ('completed', 'declined', 'voicemail_left', 'wrong_number', 'not_productive', 'other')
# The input is one short enum, so strict validation is worth more than streaming it early.
END_CALL_TOOL = {
    'name': 'end_call',
    'description': (
        'Legt auf. Nur verwenden, nachdem du dich verabschiedet hast oder wenn die Person das '
        'Gespräch klar beendet hat. Gründe: completed (Gespräch normal beendet), declined (die '
        'Person will nicht), voicemail_left (nur nachdem du auf einer Mailbox eine Nachricht '
        'hinterlassen hast), wrong_number (falsche Nummer), not_productive (das Gespräch führt zu '
        'nichts), other (anderer Grund).'),
    'input_schema': {
        'type': 'object',
        'properties': {'reason': {'type': 'string', 'enum': list(END_REASONS)}},
        'required': ['reason'],
        'additionalProperties': False,
    },
    'strict': True,
}

# The first turn, before anyone has spoken: the API needs a user turn to answer.
OPENING_TURN = '(Der Anruf ist verbunden. Die Person hat abgenommen und noch nichts gesagt.)'
# A later turn the call asks for (a note such as "time is almost up") while they said nothing new.
QUIET_TURN = '(Die Person hat nichts Neues gesagt.)'

# A sentence ends at . ! ? or … before a space, or at a line break; a full stop right after a digit
# does not end one ("am 1. Mai").
_SENTENCE_END = re.compile(r'(?:(?<!\d)\.|[!?…])+["“”«»\')]*\s+|\n+')


class SentenceSplitter:
    """Collects streamed text and hands back each sentence once it is complete."""

    def __init__(self):
        self._buffer = ''

    def add(self, text):
        self._buffer += text
        sentences = []
        while (end := _SENTENCE_END.search(self._buffer)):
            sentences.append(self._buffer[:end.end()])
            self._buffer = self._buffer[end.end():]
        return sentences

    def rest(self):
        rest, self._buffer = self._buffer, ''
        return rest


def failure_code(error):
    """A short name for why the API could not answer, for the call's record."""
    if isinstance(error, anthropic.AuthenticationError):
        return 'claude_auth'
    if isinstance(error, anthropic.RateLimitError):
        return 'claude_rate_limited'
    if isinstance(error, anthropic.APIStatusError):
        return f'claude_http_{error.status_code}'
    if isinstance(error, anthropic.APIConnectionError):
        return 'claude_unreachable'
    return 'claude_error'


def trouble_line(brief, locale):
    """What the voice says when Claude cannot answer; it discloses the AI in case nothing has yet."""
    who = brief.on_behalf_of
    return (f'Entschuldigung, hier ist der KI-Assistent von {who}: Ich habe gerade eine technische '
            f'Störung. {who} meldet sich bei Ihnen. {locale.goodbye}.')


def refusal_line(brief, locale):
    """What the voice says when Claude declines to go on."""
    who = brief.on_behalf_of
    return (f'Dazu kann ich leider nichts sagen, ich bin nur der KI-Assistent von {who}. '
            f'{who} meldet sich bei Ihnen. {locale.goodbye}.')


def interruption_note(spoken):
    """The system message after a reply the other person talked over."""
    spoken = ' '.join(str(spoken or '').split())
    if not spoken:
        return 'Die Person hat dich unterbrochen, bevor sie etwas von deiner letzten Antwort hörte.'
    return (f'Die Person hat dich unterbrochen. Von deiner letzten Antwort hat sie nur das gehört: '
            f'"{spoken}". Der Rest kam nicht an; wiederhole Wichtiges, wenn es nötig ist.')


def _end_call_arguments(arguments):
    reason = arguments.get('reason') if isinstance(arguments, dict) else None
    return {'reason': reason if reason in END_REASONS else 'other'}


class ClaudeBrain:
    """The VoiceBrain on Claude: the brief as the system prompt, the call as the conversation.

    `failure` names why the API could not answer (see failure_code), and `tokens` adds up the
    usage of every turn for the call's costs.
    """

    def __init__(self, brief, locale, *, api_key=None, client=None, contact=None, model=MODEL,
                 effort=EFFORT):
        if client is None:
            if not (api_key or '').strip():
                raise ValueError('Claude needs an Anthropic API key (ANTHROPIC_API_KEY).')
            client = anthropic.AsyncAnthropic(api_key=api_key.strip(), timeout=TIMEOUT_SECONDS,
                                              max_retries=MAX_RETRIES)
        self.brief = brief
        self.locale = locale
        self.client = client
        self.model = model
        self.effort = effort
        self.system = system_prompt(brief, locale, contact=contact)
        self.messages = []
        self.failure = None
        self.tokens = {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0,
                       'cache_creation_input_tokens': 0}
        self._open_tools = []
        self._interruption = None
        self._notes = []

    def interrupted(self, spoken):
        """They talked over the last reply and heard only SPOKEN; Claude learns it next turn."""
        self._interruption = interruption_note(spoken)

    def note(self, text):
        """Context from the call for Claude's next turn, such as a hint that a machine answered."""
        text = ' '.join(str(text or '').split())
        if text:
            self._notes.append(text)

    async def reply(self, heard, *, note=None):
        """Answer one turn: `say` per sentence as it is written, then end_call if Claude hung up,
        then `done`. Closing the iterator early (an interruption) keeps what was written.

        HEARD is what they said; None for the opening or for a turn the call asks for with NOTE,
        which Claude reads as a system message after the turn."""
        self._add_user_turn(heard, note)
        sentences = SentenceSplitter()
        written = []
        message = None
        try:
            try:
                async with self.client.messages.stream(**self._request()) as stream:
                    async for event in stream:
                        if event.type != 'text':
                            continue
                        written.append(event.text)
                        for sentence in sentences.add(event.text):
                            if said := spoken_text(sentence, self.locale):
                                yield BrainEvent.say(said)
                    message = await stream.get_final_message()
            except anthropic.APIError as error:
                self.failure = failure_code(error)
                async for event in self._hang_up(trouble_line(self.brief, self.locale)):
                    yield event
                return
            self._count(message.usage)
            if message.stop_reason == 'refusal':
                self.failure = 'claude_refused'
                message = None  # Keep only its text: the call ends here anyway.
                async for event in self._hang_up(refusal_line(self.brief, self.locale)):
                    yield event
                return
            if said := spoken_text(sentences.rest(), self.locale):
                yield BrainEvent.say(said)
            for block in message.content:
                if block.type == 'tool_use' and block.name == 'end_call':
                    yield BrainEvent.tool('end_call', _end_call_arguments(block.input))
            yield BrainEvent.done()
        finally:
            self._remember(message, ''.join(written))

    async def _hang_up(self, line):
        yield BrainEvent.say(spoken_text(line, self.locale))
        yield BrainEvent.tool('end_call', {'reason': 'other'})
        yield BrainEvent.done()

    def _request(self):
        return {
            'model': self.model,
            'max_tokens': MAX_TOKENS,
            'system': self.system,
            'tools': [END_CALL_TOOL],
            'thinking': {'type': 'adaptive'},
            'output_config': {'effort': self.effort},
            # Caches the system prompt and the conversation so far; each turn reads it back.
            'cache_control': {'type': 'ephemeral'},
            'messages': self.messages,
        }

    def _add_user_turn(self, heard, note=None):
        # A system message must be followed by Claude's turn; if the last turn left none (cut off
        # before Claude wrote anything), the note moves behind the new user turn.
        stranded = None
        if self.messages and self.messages[-1]['role'] == 'system':
            stranded = self.messages.pop()['content']
        opening = not any(turn['role'] == 'assistant' for turn in self.messages)
        notes = [text for text in (stranded, self._interruption, *self._notes,
                                   ' '.join(str(note or '').split())) if text]
        content = [{'type': 'tool_result', 'tool_use_id': tool_id, 'content': 'ok'}
                   for tool_id in self._open_tools]
        content.append({'type': 'text', 'text': heard or (OPENING_TURN if opening else QUIET_TURN)})
        self._open_tools = []
        self._interruption = None
        self._notes = []
        self.messages.append({'role': 'user', 'content': content})
        if notes:
            self.messages.append({'role': 'system', 'content': '\n\n'.join(notes)})

    def _remember(self, message, written):
        """Add Claude's turn as it came, thinking included; a cut-off turn keeps only its text."""
        if message is not None and any(block.type in ('text', 'tool_use')
                                       for block in message.content):
            self.messages.append({'role': 'assistant', 'content': message.content})
            self._open_tools = [block.id for block in message.content if block.type == 'tool_use']
        elif written.strip():
            self.messages.append({'role': 'assistant', 'content': written})

    def _count(self, usage):
        for key in self.tokens:
            self.tokens[key] += getattr(usage, key, None) or 0
