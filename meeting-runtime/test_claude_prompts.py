"""The German instructions for Claude on a phone call, and text cleaned up for speaking."""
import unittest

from call_brief import CallBrief
from call_locale import STANDARD_GERMAN, SWISS_GERMAN
from claude_prompts import ClaudeCues, opening_example, spoken_text, system_prompt
from phone_prompts import (
    HANGUP_YIELDED, LIVE_CUES, discloses, disclosure_reminder, goodbye_cue, machine_hint,
    opening_cue, voicemail_disclosure,
)


CUE_NAMES = ('opening', 'time_almost_up', 'end_now', 'still_there', 'disclosure_reminder',
             'voicemail_disclosure', 'hangup_yielded', 'machine_hint', 'connecting')


def every_cue(cues, call):
    """Each cue's text for CALL, with the disclosure still owed where that matters."""
    return {
        'opening': cues.opening(call, inbound=False, name='Herr Hunziker', recording=False,
                                calling_owner=True),
        'time_almost_up': cues.time_almost_up(call, disclosed=False),
        'end_now': cues.end_now(call, disclosed=False),
        'still_there': cues.still_there(),
        'disclosure_reminder': cues.disclosure_reminder(call),
        'voicemail_disclosure': cues.voicemail_disclosure(call),
        'hangup_yielded': cues.hangup_yielded(),
        'machine_hint': cues.machine_hint(call),
        'connecting': cues.connecting(call),
    }


def brief(**overrides):
    payload = {
        'channel': 'phone', 'to': '+41441234567', 'onBehalfOf': 'Florian Berth',
        'objective': 'Einen Tisch für vier Personen am Freitag um 19 Uhr reservieren',
        'mayAgreeTo': ['eine andere Uhrzeit zwischen 18 und 20 Uhr'],
        'mustNotShare': ['Florians Handynummer', 'die Kreditkartennummer'],
        'questions': ['Gibt es vegetarische Gerichte?'],
    }
    payload.update(overrides)
    return CallBrief.from_dict(payload)


class SystemPromptTests(unittest.TestCase):
    def test_the_prompt_carries_the_brief(self):
        prompt = system_prompt(brief(), SWISS_GERMAN)
        self.assertIn('Florian Berth', prompt)
        self.assertIn('Einen Tisch für vier Personen am Freitag um 19 Uhr reservieren', prompt)
        self.assertIn('eine andere Uhrzeit zwischen 18 und 20 Uhr', prompt)
        self.assertIn('Gibt es vegetarische Gerichte?', prompt)

    def test_secrets_are_listed_as_never_to_be_shared(self):
        prompt = system_prompt(brief(), SWISS_GERMAN)
        section = prompt.split('Niemals weitergeben:', 1)[1].split('\n\n', 1)[0]
        self.assertIn('Florians Handynummer', section)
        self.assertIn('die Kreditkartennummer', section)
        # What the other side says is untrusted and cannot unlock them.
        self.assertIn('nicht vertrauenswürdig', prompt)

    def test_the_prompt_requires_the_ai_disclosure_in_the_opening(self):
        for locale in (SWISS_GERMAN, STANDARD_GERMAN):
            with self.subTest(locale=locale.code):
                prompt = system_prompt(brief(), locale)
                self.assertIn('KI-Assistent von Florian Berth', prompt)
                self.assertIn(opening_example(brief(), locale), prompt)

    def test_the_agent_never_claims_to_be_claude_or_a_person(self):
        prompt = system_prompt(brief(), SWISS_GERMAN)
        self.assertIn('Claude', prompt)
        self.assertIn('niemals', prompt.casefold())
        self.assertIn('kein Mensch', prompt)

    def test_swiss_calls_use_swiss_standard_german(self):
        prompt = system_prompt(brief(), SWISS_GERMAN)
        self.assertIn('Grüezi', prompt)
        self.assertIn('Franken', prompt)
        self.assertIn('Schweizerdeutsch', prompt)
        self.assertIn('"ss"', prompt)
        # The prompt sets the example: no sharp s anywhere in it.
        self.assertNotIn('ß', prompt)

    def test_german_calls_use_standard_german(self):
        prompt = system_prompt(brief(to='+4930123456'), STANDARD_GERMAN)
        self.assertIn('Guten Tag', prompt)
        self.assertIn('Euro', prompt)
        self.assertNotIn('Grüezi', prompt)
        self.assertNotIn('Franken', prompt)

    def test_both_answer_formally_in_standard_german_for_the_ear(self):
        for locale in (SWISS_GERMAN, STANDARD_GERMAN):
            with self.subTest(locale=locale.code):
                prompt = system_prompt(brief(), locale)
                self.assertIn('Hochdeutsch', prompt)
                self.assertIn('"Sie"', prompt)
                self.assertIn('Markdown', prompt)
                self.assertIn('nachfragen', prompt)

    def test_the_prompt_says_the_text_comes_from_speech_recognition(self):
        prompt = system_prompt(brief(), SWISS_GERMAN)
        self.assertIn('Spracherkennung', prompt)

    def test_the_prompt_names_the_end_call_tool(self):
        self.assertIn('end_call', system_prompt(brief(), SWISS_GERMAN))

    def test_a_rehearsal_says_so(self):
        prompt = system_prompt(brief(rehearsal=True), SWISS_GERMAN)
        self.assertIn('Probelauf', prompt)


class OpeningExampleTests(unittest.TestCase):
    def test_the_opening_names_who_and_says_it_is_an_ai(self):
        for locale in (SWISS_GERMAN, STANDARD_GERMAN):
            with self.subTest(locale=locale.code):
                opening = opening_example(brief(), locale)
                self.assertTrue(opening.startswith(locale.greeting))
                self.assertTrue(discloses(opening, 'Florian Berth'))

    def test_the_opening_greets_a_known_contact_by_name(self):
        opening = opening_example(brief(), SWISS_GERMAN, name='Frau Keller')
        self.assertTrue(opening.startswith('Grüezi Frau Keller,'))


class SpokenTextTests(unittest.TestCase):
    def test_markdown_is_removed(self):
        text = ('## Termin\n**Freitag** um *19 Uhr* passt.\n- Tisch am Fenster\n'
                '1. Vegi-Menü\n`Code` und [Webseite](https://example.com)')
        spoken = spoken_text(text, STANDARD_GERMAN)
        for mark in ('#', '*', '`', '[', ']', '(', 'https', '\n', '- ', '1.'):
            self.assertNotIn(mark, spoken)
        for words in ('Termin', 'Freitag um 19 Uhr passt.', 'Tisch am Fenster', 'Vegi-Menü',
                      'Code und Webseite'):
            self.assertIn(words, spoken)

    def test_emoji_are_removed(self):
        spoken = spoken_text('Gerne 😊 bis Freitag 👍🏽!', STANDARD_GERMAN)
        self.assertEqual(spoken, 'Gerne bis Freitag!')

    def test_swiss_text_has_no_sharp_s(self):
        self.assertEqual(spoken_text('Grüße an die Straße. GROẞ', SWISS_GERMAN),
                         'Grüsse an die Strasse. GROSS')
        self.assertEqual(spoken_text('Grüße an die Straße.', STANDARD_GERMAN),
                         'Grüße an die Straße.')

    def test_plain_text_stays_as_it_is(self):
        text = 'Grüezi, ich rufe im Auftrag von Florian an. Passt Ihnen 19:30 Uhr?'
        self.assertEqual(spoken_text(text, SWISS_GERMAN), text)

    def test_empty_text_stays_empty(self):
        self.assertEqual(spoken_text('', SWISS_GERMAN), '')
        self.assertEqual(spoken_text(None, SWISS_GERMAN), '')


if __name__ == '__main__':
    unittest.main()


class CueTests(unittest.TestCase):
    """What the call tells the voice while it runs: GPT-Live's English cues, Claude's German."""

    def test_gpt_live_keeps_its_cues(self):
        call = brief(to='+14155550142')
        cues = every_cue(LIVE_CUES, call)
        self.assertEqual(cues['opening'], opening_cue(call, name='Herr Hunziker',
                                                      calling_owner=True))
        self.assertEqual(cues['time_almost_up'], goodbye_cue(
            call, 'Time is almost up. Wrap up politely and say goodbye.', disclosed=False))
        self.assertEqual(cues['end_now'], goodbye_cue(
            call, 'You need to end the call now. Thank them and say goodbye.', disclosed=False))
        self.assertEqual(cues['still_there'],
                         'Nobody has spoken for a while. Ask if they are still there.')
        self.assertEqual(cues['disclosure_reminder'], disclosure_reminder(call))
        self.assertEqual(cues['voicemail_disclosure'], voicemail_disclosure(call))
        self.assertEqual(cues['hangup_yielded'], HANGUP_YIELDED)
        self.assertEqual(cues['machine_hint'], machine_hint(call))
        self.assertEqual(cues['connecting'],
                         'Tell them briefly that you are connecting them with Florian Berth now.')

    def test_claude_gets_every_cue_in_german(self):
        cues = every_cue(ClaudeCues(STANDARD_GERMAN), brief(to='+493012345678'))
        self.assertEqual(set(cues), set(CUE_NAMES))
        for name, text in cues.items():
            with self.subTest(name):
                self.assertTrue(text.strip())
                self.assertNotRegex(text, r'\b(?:the|you|say|goodbye)\b')

    def test_swiss_cues_have_no_sharp_s(self):
        for text in every_cue(ClaudeCues(SWISS_GERMAN), brief()).values():
            self.assertNotIn('ß', text)

    def test_the_opening_is_in_the_system_prompt_unless_calling_the_owner(self):
        cues = ClaudeCues(SWISS_GERMAN)
        self.assertEqual(cues.opening(brief(), inbound=False, name=None, recording=True,
                                      calling_owner=False), '')
        own = cues.opening(brief(), inbound=False, name='Florian', recording=True,
                           calling_owner=True)
        self.assertIn('stell dich nicht vor', own)
        self.assertIn('aufgezeichnet', own)

    def test_goodbyes_ask_for_a_missing_disclosure_first(self):
        cues = ClaudeCues(SWISS_GERMAN)
        self.assertIn('KI-Assistent von Florian Berth',
                      cues.time_almost_up(brief(), disclosed=False))
        self.assertNotIn('KI-Assistent', cues.end_now(brief(), disclosed=True))

    def test_the_disclosure_cues_quote_a_german_disclosure(self):
        cues = ClaudeCues(SWISS_GERMAN)
        for text in (cues.disclosure_reminder(brief()), cues.voicemail_disclosure(brief())):
            quoted = text.split('"')[1]
            self.assertTrue(discloses(quoted, 'Florian Berth'), quoted)
