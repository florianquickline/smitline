"""Guardrails and the AI disclosure for calls to Switzerland and Germany."""
from datetime import datetime, timezone
import unittest

import phonenumbers

from call_brief import CallBrief, emergency_number
from call_policy import PolicyProblem, check_cost, check_hours
from phone_prompts import discloses


def phone_brief(**overrides):
    payload = {'channel': 'phone', 'to': '+41791234567', 'onBehalfOf': 'Florian Berth',
               'objective': 'Einen Tisch für vier Personen um 19 Uhr reservieren'}
    payload.update(overrides)
    return payload


def premium_example(region):
    return phonenumbers.format_number(
        phonenumbers.example_number_for_type(region, phonenumbers.PhoneNumberType.PREMIUM_RATE),
        phonenumbers.PhoneNumberFormat.E164)


class EmergencyNumberTests(unittest.TestCase):
    SWISS = ('112', '117', '118', '144', '143', '145', '147', '1414')
    GERMAN = ('110', '112', '116117', '116111', '116123', '116006')

    def test_swiss_emergency_and_crisis_numbers_are_never_dialed(self):
        for code in self.SWISS:
            for number in (code, f'+41{code}', f'+41 {code}'):
                with self.subTest(number=number):
                    self.assertTrue(emergency_number(number))

    def test_german_emergency_and_helpline_numbers_are_never_dialed(self):
        for code in self.GERMAN:
            for number in (code, f'+49{code}', f'+49 {code}'):
                with self.subTest(number=number):
                    self.assertTrue(emergency_number(number))

    def test_a_brief_to_a_swiss_emergency_number_is_rejected(self):
        for number in ('+41144', '+411414', '+49116117'):
            with self.subTest(number=number), self.assertRaises(ValueError) as caught:
                CallBrief.from_dict(phone_brief(to=number))
            self.assertIn('never calls emergency', str(caught.exception))

    def test_ordinary_numbers_that_contain_those_digits_are_fine(self):
        for number in ('+41441441234', '+41791441414', '+4930116117', '+49301101234'):
            with self.subTest(number=number):
                self.assertFalse(emergency_number(number))
                CallBrief.from_dict(phone_brief(to=number))


class PremiumNumberTests(unittest.TestCase):
    def test_swiss_premium_numbers_are_refused(self):
        for number in ('+41900123456', '+41901123456', '+41906123456'):
            with self.subTest(number=number), self.assertRaises(PolicyProblem) as caught:
                check_cost(number, {})
            self.assertEqual(caught.exception.code, 'high_cost_number')

    def test_german_premium_numbers_are_refused(self):
        number = premium_example('DE')
        self.assertTrue(number.startswith('+49900'))
        with self.assertRaises(PolicyProblem) as caught:
            check_cost(number, {})
        self.assertEqual(caught.exception.code, 'high_cost_number')

    def test_ordinary_swiss_and_german_numbers_are_allowed(self):
        for number in ('+41791234567', '+41441234567', '+4930123456', '+4915112345678'):
            with self.subTest(number=number):
                check_cost(number, {})


class CallingHoursTests(unittest.TestCase):
    def test_swiss_and_german_numbers_follow_their_local_time(self):
        # 21:30 UTC in November is 22:30 in Zurich and Berlin: too late to call.
        late = datetime(2026, 11, 13, 21, 30, tzinfo=timezone.utc)
        # 09:00 UTC is 10:00 there.
        morning = datetime(2026, 11, 13, 9, 0, tzinfo=timezone.utc)
        for number in ('+41441234567', '+4930123456'):
            with self.subTest(number=number):
                with self.assertRaises(PolicyProblem) as caught:
                    check_hours(number, late, {})
                self.assertEqual(caught.exception.code, 'outside_calling_hours')
                check_hours(number, morning, {})


class GermanDisclosureTests(unittest.TestCase):
    NAME = 'Florian Berth'

    def test_german_openings_count_as_a_disclosure(self):
        for text in (
            'Grüezi Frau Muster, ich rufe im Auftrag von Florian an wegen des Termins. '
            'Ich bin sein KI-Assistent.',
            'Guten Tag, hier ist der KI-Assistent von Florian.',
            'Ich bin Florians KI-Assistent.',
            'Ich bin die KI-Assistentin von Florian.',
            'Kurz zur Info: Ich bin eine KI und rufe für Florian an.',
        ):
            with self.subTest(text=text):
                self.assertTrue(discloses(text, self.NAME))

    def test_ai_as_a_topic_or_a_missing_name_does_not_count(self):
        for text in (
            'Florian baut gerade eine KI-App.',
            'Guten Tag, hier spricht der Assistent von Florian.',
            'Ich bin ein KI-Assistent.',
        ):
            with self.subTest(text=text):
                self.assertFalse(discloses(text, self.NAME))


if __name__ == '__main__':
    unittest.main()
