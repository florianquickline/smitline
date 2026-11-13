"""Which language a call runs in, chosen from the number it dials."""
import unittest

from call_locale import SWISS_GERMAN, STANDARD_GERMAN, call_locale
from call_policy import PolicyProblem


class CallLocaleTests(unittest.TestCase):
    def test_swiss_numbers_run_in_swiss_german(self):
        for number in ('+41791234567', '+41441234567', '+41 79 123 45 67'):
            with self.subTest(number=number):
                locale = call_locale(number)
                self.assertIs(locale, SWISS_GERMAN)
                self.assertEqual(locale.code, 'de-CH')
                self.assertEqual(locale.country, 'CH')

    def test_german_numbers_run_in_standard_german(self):
        for number in ('+4930123456', '+4915112345678'):
            with self.subTest(number=number):
                locale = call_locale(number)
                self.assertIs(locale, STANDARD_GERMAN)
                self.assertEqual(locale.code, 'de-DE')
                self.assertEqual(locale.country, 'DE')

    def test_other_countries_are_refused(self):
        # Fail closed: a call to a country without a language set up never rings.
        for number in ('+14155550142', '+436641234567', '+33612345678', '+447700900123'):
            with self.subTest(number=number), self.assertRaises(PolicyProblem) as caught:
                call_locale(number)
            self.assertEqual(caught.exception.code, 'unsupported_country')
            self.assertIn('+41', caught.exception.message)
            self.assertIn('+49', caught.exception.message)

    def test_unreadable_numbers_are_refused(self):
        for number in ('', None, 'not a number', '0791234567'):
            with self.subTest(number=number), self.assertRaises(PolicyProblem) as caught:
                call_locale(number)
            self.assertEqual(caught.exception.code, 'unsupported_country')

    def test_each_locale_carries_what_the_voice_needs(self):
        self.assertEqual(SWISS_GERMAN.greeting, 'Grüezi')
        self.assertEqual(SWISS_GERMAN.currency, 'Franken')
        self.assertFalse(SWISS_GERMAN.sharp_s)
        self.assertEqual(STANDARD_GERMAN.greeting, 'Guten Tag')
        self.assertEqual(STANDARD_GERMAN.currency, 'Euro')
        self.assertTrue(STANDARD_GERMAN.sharp_s)
        # Both understand Standard German; the Swiss one also hears dialect.
        self.assertEqual(SWISS_GERMAN.recognizer_language, 'de-CH')
        self.assertEqual(STANDARD_GERMAN.recognizer_language, 'de-DE')


if __name__ == '__main__':
    unittest.main()
