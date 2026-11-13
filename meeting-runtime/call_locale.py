"""The language a phone call runs in, chosen from the number it dials.

Smitline calls Switzerland and Germany. A Swiss number is heard as Swiss German, so the speech
recognizer is set to de-CH, which turns dialect into Standard German text; the voice answers in
Swiss Standard German ("Grüezi", "ss" for "ß", francs). A German number runs in Standard German.
Any other country is refused before dialing: there is no language set up for it (fail closed).
"""
from dataclasses import dataclass

import phonenumbers

from call_policy import PolicyProblem


@dataclass(frozen=True)
class CallLocale:
    code: str
    country: str
    recognizer_language: str
    greeting: str
    goodbye: str
    currency: str
    # Whether written text may use "ß". Swiss Standard German writes "ss" instead.
    sharp_s: bool


SWISS_GERMAN = CallLocale(code='de-CH', country='CH', recognizer_language='de-CH',
                          greeting='Grüezi', goodbye='Auf Wiederhören', currency='Franken',
                          sharp_s=False)
STANDARD_GERMAN = CallLocale(code='de-DE', country='DE', recognizer_language='de-DE',
                             greeting='Guten Tag', goodbye='Auf Wiederhören', currency='Euro',
                             sharp_s=True)

LOCALES = {41: SWISS_GERMAN, 49: STANDARD_GERMAN}
UNSUPPORTED = ('Smitline calls numbers in Switzerland (+41) and Germany (+49) only; {number} is '
               'neither. Give the number in international format, such as +41 79 123 45 67.')


def call_locale(number):
    """The locale for a call to NUMBER; PolicyProblem('unsupported_country') for any other."""
    try:
        parsed = phonenumbers.parse(str(number or ''))
    except phonenumbers.NumberParseException:
        parsed = None
    locale = LOCALES.get(parsed.country_code) if parsed else None
    if locale is None:
        raise PolicyProblem('unsupported_country', UNSUPPORTED.format(number=number or 'the number'))
    return locale
