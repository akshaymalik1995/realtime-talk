"""Numbers the caller said, from their own words (the transcript), in Hindi, Tamil or English.

Why: a speech-to-speech model hearing digits directly repeats or drops one (measured: 9845012367 heard as
98450012367, 8130653310 as 81306553310, three times running), while the transcript of the same audio had them
right. So the transcript is the evidence: `fix_digits` replaces a phone number or code the model got wrong with
the one the caller said. It only steps in when the model's value is impossible (wrong length) and the caller's
words hold exactly one number that fits, so it never overrides a plausible value with a guess.
Dates the same way (`fix_date`): the model heard "एक दिसंबर" (1 December) and looked up 8 December.
"""
import re
from datetime import date, timedelta

# Spoken digits 0-9 as callers say them: English, Hindi, English words written in Devanagari, Tamil (formal and
# everyday). ponytail: single digits only ("बारह" = 12 stays a word)
DIGIT_WORDS = [
    "zero oh शून्य शुन्य जीरो ज़ीरो பூஜ்ஜியம் பூஜ்யம் சைபர் ஜீரோ",
    "one एक वन ஒன்று ஒன்னு",
    "two दो टू ரெண்டு இரண்டு",
    "three तीन थ्री மூன்று மூணு",
    "four चार फोर फ़ोर நான்கு நாலு",
    "five पाँच पांच फाइव फ़ाइव ஐந்து அஞ்சு",
    "six छह छः छै छे सिक्स ஆறு",
    "seven सात सेवन ஏழு",
    "eight आठ एट எட்டு",
    "nine नौ नाइन ஒன்பது ஒம்பது",
]
DIGIT = {w: str(d) for d, words in enumerate(DIGIT_WORDS) for w in words.split()}
REPEAT = {"double": 2, "डबल": 2, "டபுள்": 2, "triple": 3, "ट्रिपल": 3, "ட்ரிபிள்": 3}
_d = "|".join(sorted(DIGIT, key=len, reverse=True)) + r"|\d+"
_unit = rf"(?:(?:{'|'.join(REPEAT)})\s*)?(?:{_d})"
_edge = r"\wऀ-ॣ०-෿"  # letters + Indic letters and vowel signs (not the । ॥ full stops)
DIGIT_RUN = re.compile(rf"(?<![{_edge}]){_unit}(?:[\s,–-]+{_unit})*(?![{_edge}])", re.IGNORECASE)
DIGIT_UNIT = re.compile(rf"(?:({'|'.join(REPEAT)})\s*)?({_d})", re.IGNORECASE)

# Arguments we check, and the lengths a real value can have
VALID = {"phone": (10, 12), "code": (6,)}

# "1 दिसंबर", "एक दिसंबर", "3rd December". ponytail: day as digits or one digit word ("छब्बीस" = 26 is not read)
MONTHS = {m: i for i, names in enumerate([
    "january जनवरी", "february फरवरी फ़रवरी", "march मार्च", "april अप्रैल", "may मई", "june जून", "july जुलाई",
    "august अगस्त", "september सितंबर सितम्बर", "october अक्टूबर", "november नवंबर नवम्बर", "december दिसंबर दिसम्बर"],
    start=1) for m in names.split()}
DATE_SAID = re.compile(rf"(?<![{_edge}])(\d{{1,2}}|{'|'.join(sorted(DIGIT, key=len, reverse=True))})\s*(?:st|nd|rd|th)?"
                       rf"\s+({'|'.join(MONTHS)})", re.IGNORECASE)
RELATIVE = re.compile(r"बाद|पहले|after|before", re.IGNORECASE)  # "26 नवंबर के बाद": the date is a bound, not the day
WEEKDAYS = {w: i for i, names in enumerate(["monday सोमवार", "tuesday मंगलवार", "wednesday बुधवार",
                                            "thursday गुरुवार बृहस्पतिवार", "friday शुक्रवार", "saturday शनिवार",
                                            "sunday रविवार इतवार"]) for w in names.split()}
WEEKDAY_SAID = re.compile("|".join(WEEKDAYS), re.IGNORECASE)


def spoken_digits(text: str) -> str:
    """Numbers said digit by digit become digits: "नौ आठ चार पाँच" -> "9845", "nine double eight" -> "988",
    "81306 5 double 3 10" -> "8130653310". Runs shorter than 3 digits stay as said; runs already in plain digits
    ("10, 11") are left alone."""
    def join(m):
        units = DIGIT_UNIT.findall(m[0])
        if not any(w.lower() in DIGIT or r for r, w in units):
            return m[0]
        digits = "".join(DIGIT.get(w.lower(), w) * REPEAT.get(r.lower(), 1) for r, w in units)
        return digits if len(digits) >= 3 else m[0]
    return DIGIT_RUN.sub(join, text)


def caller_numbers(text: str) -> list[str]:
    """Every number in the caller's words, digit groups joined ("98450 12367" -> "9845012367")."""
    return [re.sub(r"\D", "", g) for g in re.findall(r"\d[\d\s–-]*\d|\d", spoken_digits(text))]


def fix_digits(args: dict, said: str) -> dict[str, tuple[str, str]]:
    """Correct `args` in place from the caller's words. Returns {argument: (model's value, caller's value)}."""
    fixed = {}
    heard = caller_numbers(said)
    for key, lengths in VALID.items():
        v = args.get(key)
        if not isinstance(v, str) or len(re.sub(r"\D", "", v)) in lengths:
            continue
        fits = {n for n in heard if len(n) in lengths}
        if len(fits) == 1:
            args[key] = fits.pop()
            fixed[key] = (v, args[key])
    return fixed


def fix_date(args: dict, said: str, today: date) -> dict[str, tuple[str, str]]:
    """Correct `args["date"]` in place when the caller named exactly one day and month and the model asked for
    another day. Returns {"date": (model's value, caller's value)}."""
    v = args.get("date")
    try:
        model = date.fromisoformat(v)
    except (TypeError, ValueError):
        return {}
    if RELATIVE.search(said):
        return {}  # "after the 26th": the caller gave a bound, not the day
    days = {(int(DIGIT.get(d.lower(), d)), MONTHS[m.lower()]) for d, m in DATE_SAID.findall(said)}
    named = {WEEKDAYS[w.lower()] for w in WEEKDAY_SAID.findall(said)}
    if not days and len(named) == 1:  # "बुधवार को": wrong only if the model's date is another weekday
        wd = named.pop()
        if model.weekday() == wd:
            return {}
        want = today + timedelta(days=(wd - today.weekday() - 1) % 7 + 1)  # the next one after today
    elif len(days) == 1 and len(re.findall(r"\d+", spoken_digits(said))) <= 1:
        day, month = days.pop()
        try:
            want = date(today.year, month, day)
            if want < today:
                want = date(today.year + 1, month, day)
        except ValueError:  # "31 नवंबर": nothing real to correct to
            return {}
        if want == model:
            return {}
    else:
        return {}  # no single plain date ("1 or 2 December", "1 December at 2"): leave it
    args["date"] = want.isoformat()
    return {"date": (v, args["date"])}


if __name__ == "__main__":
    assert spoken_digits("मेरा नंबर है नौ आठ चार पाँच, शून्य एक दो, तीन छः सात।") == "मेरा नंबर है 9845012367।"
    assert spoken_digits("एट वन थ्री जीरो सिक्स फाइव डबल थ्री वन जीरो।") == "8130653310।"
    assert caller_numbers("एट वन थ्री जीरो सिक्स five three three one zero") == ["8130653310"]
    assert caller_numbers("81306 5 double 3 10") == ["8130653310"]
    assert caller_numbers("slots at 10, 11, 12") == ["10", "11", "12"]  # commas separate numbers
    assert spoken_digits("मुझे एक अपॉइंटमेंट चाहिए") == "मुझे एक अपॉइंटमेंट चाहिए"
    a = {"phone": "81306553310"}
    assert fix_digits(a, "एट वन थ्री जीरो सिक्स फाइव डबल थ्री वन जीरो") == {"phone": ("81306553310", "8130653310")}
    a = {"phone": "9845012367"}                     # plausible: never overridden
    assert fix_digits(a, "nine eight four five zero one two three six six") == {} and a["phone"] == "9845012367"
    a = {"code": "49738"}
    assert fix_digits(a, "कोड है चार नौ सात तीन आठ आठ") == {"code": ("49738", "497388")}
    a = {"phone": "123"}                             # nothing that fits in the words: left for the tool to refuse
    assert fix_digits(a, "I don't remember") == {}
    t = date(2026, 11, 24)
    a = {"date": "2026-12-08"}
    assert fix_date(a, "एक दिसंबर को कौन से स्लॉट खाली हैं?", t) == {"date": ("2026-12-08", "2026-12-01")}
    a = {"date": "2026-11-27"}                       # "after the 26th": the model's next day is right
    assert fix_date(a, "26 नवंबर के बाद कब फ्री हैं?", t) == {} and a["date"] == "2026-11-27"
    a = {"date": "2027-01-03"}
    assert fix_date(a, "3rd January please", t) == {}  # already right
    a = {"date": "2026-11-25"}
    assert fix_date(a, "बुधवार को", t) == {} and fix_date(a, "1 or 2 December", t) == {}  # right weekday / two dates
    a = {"date": "2026-11-26"}                       # Thursday for "next Wednesday": the next Wednesday
    assert fix_date(a, "Which slots next Wednesday?", t) == {"date": ("2026-11-26", "2026-11-25")}
    a = {"date": "2026-12-02"}                       # a later Wednesday is a fair reading: kept
    assert fix_date(a, "next Wednesday", t) == {} and fix_date(a, "मंगलवार या बुधवार?", t) == {}
    print("numbers ok")
