# Clinic voice front desk

A phone-style front desk for a clinic, built on the OpenAI Realtime API. A patient talks to it in English or Hindi.
It verifies them with an SMS code, registers new patients, finds a doctor's free slots, and books or cancels
appointments. It runs on a built-in demo clinic (SQLite) or on DocAssist's database.

SMS is simulated: codes appear on the page under "Text messages".

## Run

Python 3.11 or newer, Windows PowerShell:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
$env:OPENAI_API_KEY = "sk-..."
.venv\Scripts\python -m app
```

Open http://127.0.0.1:8000 in Chrome, allow the microphone, press **Start call**. Headphones avoid echo.

## Try it

Demo patients:

| Patient | Mobile |
|---|---|
| Vikram Nair | 98450 12367 |
| Asha Sharma and her son Rohan | 98123 45678 |

Things to say:

- "I want to see my appointments." Give the mobile number, then read the code from the page.
- "Which doctors do you have?" or "डॉक्टर विकास शाह 26 नवंबर के बाद कब फ्री हैं?"
- "Book Dr. Kapoor on Friday at 11." Or register yourself as a new patient.

To use a phone on the same Wi-Fi: `.venv\Scripts\python lan.py` (HTTPS with a self-signed certificate; it prints
the address).

## DocAssist

To run on DocAssist's MySQL instead of the demo clinic, scoped to one hospital:

```
$env:APP__STORE__BACKEND = '"docassist"'
.venv\Scripts\python -m app
```

Credentials come from `DOCASSIST_SQL_HOST`, `DOCASSIST_SQL_USER`, `DOCASSIST_SQL_PASSWORD` and
`DOCASSIST_SQL_DATABASE`, or from the `sql_*` keys of DocAssist's `config.ini` (path, section and hospital in
`[docassist]` of `config/app.toml`). Calls then register and book in that database.
`.venv\Scripts\python -m app.docassist` registers, books and cancels a test patient in one transaction and rolls it
back.

## How it's built

The browser sends audio to OpenAI over WebRTC. The server holds the API key, creates the call and runs the
tools the model asks for: plain Python functions that check every value in code (who is verified, which slots are
free, numbers and dates against the caller's own words) before touching the database.

- `app/skills/` tools the model can call, one typed function each
- `app/domain/` clinic rules: patients, appointments, numbers and dates
- `app/store.py`, `app/docassist.py` the two databases
- `app/server.py`, `app/webrtc.py`, `app/realtime.py` the call
- `app/static/index.html` the page
- `prompts/` what the desk is told; `config/` every setting (`APP__SECTION__KEY` environment variables override)

## Check

```
.venv\Scripts\python -m app.selfcheck
```

Runs every part end to end against a fake OpenAI. No key needed, no cost.
