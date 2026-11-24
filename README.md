# Realtime Talk: a voice front desk on OpenAI Realtime

A clinic's phone front desk that patients talk to: it verifies them by SMS code, registers new patients, finds
doctors' free slots and books or cancels appointments, in English or Hindi. It runs on the demo clinic below, or on
DocAssist's database (see DocAssist).

## Try the demo

```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
set OPENAI_API_KEY=sk-...            # your own key; never commit it
.venv\Scripts\python -m app          # open http://127.0.0.1:8000 in Chrome, allow the microphone
```

Press **Start call**; the desk greets you. The demo clinic has three patients (SMS codes appear on the page, under
Text messages): Vikram Nair `98450 12367`, and Asha Sharma with her son Rohan on `98123 45678`. Try:

- "I want to see my appointments." Then give the number and read back the code.
- "Which doctors do you have?" / "डॉक्टर विकास शाह 26 नवंबर के बाद कब फ्री हैं?"
- "Book Dr. Kapoor on Friday at 11." Or register as a new patient with your own name.

Headphones help (a laptop speaker can echo into the microphone). Phone on the same Wi-Fi: `python lan.py`.

## Run

```
python -m app                       # dev: config/app.toml, http://127.0.0.1:8000
python -m app.selfcheck             # free checks, no OpenAI: settings, skills, session, stall recovery, server end to end
APP_CONFIG=config/production.toml OPENAI_API_KEY=... python -m app
docker build -t desk . && docker run -e OPENAI_API_KEY=... -v deskdata:/data -p 8000:8000 desk
```

Behind a TLS proxy in production (browsers need https/wss for the microphone; allow websocket upgrades on `/ws`).
Endpoints: `/` page, `/ws?agent=<name>` conversation, `/healthz` (503 while draining), `/metrics` (Prometheus).

## Add a function

Write a typed function with a docstring in any module under `app/skills/`. That's all: it is discovered at
startup, its schema comes from the type hints, its description, "When:" line and argument help from the docstring.

```python
@skill(needs_patient=True)          # guard: only for a verified caller
def cancel_appointment(ctx, appointment_id: int) -> dict:
    """Cancel one of the verified patient's appointments.

    When: after the caller said yes to cancelling a specific appointment.
    Args:
        appointment_id: from my_appointments
    """
```

A new function in an existing module joins every agent that uses that module (toolset). A new module is one
name added to an agent's `toolsets` in config.

## Engines

`[engine] kind` in config picks how the voice runs; skills, session, guards and the page are the same for all:

- `webrtc`: the browser's audio goes straight to OpenAI over WebRTC; the server creates the call with its key
  and runs tools on a side connection (`app/webrtc.py`). No frozen replies from lost packets.
- `realtime`: audio relayed through our server over a websocket (`app/realtime.py`), with stall recovery.
- `cascade`: speech-to-text → text model + tools → text-to-speech (`app/cascade.py`); each part is a provider
  module in `app/providers/` chosen by `[engine] stt / llm / tts`. Add Sarvam or ElevenLabs as one file.

Phone on the same Wi-Fi: `python lan.py` (HTTPS with a self-signed certificate; the page prints the URL).

## DocAssist

`[store] backend = "docassist"` runs the same skills on DocAssist's MySQL (`app/docassist.py`): its Patient,
Patient_Hospital, Doctor, Doctor_Hospital, doctor_availability and Appointments tables, scoped to
`[docassist] hospital_id`. Credentials: `DOCASSIST_SQL_HOST/_USER/_PASSWORD/_DATABASE`, or in development the
`sql_*` keys of `[docassist] config_ini`'s section. Calls then register and book for real in that database.

```
python -m app.docassist                                   # register, book, cancel on it: one transaction, rolled back
APP__STORE__BACKEND='"docassist"' python lan.py           # the phone desk on DocAssist
```

A store is `connect()` plus named SQL statements (`app/store.py` `Sqlite.SQL`); the rules in `app/domain` only run
statements by name, so another database is one more store.

## Tune without code

`config/*.toml`: model, voice, turn detection per agent, token cap, limits, OTP rules, stall recovery.
`prompts/*.md`: each agent's system prompt (`{clinic}`, `{today}`, `{skills}` are filled in).
Any value: `APP__<SECTION>__<KEY>` environment variables.
