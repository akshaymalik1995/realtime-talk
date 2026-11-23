# Realtime Talk: a voice front desk on OpenAI Realtime

`app/` is the production backend. `live/` + notebooks 01–07 are how it was explored; `course/` rebuilds `app/` step by step.

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

## Tune without code

`config/*.toml`: model, voice, turn detection per agent, token cap, limits, OTP rules, stall recovery.
`prompts/*.md`: each agent's system prompt (`{clinic}`, `{today}`, `{skills}` are filled in).
Any value: `APP__<SECTION>__<KEY>` environment variables.
