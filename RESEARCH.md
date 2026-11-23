# Phase 0: How "live" voice works (one page)

## Two architectures
| | Cascaded pipeline | Native speech-to-speech (Realtime API) |
|---|---|---|
| Flow | mic → VAD → STT → LLM → TTS → speaker | mic → **one model** (audio in, audio out) → speaker |
| Latency | sum of 4 stages (~1–2 s) | one hop (~300–800 ms to first audio) |
| Tone/emotion | lost in STT | preserved |
| Control | full (swap any part) | less, but tools/instructions still work |
| Used by | most DIY voice bots | ChatGPT Advanced Voice, Gemini Live |

## What makes it feel "live"? Mostly the design, not the model
1. **Streaming everywhere.** Send ~100 ms mic chunks continuously and play audio deltas as they arrive.
2. **Server-side VAD (turn detection).** The server decides when you stopped talking (`server_vad` = silence-based; `semantic_vad` = meaning-based).
3. **Barge-in.** On `input_audio_buffer.speech_started` → **stop playback now**, then send `conversation.item.truncate` with `audio_end_ms` (= how much the user actually heard) so the model's memory matches reality.
4. **Echo.** If the speaker leaks into the mic, the model interrupts itself. Fix: headphones (simplest).
5. **Full duplex.** Two concurrent loops (send mic / receive events) over one WebSocket, which needs asyncio.

## Architecture (Phase 1)
```
 mic (sounddevice, 24kHz PCM16)                           speaker (24kHz PCM16)
      │ callback → asyncio.Queue                                ▲ playback buffer
      ▼                                                         │ (cleared on barge-in)
 [send loop] ──input_audio_buffer.append (b64)──► OpenAI ───────┘
                                                Realtime WSS
 [recv loop] ◄── response.output_audio.delta, response.output_audio_transcript.delta,
                 input_audio_buffer.speech_started, response.done
```

## OpenAI Realtime: the protocol facts
- `wss://api.openai.com/v1/realtime?model=gpt-realtime-mini`, header `Authorization: Bearer KEY`
- Configure: `{"type":"session.update","session":{"type":"realtime","instructions":..., "audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"turn_detection":{"type":"semantic_vad"}},"output":{"voice":"marin"}}}}`
- Audio in: `{"type":"input_audio_buffer.append","audio":<b64>}`
- Audio out: `response.output_audio.delta` → `delta` (b64 PCM16, 24 kHz)
- Interrupted response → `response.done` with `status: "cancelled"`
- Model: the key (borrowed from DocAssist, loaded at runtime by `live/config.py`) can access `gpt-realtime-mini` only.

## Agentic parts later, on the same socket
`tools` in `session.update` → the model emits a function call → **our code** runs it → we send `conversation.item.create` (function_call_output) + `response.create`. Determinism lives in our code, not the model. RAG = just one more tool.

## Dependencies (.venv, Python 3.11)
`websockets` (WSS) · `sounddevice` (mic/speaker) · `numpy` (audio arrays) · `matplotlib` (plots) · `jupyterlab` (notebooks)

## Notebook plan
01 audio I/O ✅ → 02 first Realtime session (text in, audio out) ✅ → 03 full voice loop ✅ → 04 barge-in & transcripts ✅ → 05 tool calling ✅ → 06 memory → 07 (optional) cascaded pipeline from scratch.
