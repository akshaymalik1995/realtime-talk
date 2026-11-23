"""API key + model. The key is loaded at runtime and never stored in this repo or printed."""
import configparser
import os

MODEL = "gpt-realtime-mini"  # only realtime chat model the key can access (checked via /v1/models)
URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

# ponytail: borrows DocAssist's key; set OPENAI_API_KEY to override.
DOCASSIST_INI = r"D:\abhinay\DocAssist\pyDocAssist\config.ini"


def api_key() -> str:
    if key := os.environ.get("OPENAI_API_KEY"):
        return key
    c = configparser.ConfigParser(interpolation=None)
    if not c.read(DOCASSIST_INI) or not c.has_option("dev", "openai_key"):
        raise RuntimeError(f"No OPENAI_API_KEY env var and no [dev] openai_key in {DOCASSIST_INI}")
    return c["dev"]["openai_key"].strip()
