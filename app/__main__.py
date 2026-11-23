"""python -m app [--config path]   (or APP_CONFIG=path python -m app)"""
import argparse
import asyncio
import sys

from app import logs, server, settings


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m app", description="Voice front desk server")
    p.add_argument("--config", help="settings file (default: $APP_CONFIG or config/app.toml)")
    args = p.parse_args()
    try:
        s = settings.load(args.config)
        settings.api_key(s)  # fail fast without a key
    except settings.SettingsError as e:
        print(f"settings error: {e}", file=sys.stderr)
        return 2
    logs.setup(s.server.log_level, s.server.log_format)
    try:
        asyncio.run(server.run(s))
    except KeyboardInterrupt:  # Windows: Ctrl-C arrives here (no loop signal handlers)
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
