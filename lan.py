"""Dev only: serve the app over HTTPS on the Wi-Fi so a phone can use it.

Phone browsers allow the microphone only on https (or localhost), so this wraps the normal server in TLS with a
self-signed certificate for this PC's Wi-Fi address. Production terminates TLS at a proxy instead.

    python lan.py [ip] [port]      -> open https://<ip>:<port> on the phone, accept the certificate warning once
"""
import asyncio
import functools
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
from pathlib import Path

from app import logs, server, settings

GIT_OPENSSL = Path(r"C:\Program Files\Git\mingw64\bin\openssl.exe")


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  # no packet is sent; just picks the outgoing interface
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def cert_for(ip: str) -> tuple[Path, Path]:
    """A self-signed certificate naming this IP (made once per IP with openssl)."""
    d = Path("data/lan")
    cert, key = d / f"{ip}.pem", d / f"{ip}.key"
    if not cert.exists():
        d.mkdir(parents=True, exist_ok=True)
        openssl = shutil.which("openssl") or str(GIT_OPENSSL)
        subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
                        "-keyout", str(key), "-out", str(cert), "-subj", f"/CN={ip}",
                        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1"], check=True, capture_output=True)
    return cert, key


def main() -> int:
    ip = sys.argv[1] if len(sys.argv) > 1 else lan_ip()
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8443
    s = settings.load(env={**os.environ,
                           "APP__SERVER__HOST": "0.0.0.0",
                           "APP__SERVER__PORT": str(port),
                           "APP__SERVER__ORIGINS": json.dumps([f"https://{ip}:{port}", f"https://127.0.0.1:{port}"])})
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(*cert_for(ip))
    server.serve = functools.partial(server.serve, ssl=ctx)  # ponytail: dev-only patch keeps app/ TLS-free
    logs.setup(s.server.log_level, s.server.log_format)
    print(f"\n  On the phone (same Wi-Fi): https://{ip}:{port}\n", flush=True)
    try:
        asyncio.run(server.run(s))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
