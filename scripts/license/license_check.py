"""
Quadra-Seer Sovereign — startup license check for the Python backend.

Verifies the activation certificate (QSC1 token) offline using the embedded
Ed25519 public key. Called once at process start; no network calls.

Behavior:
  QS_LICENSE_REQUIRED != "1"  -> warn-only (dev default)
  QS_LICENSE_REQUIRED == "1"  -> require a valid cert or exit(1)

Certificate locations: QS_ACTIVATION_FILE env, ./.qs-activation.json,
~/.quadra-seer/activation.json
Public key: QS_LICENSE_PUBLIC_KEY env (PEM) or ./license-public.pem
"""
import base64
import datetime
import json
import os
import sys
from pathlib import Path

CERT_PREFIX = "QSC1"


def _b64u_decode(s: str) -> bytes:
    s = s.strip()
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s)


def _verify_token(token: str, public_key):
    parts = token.strip().split(".")
    if len(parts) != 3 or parts[0] != CERT_PREFIX:
        return None
    try:
        payload_bytes = _b64u_decode(parts[1])
        sig = _b64u_decode(parts[2])
        public_key.verify(sig, payload_bytes)
        return json.loads(payload_bytes)
    except Exception:
        return None


def assert_licensed() -> bool:
    if os.environ.get("QS_LICENSE_REQUIRED") != "1":
        print("[license] dev mode — no enforcement (set QS_LICENSE_REQUIRED=1 to enforce)")
        return True

    try:
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        print("[license] ERROR: 'cryptography' package required for license check", file=sys.stderr)
        sys.exit(1)

    pem = os.environ.get("QS_LICENSE_PUBLIC_KEY")
    if not pem and Path("./license-public.pem").exists():
        pem = Path("./license-public.pem").read_text()
    if not pem:
        print("[license] ERROR: QS_LICENSE_REQUIRED=1 but no public key (QS_LICENSE_PUBLIC_KEY or ./license-public.pem)", file=sys.stderr)
        sys.exit(1)
    public_key = serialization.load_pem_public_key(pem.encode())

    candidates = [
        os.environ.get("QS_ACTIVATION_FILE"),
        "./.qs-activation.json",
        str(Path.home() / ".quadra-seer" / "activation.json"),
    ]
    cert = None
    for c in candidates:
        if c and Path(c).exists():
            try:
                cert = json.loads(Path(c).read_text())["cert"]
                break
            except Exception:
                continue
    if not cert:
        print('[license] ERROR: no activation certificate. Run: node scripts/license/activate.mjs --license "QS1…"', file=sys.stderr)
        sys.exit(1)

    payload = _verify_token(cert, public_key)
    if not payload:
        print("[license] ERROR: invalid activation certificate signature.", file=sys.stderr)
        sys.exit(1)
    exp = datetime.datetime.fromisoformat(payload["exp"].replace("Z", "+00:00"))
    if exp < datetime.datetime.now(datetime.timezone.utc):
        print("[license] ERROR: activation certificate expired. Re-activate this machine.", file=sys.stderr)
        sys.exit(1)
    print(f"[license] Activated ✓ (license {payload.get('licenseId')}, expires {payload['exp'][:10]})")
    return True


if __name__ == "__main__":
    assert_licensed()
