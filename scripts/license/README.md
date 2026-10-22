# Quadra-Seer Sovereign — licensing & distribution

Sovereign licensing: Ed25519-signed keys, verified offline. No license server
phone-home after the one-time activation.

## Contents

| File | Purpose |
|---|---|
| `lib.mjs` | Shared crypto (mint/verify licenses & certs, machine fingerprint) |
| `mint.mjs` | CLI: `--init`, `--mint`, `--verify` |
| `activate-server.mjs` | One-time activation server (Node 22.5+, SQLite) |
| `activate.mjs` | Customer CLI: activate this machine |
| `check.mjs` | Startup check imported by `server.js` |
| `license_check.py` | Startup check for the Python backend (compiled bundle) |

## One-time setup (keeper of the keys)

```bash
node scripts/license/mint.mjs --init
# → license-private.pem (SECRET — never commit) + license-public.pem
```

`scripts/license/*.pem` is gitignored. Back up the private key offline.

## Fulfilling an order (manual, Phase 1)

1. Confirm the $14.99/mo Stripe payment; copy the customer's email.
2. Mint: `node scripts/license/mint.mjs --mint --email customer@example.com --months 1`
3. Build once per release: `node scripts/dist/build.mjs` → `dist-package/`
4. Package per customer:
   `node scripts/dist/package.mjs --license "QS1.…" --out quadra-seer-customer.zip`
5. Email: the license key, the zip, `EULA.md`, and `README.md` (inside the zip).

## Activation server (deploy once)

```bash
QS_PRIVATE_KEY="$(cat license-private.pem)" QS_ADMIN_KEY=<random> \
QS_DB_FILE=./activations.db node scripts/license/activate-server.mjs
```

- `POST /activate` `{license, fingerprint, label?}` → `{cert}` (max 3 devices/key)
- `GET /health`, `GET /admin/activations` (header `x-admin-key`)
- Deploy target: Render/Railway/VPS, Node ≥ 22.5. Point
  `QS_ACTIVATION_URL` (default `https://license.axiomzetainnovations.com`)
  at it and update `activate.mjs`'s default.
- SQLite file persists activations — use a persistent disk/volume.

## Customer flow

1. `node scripts/license/activate.mjs --license "QS1.…"` → cert saved locally
2. `docker compose up -d --build` (from the licensed bundle)
3. Backend (`QS_LICENSE_REQUIRED=1`) and `server.js` verify the cert offline
   on every start. No further network calls.

## Key hygiene

- Leaked key: let it expire, or rotate everything with `mint.mjs --init --force`
  (invalidates ALL keys — notify customers).
- Device reset: delete the row from `activations` (SQLite) for that license.
- Cert TTL is 365 days; license expiry is enforced at activation AND inside
  the cert (`exp` = min(cert TTL, license expiry)).
