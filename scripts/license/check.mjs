/**
 * Quadra-Seer Sovereign — startup license check (imported by server.js).
 *
 * Behavior:
 *   QS_LICENSE_REQUIRED != "1"  → warn-only (dev default; existing flow untouched)
 *   QS_LICENSE_REQUIRED == "1"  → require a valid activation certificate or exit(1)
 *
 * Certificate locations (first hit wins):
 *   QS_ACTIVATION_FILE env, ./.qs-activation.json, ~/.quadra-seer/activation.json
 * Public key: QS_LICENSE_PUBLIC_KEY env (PEM) or ./license-public.pem
 */
import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { loadPublicKey, verifyCert } from "./lib.mjs";

function findCert() {
  const candidates = [
    process.env.QS_ACTIVATION_FILE,
    "./.qs-activation.json",
    join(homedir(), ".quadra-seer", "activation.json"),
  ].filter(Boolean);
  for (const p of candidates) {
    if (existsSync(p)) {
      try {
        const data = JSON.parse(readFileSync(p, "utf8"));
        if (data.cert) return { path: p, cert: data.cert };
      } catch { /* try next */ }
    }
  }
  return null;
}

function loadPubKey() {
  if (process.env.QS_LICENSE_PUBLIC_KEY) return loadPublicKey(process.env.QS_LICENSE_PUBLIC_KEY.replace(/\\n/g, "\n"));
  if (existsSync("./license-public.pem")) return loadPublicKey(readFileSync("./license-public.pem", "utf8"));
  return null;
}

export function assertLicensed() {
  const required = process.env.QS_LICENSE_REQUIRED === "1";
  if (!required) {
    console.warn("[license] QS_LICENSE_REQUIRED != 1 — running without license enforcement (dev mode).");
    return true;
  }
  const pub = loadPubKey();
  if (!pub) {
    console.error("[license] QS_LICENSE_REQUIRED=1 but no public key found (QS_LICENSE_PUBLIC_KEY or ./license-public.pem).");
    process.exit(1);
  }
  const found = findCert();
  if (!found) {
    console.error("[license] No activation certificate found. Run: node scripts/license/activate.mjs --license \"QS1…\"");
    process.exit(1);
  }
  const r = verifyCert(found.cert, pub);
  if (!r.ok || !r.valid) {
    console.error(`[license] Invalid activation certificate (${r.error || "signature/expiry"}). Re-activate this machine.`);
    process.exit(1);
  }
  console.log(`[license] Activated ✓ (license ${r.payload.licenseId}, expires ${r.payload.exp.slice(0, 10)})`);
  return true;
}
