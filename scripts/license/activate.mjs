#!/usr/bin/env node
/**
 * Quadra-Seer Sovereign — activate this machine (customer-facing).
 *
 * One-time online step: binds your license key to this machine and saves an
 * activation certificate locally. The app verifies the certificate offline
 * afterwards — no further network calls.
 *
 *   node scripts/license/activate.mjs --license "QS1...."
 *
 * Env:
 *   QS_ACTIVATION_URL   activation server (default https://quadra-seer-licensing.onrender.com)
 *   QS_ACTIVATION_FILE  where to save the cert (default ./.qs-activation.json)
 */
import { writeFileSync } from "node:fs";
import { machineFingerprint, licenseId } from "./lib.mjs";

const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : undefined;
};
const die = (msg) => { console.error(`error: ${msg}`); process.exit(1); };

const license = flag("--license") || process.env.QS_LICENSE_KEY || die("--license is required");
const server = (process.env.QS_ACTIVATION_URL || "https://quadra-seer-licensing.onrender.com").replace(/\/$/, "");
const outFile = process.env.QS_ACTIVATION_FILE || "./.qs-activation.json";
const fingerprint = machineFingerprint();

console.log(`activating on this machine (fingerprint ${fingerprint.slice(0, 12)}…)…`);
let res;
try {
  res = await fetch(`${server}/activate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ license, fingerprint, label: process.env.QS_MACHINE_LABEL || "" }),
  });
} catch (e) {
  die(`could not reach activation server: ${e.message}`);
}
const data = await res.json().catch(() => ({}));
if (!res.ok) {
  die(data.error ? `${data.error}${data.hint ? " — " + data.hint : ""}` : `activation failed (HTTP ${res.status})`);
}
writeFileSync(outFile, JSON.stringify({
  cert: data.cert,
  licenseId: licenseId(license),
  fingerprint,
  activatedAt: new Date().toISOString(),
  server,
}, null, 2));
console.log(`activated ✓  (${data.devices}/${data.maxDevices} devices used)`);
console.log(`certificate saved to ${outFile} — keep this file; the app checks it on startup.`);
