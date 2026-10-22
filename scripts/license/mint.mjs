#!/usr/bin/env node
/**
 * Quadra-Seer Sovereign — license key minting & verification.
 * (Crypto lives in ./lib.mjs; this is the CLI.)
 *
 *   node scripts/license/mint.mjs --init
 *   node scripts/license/mint.mjs --mint --email customer@example.com [--months 1] [--key ./license-private.pem]
 *   node scripts/license/mint.mjs --verify --license "QS1...." [--public ./license-public.pem]
 */
import { readFileSync, writeFileSync, chmodSync, existsSync } from "node:fs";
import {
  generateKeypair,
  loadPrivateKey,
  loadPublicKey,
  mintLicense,
  verifyLicense,
} from "./lib.mjs";

const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : undefined;
};
const has = (name) => args.includes(name);
const die = (msg) => { console.error(`error: ${msg}`); process.exit(1); };

if (has("--init")) {
  const privPath = flag("--out-private") || "./license-private.pem";
  const pubPath = flag("--out-public") || "./license-public.pem";
  if (existsSync(privPath) && !has("--force")) die(`${privPath} exists — pass --force to overwrite`);
  const { privatePem, publicPem } = generateKeypair();
  writeFileSync(privPath, privatePem, { mode: 0o600 });
  chmodSync(privPath, 0o600);
  writeFileSync(pubPath, publicPem);
  console.log(`private key → ${privPath}  (KEEP SECRET — never commit)`);
  console.log(`public key  → ${pubPath}  (embed in the app for offline verification)`);
  process.exit(0);
}

if (has("--mint")) {
  const email = flag("--email") || die("--email is required");
  const months = parseInt(flag("--months") || "1", 10);
  if (!Number.isFinite(months) || months < 1) die("--months must be a positive integer");
  const keyPath = flag("--key") || "./license-private.pem";
  if (!existsSync(keyPath)) die(`private key not found at ${keyPath} — run --init first`);
  const privateKey = loadPrivateKey(readFileSync(keyPath, "utf8"));
  const { license, payload } = mintLicense({ email, months, privateKey });
  console.log(license);
  console.error(`\nissued to ${payload.email}, valid until ${payload.exp}`);
  process.exit(0);
}

if (has("--verify")) {
  const license = flag("--license") || die("--license is required");
  const pubPath = flag("--public") || "./license-public.pem";
  if (!existsSync(pubPath)) die(`public key not found at ${pubPath}`);
  const publicKey = loadPublicKey(readFileSync(pubPath, "utf8"));
  const r = verifyLicense(license, publicKey);
  if (!r.ok) { console.log(JSON.stringify(r, null, 2)); process.exit(1); }
  console.log(JSON.stringify({ valid: r.valid, signatureOk: r.signatureOk, notExpired: r.notExpired, payload: r.payload }, null, 2));
  process.exit(r.valid ? 0 : 1);
}

console.error("usage: mint.mjs --init | --mint --email <email> [--months N] | --verify --license <key>");
process.exit(1);
