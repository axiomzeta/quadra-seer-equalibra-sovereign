/**
 * Shared license cryptography for Quadra-Seer Sovereign.
 *
 * Two token types, one Ed25519 keypair:
 *   QS1.<payload>.<sig>   — license key (issued at purchase; emailed to customer)
 *   QSC1.<payload>.<sig>  — activation certificate (issued by the activation
 *                            server, bound to one machine fingerprint)
 *
 * Both verify offline with the public key. No phone-home after activation.
 */
import {
  generateKeyPairSync,
  sign,
  verify,
  createHash,
  createPrivateKey,
  createPublicKey,
} from "node:crypto";
import { hostname, platform, networkInterfaces } from "node:os";

export const LICENSE_PREFIX = "QS1";
export const CERT_PREFIX = "QSC1";
export const PRODUCT = "quadra-seer-sovereign";

const b64u = (buf) => Buffer.from(buf).toString("base64url");
const unb64u = (s) => Buffer.from(s, "base64url");

export function generateKeypair() {
  const { privateKey, publicKey } = generateKeyPairSync("ed25519");
  return {
    privatePem: privateKey.export({ type: "pkcs8", format: "pem" }),
    publicPem: publicKey.export({ type: "spki", format: "pem" }),
  };
}

export function loadPrivateKey(pem) {
  return createPrivateKey(pem);
}

export function loadPublicKey(pem) {
  return createPublicKey(pem);
}

function signPayload(payload, privateKey, prefix) {
  const bytes = Buffer.from(JSON.stringify(payload), "utf8");
  const sig = sign(null, bytes, privateKey);
  return `${prefix}.${b64u(bytes)}.${b64u(sig)}`;
}

function parseToken(token, prefix) {
  const parts = String(token || "").trim().split(".");
  if (parts.length !== 3 || parts[0] !== prefix) {
    return { ok: false, error: "malformed token" };
  }
  let payload;
  try {
    payload = JSON.parse(unb64u(parts[1]).toString("utf8"));
  } catch {
    return { ok: false, error: "payload is not valid JSON" };
  }
  return { ok: true, payload, payloadBytes: unb64u(parts[1]), sig: unb64u(parts[2]) };
}

function verifyToken(token, prefix, publicKey) {
  const parsed = parseToken(token, prefix);
  if (!parsed.ok) return parsed;
  let sigOk = false;
  try {
    sigOk = verify(null, parsed.payloadBytes, publicKey, parsed.sig);
  } catch {
    sigOk = false;
  }
  return { ok: true, signatureOk: sigOk, payload: parsed.payload };
}

/** Short stable id for a license key (used for activation records & watermarks). */
export function licenseId(license) {
  return createHash("sha256").update(String(license).trim()).digest("hex").slice(0, 16);
}

/** Stable-ish fingerprint for the current machine. */
export function machineFingerprint() {
  const macs = Object.values(networkInterfaces())
    .flat()
    .filter((n) => n && !n.internal && n.mac && n.mac !== "00:00:00:00:00:00")
    .map((n) => n.mac)
    .sort()
    .join(",");
  return createHash("sha256")
    .update(`${hostname()}|${platform()}|${macs}`)
    .digest("hex")
    .slice(0, 32);
}

/**
 * Mint a license key. months = subscription length from now.
 * months = null mints a PERPETUAL license (one-time purchase, own-it-forever
 * model, 2026-10-21): plan "perpetual", expiry 100 years out — verifyLicense
 * only checks exp > now, so no other code path needs to know.
 */
export function mintLicense({ email, months, privateKey }) {
  const now = new Date();
  const exp = new Date(now);
  const perpetual = months === null || months === undefined;
  if (perpetual) {
    exp.setFullYear(exp.getFullYear() + 100);
  } else {
    exp.setMonth(exp.getMonth() + months);
  }
  const payload = {
    v: 1,
    product: PRODUCT,
    plan: perpetual ? "perpetual" : "monthly",
    email: String(email).toLowerCase().trim(),
    iat: now.toISOString(),
    exp: exp.toISOString(),
  };
  return { license: signPayload(payload, privateKey, LICENSE_PREFIX), payload };
}

export function verifyLicense(license, publicKey) {
  const r = verifyToken(license, LICENSE_PREFIX, publicKey);
  if (!r.ok) return r;
  const notExpired = new Date(r.payload.exp) > new Date();
  return {
    ok: true,
    valid: r.signatureOk && notExpired,
    signatureOk: r.signatureOk,
    notExpired,
    payload: r.payload,
  };
}

/**
 * Mint an activation certificate binding a license to one machine fingerprint.
 * ttlDays defaults to 365 (re-activate yearly; offline in between).
 */
export function mintCert({ license, fingerprint, months, privateKey, ttlDays = 365 }) {
  const now = new Date();
  const exp = new Date(now);
  exp.setDate(exp.getDate() + ttlDays);
  // Cert never outlives the license itself.
  const licExp = new Date(months?.exp || 0);
  const payload = {
    v: 1,
    product: PRODUCT,
    licenseId: licenseId(license),
    fp: fingerprint,
    iat: now.toISOString(),
    exp: (licExp > now && licExp < exp ? licExp : exp).toISOString(),
  };
  return { cert: signPayload(payload, privateKey, CERT_PREFIX), payload };
}

export function verifyCert(cert, publicKey) {
  const r = verifyToken(cert, CERT_PREFIX, publicKey);
  if (!r.ok) return r;
  const notExpired = new Date(r.payload.exp) > new Date();
  return {
    ok: true,
    valid: r.signatureOk && notExpired,
    signatureOk: r.signatureOk,
    notExpired,
    payload: r.payload,
  };
}
