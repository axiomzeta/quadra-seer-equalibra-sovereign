#!/usr/bin/env node
/**
 * Quadra-Seer Sovereign — activation server.
 *
 * One-time activation: binds a license key to a machine fingerprint (up to
 * QS_MAX_DEVICES machines, default 3), returns a signed activation certificate.
 * The app verifies the certificate offline afterwards — no further phone-home.
 *
 *   QS_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----..."  (same keypair as mint.mjs)
 *   QS_ADMIN_KEY=choose-a-long-random-string
 *   node scripts/license/activate-server.mjs
 *
 * Endpoints:
 *   GET  /health
 *   POST /activate            { license, fingerprint, label? } -> { cert } | 403
 *   GET  /admin/activations   (header x-admin-key) -> recent activations
 *
 * Deploy: any Node 22.5+ host (Render/Railway/VPS). SQLite file at QS_DB_FILE.
 */
import { createServer } from "node:http";
import { DatabaseSync } from "node:sqlite";
import { readFileSync, existsSync } from "node:fs";
import {
  loadPrivateKey,
  verifyLicense,
  mintCert,
  licenseId,
  PRODUCT,
} from "./lib.mjs";

const PORT = parseInt(process.env.PORT || "8787", 10);
const MAX_DEVICES = parseInt(process.env.QS_MAX_DEVICES || "3", 10);
const TTL_DAYS = parseInt(process.env.QS_CERT_TTL_DAYS || "365", 10);
const ADMIN_KEY = process.env.QS_ADMIN_KEY || "";
const DB_FILE = process.env.QS_DB_FILE || "./activations.db";

function loadKey() {
  if (process.env.QS_PRIVATE_KEY) return process.env.QS_PRIVATE_KEY.replace(/\\n/g, "\n");
  const f = process.env.QS_PRIVATE_KEY_FILE || "./license-private.pem";
  if (!existsSync(f)) {
    console.error(`error: no private key — set QS_PRIVATE_KEY or place license-private.pem here`);
    process.exit(1);
  }
  return readFileSync(f, "utf8");
}

const privateKey = loadPrivateKey(loadKey());
import { createPublicKey } from "node:crypto";
const publicKey = createPublicKey(privateKey);

const db = new DatabaseSync(DB_FILE);
db.exec(`CREATE TABLE IF NOT EXISTS activations (
  license_id TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  label TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY (license_id, fingerprint)
)`);

const byLicense = db.prepare("SELECT fingerprint, label, created_at FROM activations WHERE license_id = ?");
const countByLicense = db.prepare("SELECT COUNT(*) AS n FROM activations WHERE license_id = ?");
const insertActivation = db.prepare(
  "INSERT OR IGNORE INTO activations (license_id, fingerprint, label, created_at) VALUES (?, ?, ?, ?)"
);
const recentActivations = db.prepare(
  "SELECT license_id, fingerprint, label, created_at FROM activations ORDER BY created_at DESC LIMIT 200"
);

// Tiny in-memory rate limiter: 30 activations/IP/hour.
const hits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < 3600_000);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > 30;
}

function json(res, code, obj) {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(JSON.stringify(obj));
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let s = "";
    req.on("data", (c) => { s += c; if (s.length > 8192) reject(new Error("body too large")); });
    req.on("end", () => resolve(s));
    req.on("error", reject);
  });
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const ip = req.headers["x-forwarded-for"]?.split(",")[0]?.trim() || req.socket.remoteAddress || "?";

  if (req.method === "GET" && url.pathname === "/health") {
    return json(res, 200, { ok: true, product: PRODUCT });
  }

  if (req.method === "GET" && url.pathname === "/admin/activations") {
    if (!ADMIN_KEY || req.headers["x-admin-key"] !== ADMIN_KEY) return json(res, 401, { error: "unauthorized" });
    return json(res, 200, { activations: recentActivations.all() });
  }

  if (req.method === "POST" && url.pathname === "/activate") {
    if (rateLimited(ip)) return json(res, 429, { error: "too many requests" });
    let body;
    try { body = JSON.parse(await readBody(req)); }
    catch { return json(res, 400, { error: "invalid JSON" }); }
    const { license, fingerprint, label } = body || {};
    if (!license || !fingerprint || !/^[0-9a-f]{32}$/.test(fingerprint)) {
      return json(res, 400, { error: "license and fingerprint (32 hex chars) are required" });
    }

    const check = verifyLicense(license, publicKey);
    if (!check.ok || !check.valid) {
      return json(res, 403, { error: "invalid or expired license", detail: check.error || undefined });
    }

    const lid = licenseId(license);
    const existing = byLicense.all(lid);
    if (existing.some((r) => r.fingerprint === fingerprint)) {
      const { cert } = mintCert({ license, fingerprint, months: check.payload, privateKey, ttlDays: TTL_DAYS });
      return json(res, 200, { cert, renewed: true, devices: existing.length, maxDevices: MAX_DEVICES });
    }
    if (existing.length >= MAX_DEVICES) {
      return json(res, 403, {
        error: "device limit reached",
        devices: existing.length,
        maxDevices: MAX_DEVICES,
        hint: "Contact support to reset a device.",
      });
    }
    insertActivation.run(lid, fingerprint, String(label || "").slice(0, 120), new Date().toISOString());
    const { cert } = mintCert({ license, fingerprint, months: check.payload, privateKey, ttlDays: TTL_DAYS });
    console.log(`activated ${lid} on ${fingerprint.slice(0, 8)}… (${existing.length + 1}/${MAX_DEVICES})`);
    return json(res, 200, { cert, renewed: false, devices: existing.length + 1, maxDevices: MAX_DEVICES });
  }

  return json(res, 404, { error: "not found" });
});

server.listen(PORT, () => {
  console.log(`activation server on :${PORT} (max ${MAX_DEVICES} devices/key, cert TTL ${TTL_DAYS}d)`);
});
