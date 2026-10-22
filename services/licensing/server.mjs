#!/usr/bin/env node
/**
 * Quadra-Seer Sovereign — licensing service.
 *
 * One Render web service that does three jobs:
 *   1. Stripe webhook fulfillment — mints a QS1 license key, builds a
 *      per-customer watermarked zip, and emails key + download link.
 *   2. One-time activation server — binds a license key to a machine
 *      fingerprint (up to QS_MAX_DEVICES machines), returns a signed
 *      activation certificate. Offline verification afterwards.
 *   3. Licensed download hosting — tokenized, expiring download links.
 *
 * Endpoints:
 *   GET  /health
 *   POST /webhooks/stripe            Stripe events (signature-verified)
 *   POST /activate                   { license, fingerprint, label? } -> { cert }
 *   GET  /download/<token>           licensed zip (expires after 7 days)
 *   GET  /admin/orders              (x-admin-key) recent fulfilled orders
 *   POST /admin/orders/:id/retry     (x-admin-key) re-send license email
 *   GET  /admin/activations         (x-admin-key) recent activations
 *
 * Env:
 *   QS_PRIVATE_KEY            Ed25519 private PEM (or QS_PRIVATE_KEY_FILE)
 *   QS_ADMIN_KEY              admin header value
 *   QS_DATA_DIR               persistent dir (Render disk). Default ./data
 *   QS_DB_FILE                sqlite path (default <QS_DATA_DIR>/licensing.db)
 *   QS_MAX_DEVICES            default 3
 *   QS_CERT_TTL_DAYS          default 365
 *   STRIPE_WEBHOOK_SECRET     whsec_… from the Stripe webhook endpoint
 *   QS_STRIPE_PRICE_ID        recurring price id for renewal matching
 *   PUBLIC_BASE_URL           https://<service>.onrender.com (download links)
 *   RESEND_API_KEY            transactional email (optional; orders queue if absent)
 *   EMAIL_FROM                default Quadra-Seer Licensing <licensing@axiomzetainnovations.com>
 *   PORT                      Render sets this
 */
import { createServer } from "node:http";
import { DatabaseSync } from "node:sqlite";
import {
  createHmac,
  timingSafeEqual,
  randomBytes,
  createPublicKey,
  createPrivateKey,
} from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  mkdirSync,
  existsSync,
  createReadStream,
  statSync,
  readFileSync,
} from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  loadPrivateKey,
  verifyLicense,
  mintLicense,
  mintCert,
  licenseId,
  PRODUCT,
} from "../../scripts/license/lib.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const PORT = parseInt(process.env.PORT || "8787", 10);
const DATA_DIR = process.env.QS_DATA_DIR || join(ROOT, "data");
const DOWNLOADS = join(DATA_DIR, "downloads");
const MAX_DEVICES = parseInt(process.env.QS_MAX_DEVICES || "3", 10);
const TTL_DAYS = parseInt(process.env.QS_CERT_TTL_DAYS || "365", 10);
const ADMIN_KEY = process.env.QS_ADMIN_KEY || "";
const WEBHOOK_SECRET = process.env.STRIPE_WEBHOOK_SECRET || "";
const PRICE_ID =
  process.env.QS_STRIPE_PRICE_ID || "price_1UMu2lRuE1b4AX4Jdvok9UeT";
const PUBLIC_BASE_URL = (process.env.PUBLIC_BASE_URL || "").replace(/\/$/, "");
const RESEND_API_KEY = process.env.RESEND_API_KEY || "";
const EMAIL_FROM =
  process.env.EMAIL_FROM ||
  "Quadra-Seer Licensing <licensing@axiomzetainnovations.com>";
const SUPPORT_EMAIL = "anthony.castro@axiomzetainnovations.com";
const DOWNLOAD_TTL_DAYS = 7;

mkdirSync(DOWNLOADS, { recursive: true });

// ---- keys ----
function loadKey() {
  if (process.env.QS_PRIVATE_KEY)
    return process.env.QS_PRIVATE_KEY.replace(/\\n/g, "\n");
  const f = process.env.QS_PRIVATE_KEY_FILE || "./license-private.pem";
  if (!existsSync(f)) {
    console.error("error: no private key — set QS_PRIVATE_KEY");
    process.exit(1);
  }
  return readFileSync(f, "utf8");
}
const privateKey = loadPrivateKey(loadKey());
const publicPem = createPublicKey(
  createPrivateKey(loadKey())
).export({ type: "spki", format: "pem" });

// ---- db ----
const DB_FILE = process.env.QS_DB_FILE || join(DATA_DIR, "licensing.db");
const db = new DatabaseSync(DB_FILE);
db.exec(`CREATE TABLE IF NOT EXISTS activations (
  license_id TEXT NOT NULL, fingerprint TEXT NOT NULL, label TEXT,
  created_at TEXT NOT NULL, PRIMARY KEY (license_id, fingerprint))`);
db.exec(`CREATE TABLE IF NOT EXISTS stripe_events (
  event_id TEXT PRIMARY KEY, type TEXT NOT NULL, received_at TEXT NOT NULL)`);
db.exec(`CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL,
  license_id TEXT NOT NULL, license_key TEXT NOT NULL,
  download_token TEXT NOT NULL UNIQUE, download_expires_at TEXT NOT NULL,
  stripe_event_id TEXT, stripe_ref TEXT, kind TEXT NOT NULL,
  email_status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL)`);
const byLicense = db.prepare(
  "SELECT fingerprint, label, created_at FROM activations WHERE license_id = ?"
);
const insertActivation = db.prepare(
  "INSERT OR IGNORE INTO activations (license_id, fingerprint, label, created_at) VALUES (?, ?, ?, ?)"
);
const recentActivations = db.prepare(
  "SELECT license_id, fingerprint, label, created_at FROM activations ORDER BY created_at DESC LIMIT 200"
);
const markEvent = db.prepare(
  "INSERT OR IGNORE INTO stripe_events (event_id, type, received_at) VALUES (?, ?, ?)"
);
const insertOrder = db.prepare(
  `INSERT INTO orders (email, license_id, license_key, download_token,
   download_expires_at, stripe_event_id, stripe_ref, kind, email_status, created_at)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
);
const orderById = db.prepare("SELECT * FROM orders WHERE id = ?");
const orderByToken = db.prepare("SELECT * FROM orders WHERE download_token = ?");
const recentOrders = db.prepare(
  "SELECT id, email, license_id, kind, email_status, stripe_ref, created_at FROM orders ORDER BY created_at DESC LIMIT 100"
);
const setEmailStatus = db.prepare("UPDATE orders SET email_status = ? WHERE id = ?");

// ---- helpers ----
function json(res, code, obj) {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(JSON.stringify(obj));
}
function readRaw(req, max = 2_000_000) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let n = 0;
    req.on("data", (c) => {
      n += c.length;
      if (n > max) reject(new Error("body too large"));
      else chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}
const hits = new Map();
function rateLimited(key, limit, windowMs) {
  const now = Date.now();
  const arr = (hits.get(key) || []).filter((t) => now - t < windowMs);
  arr.push(now);
  hits.set(key, arr);
  return arr.length > limit;
}
function clientIp(req) {
  return (
    req.headers["x-forwarded-for"]?.split(",")[0]?.trim() ||
    req.socket.remoteAddress ||
    "?"
  );
}

// ---- Stripe signature ----
function verifyStripeSig(raw, header) {
  if (!WEBHOOK_SECRET || !header) return false;
  const parts = Object.fromEntries(
    String(header)
      .split(",")
      .map((p) => p.split("="))
  );
  const t = parseInt(parts.t, 10);
  const v1 = parts.v1;
  if (!t || !v1) return false;
  if (Math.abs(Date.now() / 1000 - t) > 300) return false;
  const expected = createHmac("sha256", WEBHOOK_SECRET)
    .update(`${t}.${raw.toString("utf8")}`)
    .digest("hex");
  const a = Buffer.from(v1, "hex");
  const b = Buffer.from(expected, "hex");
  return a.length === b.length && timingSafeEqual(a, b);
}

// ---- email ----
async function sendLicenseEmail({ to, license, downloadUrl, kind }) {
  if (!RESEND_API_KEY) return { ok: false, reason: "no_resend_key" };
  const subject =
    kind === "renewal"
      ? "Your renewed Quadra-Seer Sovereign license key"
      : "Your Quadra-Seer Sovereign license key";
  const oneTime = kind === "one-time";
  const text = [
    oneTime
      ? `Thanks for purchasing Quadra-Seer Sovereign — it's yours.`
      : `Thanks for subscribing to Quadra-Seer Sovereign.`,
    ``,
    `Your license key (keep it safe):`,
    license,
    ``,
    `Your licensed download (link expires in ${DOWNLOAD_TTL_DAYS} days):`,
    downloadUrl,
    ``,
    `Getting started:`,
    `1. Unzip the bundle on your machine (Docker + Ollama required).`,
    `2. Activate once:  node scripts/license/activate.mjs --license "<your key>"`,
    `   (up to ${MAX_DEVICES} machines per license)`,
    `3. Start the stack:  docker compose up -d --build`,
    ``,
    `The activation is a single online call; afterwards the app verifies`,
    ...(oneTime
      ? [
          `your license fully offline. Your key never expires — this version`,
          `of Sovereign is yours forever, no subscription, no renewals.`,
        ]
      : [
          `your license fully offline. Your key is valid for one month from issue.`,
          `Each renewal emails you a fresh key automatically.`,
        ]),
    ``,
    `Questions? Reply to ${SUPPORT_EMAIL}.`,
    ``,
    `-- Axiom Zeta Innovations LLC`,
  ].join("\n");
  try {
    const r = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${RESEND_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ from: EMAIL_FROM, to, subject, text }),
    });
    if (!r.ok) {
      const body = await r.text().catch(() => "");
      return { ok: false, reason: `resend_${r.status}: ${body.slice(0, 200)}` };
    }
    return { ok: true };
  } catch (e) {
    return { ok: false, reason: String(e).slice(0, 200) };
  }
}

// ---- fulfillment ----
function baseUrl(req) {
  if (PUBLIC_BASE_URL) return PUBLIC_BASE_URL;
  const host = req.headers["x-forwarded-host"] || req.headers.host;
  const proto = req.headers["x-forwarded-proto"] || "https";
  return `${proto}://${host}`;
}

async function fulfill({ email, months, kind, stripeRef, stripeEventId, req }) {
  const { license, payload } = mintLicense({
    email,
    months,
    privateKey,
  });
  const lid = licenseId(license);
  const token = randomBytes(24).toString("hex");
  const zipPath = join(DOWNLOADS, `${token}.zip`);
  try {
    execFileSync(
      "node",
      [join(ROOT, "scripts/dist/package.mjs"), "--license", license, "--out", zipPath],
      {
        cwd: ROOT,
        env: { ...process.env, QS_LICENSE_PUBLIC_KEY: publicPem },
        stdio: "pipe",
        timeout: 300_000,
        maxBuffer: 64 * 1024 * 1024,
      }
    );
  } catch (e) {
    console.error(`package failed for ${email}:`, e.message);
    const out = insertOrder.run(
      email, lid, license, token,
      new Date(Date.now() + DOWNLOAD_TTL_DAYS * 864e5).toISOString(),
      stripeEventId, stripeRef, kind, "failed", new Date().toISOString()
    );
    return { ok: false, orderId: Number(out.lastInsertRowid) };
  }
  const downloadUrl = `${baseUrl(req)}/download/${token}`;
  const emailed = await sendLicenseEmail({ to: email, license, downloadUrl, kind });
  const out = insertOrder.run(
    email, lid, license, token,
    new Date(Date.now() + DOWNLOAD_TTL_DAYS * 864e5).toISOString(),
    stripeEventId, stripeRef, kind,
    emailed.ok ? "sent" : "pending",
    new Date().toISOString()
  );
  console.log(
    `fulfilled ${kind} for ${email} (${lid}) email=${emailed.ok ? "sent" : emailed.reason}`
  );
  return { ok: true, orderId: Number(out.lastInsertRowid), emailed };
}

// ---- server ----
const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const ip = clientIp(req);

  if (req.method === "GET" && url.pathname === "/health") {
    return json(res, 200, { ok: true, product: PRODUCT });
  }

  // ---- Stripe webhook ----
  if (req.method === "POST" && url.pathname === "/webhooks/stripe") {
    let raw;
    try {
      raw = await readRaw(req);
    } catch {
      return json(res, 400, { error: "bad body" });
    }
    if (!verifyStripeSig(raw, req.headers["stripe-signature"])) {
      console.warn("webhook: bad signature");
      return json(res, 400, { error: "bad signature" });
    }
    let event;
    try {
      event = JSON.parse(raw.toString("utf8"));
    } catch {
      return json(res, 400, { error: "invalid JSON" });
    }
    // Idempotency: INSERT OR IGNORE claims the event; duplicates get changes === 0.
    const claimed = markEvent.run(event.id, event.type, new Date().toISOString());
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ received: true }));
    if (claimed.changes === 0) return; // duplicate delivery — already handled

    // Fulfill asynchronously so Stripe never times out.
    setImmediate(async () => {
      try {
        if (event.type === "checkout.session.completed") {
          const s = event.data.object;
          // "subscription" = legacy monthly flow; "payment" = the one-time
          // $77.99 own-it-forever purchase (2026-10-21). Both fulfill here.
          if (s.mode !== "subscription" && s.mode !== "payment") return;
          if (s.metadata?.product !== "quadra-seer") {
            console.log(`webhook: ignoring session ${s.id} (not quadra-seer)`);
            return;
          }
          const email =
            s.customer_details?.email || s.customer_email || "";
          if (!email) {
            console.error(`webhook: no email on session ${s.id}`);
            return;
          }
          const dup = db
            .prepare("SELECT 1 FROM orders WHERE stripe_event_id = ?")
            .get(event.id);
          if (dup) return;
          if (s.mode === "payment") {
            await fulfill({
              email, months: null, kind: "one-time",
              stripeRef: s.payment_intent || s.id, stripeEventId: event.id, req,
            });
          } else {
            await fulfill({
              email, months: 1, kind: "initial",
              stripeRef: s.subscription, stripeEventId: event.id, req,
            });
          }
        } else if (event.type === "invoice.payment_succeeded") {
          const inv = event.data.object;
          if (inv.billing_reason !== "subscription_cycle") return;
          const priceId = inv.lines?.data?.[0]?.price?.id;
          if (priceId !== PRICE_ID) {
            console.log(`webhook: ignoring invoice ${inv.id} (price ${priceId})`);
            return;
          }
          const email = inv.customer_email || "";
          if (!email) return;
          const dup = db
            .prepare("SELECT 1 FROM orders WHERE stripe_event_id = ?")
            .get(event.id);
          if (dup) return;
          await fulfill({
            email, months: 1, kind: "renewal",
            stripeRef: inv.subscription, stripeEventId: event.id, req,
          });
        } else if (event.type === "customer.subscription.deleted") {
          console.log(
            `webhook: subscription ${event.data.object.id} canceled — certs expire on their own`
          );
        }
      } catch (e) {
        console.error("webhook fulfill error:", e);
      }
    });
    return;
  }

  // ---- activation ----
  if (req.method === "POST" && url.pathname === "/activate") {
    if (rateLimited(`act:${ip}`, 30, 3600_000))
      return json(res, 429, { error: "too many requests" });
    let body;
    try {
      body = JSON.parse((await readRaw(req)).toString("utf8"));
    } catch {
      return json(res, 400, { error: "invalid JSON" });
    }
    const { license, fingerprint, label } = body || {};
    if (!license || !fingerprint || !/^[0-9a-f]{32}$/.test(fingerprint)) {
      return json(res, 400, { error: "license and fingerprint (32 hex chars) are required" });
    }
    const check = verifyLicense(license, createPublicKey(privateKey));
    if (!check.ok || !check.valid) {
      return json(res, 403, { error: "invalid or expired license" });
    }
    const lid = licenseId(license);
    const existing = byLicense.all(lid);
    if (existing.some((r) => r.fingerprint === fingerprint)) {
      const { cert } = mintCert({ license, fingerprint, months: check.payload, privateKey, ttlDays: TTL_DAYS });
      return json(res, 200, { cert, renewed: true, devices: existing.length, maxDevices: MAX_DEVICES });
    }
    if (existing.length >= MAX_DEVICES) {
      return json(res, 403, {
        error: "device limit reached", devices: existing.length,
        maxDevices: MAX_DEVICES, hint: "Contact support to reset a device.",
      });
    }
    insertActivation.run(lid, fingerprint, String(label || "").slice(0, 120), new Date().toISOString());
    const { cert } = mintCert({ license, fingerprint, months: check.payload, privateKey, ttlDays: TTL_DAYS });
    console.log(`activated ${lid} on ${fingerprint.slice(0, 8)}… (${existing.length + 1}/${MAX_DEVICES})`);
    return json(res, 200, { cert, renewed: false, devices: existing.length + 1, maxDevices: MAX_DEVICES });
  }

  // ---- download ----
  const dl = url.pathname.match(/^\/download\/([0-9a-f]{48})$/);
  if (req.method === "GET" && dl) {
    if (rateLimited(`dl:${ip}`, 60, 3600_000))
      return json(res, 429, { error: "too many requests" });
    const order = orderByToken.get(dl[1]);
    if (!order || new Date(order.download_expires_at) < new Date()) {
      return json(res, 404, { error: "link expired or invalid" });
    }
    const p = join(DOWNLOADS, `${dl[1]}.zip`);
    if (!existsSync(p)) return json(res, 404, { error: "file missing" });
    const st = statSync(p);
    res.writeHead(200, {
      "Content-Type": "application/zip",
      "Content-Length": st.size,
      "Content-Disposition": 'attachment; filename="quadra-seer-sovereign.zip"',
    });
    return createReadStream(p).pipe(res);
  }

  // ---- admin ----
  const authed =
    ADMIN_KEY && req.headers["x-admin-key"] === ADMIN_KEY;
  if (url.pathname.startsWith("/admin/")) {
    if (!authed) return json(res, 401, { error: "unauthorized" });
    if (req.method === "GET" && url.pathname === "/admin/orders") {
      return json(res, 200, { orders: recentOrders.all() });
    }
    if (req.method === "GET" && url.pathname === "/admin/activations") {
      return json(res, 200, { activations: recentActivations.all() });
    }
    const retry = url.pathname.match(/^\/admin\/orders\/(\d+)\/retry$/);
    if (req.method === "POST" && retry) {
      const order = orderById.get(retry[1]);
      if (!order) return json(res, 404, { error: "order not found" });
      const emailed = await sendLicenseEmail({
        to: order.email, license: order.license_key,
        downloadUrl: `${baseUrl(req)}/download/${order.download_token}`,
        kind: order.kind,
      });
      setEmailStatus.run(emailed.ok ? "sent" : "pending", order.id);
      return json(res, 200, { emailed: emailed.ok, reason: emailed.reason });
    }
    return json(res, 404, { error: "not found" });
  }

  return json(res, 404, { error: "not found" });
});

server.listen(PORT, () => {
  console.log(
    `licensing service on :${PORT} (product ${PRODUCT}, max ${MAX_DEVICES} devices/key)`
  );
  if (!WEBHOOK_SECRET) console.warn("warn: STRIPE_WEBHOOK_SECRET not set — webhooks will 400");
  if (!RESEND_API_KEY) console.warn("warn: RESEND_API_KEY not set — license emails will queue as pending");
  if (!ADMIN_KEY) console.warn("warn: QS_ADMIN_KEY not set — admin endpoints disabled");
});
