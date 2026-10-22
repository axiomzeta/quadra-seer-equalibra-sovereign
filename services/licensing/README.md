# Quadra-Seer Sovereign — licensing service

One Node service (Render web service) that handles license fulfillment and activation:

- **Stripe webhook** (`POST /webhooks/stripe`): verifies the Stripe signature,
  mints a `QS1` license key, packages a per-customer watermarked zip via
  `scripts/dist/package.mjs`, and emails the key + expiring download link.
  Handles `checkout.session.completed` (new subscriptions) and
  `invoice.payment_succeeded` with `billing_reason=subscription_cycle` (renewals).
- **Activation server** (`POST /activate`): one-time binding of a license key
  to a machine fingerprint (max `QS_MAX_DEVICES`, default 3). Idempotent
  re-activation. Offline verification afterwards.
- **Downloads** (`GET /download/<token>`): tokenized zip links, 7-day expiry.
- **Admin** (`GET /admin/orders`, `POST /admin/orders/:id/retry`,
  `GET /admin/activations`) — header `x-admin-key`.

## Render deploy

- Repo: `quadra-seer-equalibra-sovereign`, branch `main`, root directory = repo root
- Build command: `npm ci --include=dev && node scripts/dist/build.mjs`
- Start command: `node services/licensing/server.mjs`
- Disk: 1 GB+ mounted at `/data`
- Env vars:

| Var | Value |
|---|---|
| `QS_DATA_DIR` | `/data` |
| `QS_PRIVATE_KEY` | production Ed25519 private PEM (secret) |
| `QS_ADMIN_KEY` | long random string (secret) |
| `STRIPE_WEBHOOK_SECRET` | `whsec_…` from the Stripe webhook endpoint |
| `QS_STRIPE_PRICE_ID` | `price_1UMu2lRuE1b4AX4Jdvok9UeT` |
| `PUBLIC_BASE_URL` | `https://<service>.onrender.com` |
| `RESEND_API_KEY` | Resend API key (secret; email queues as pending if absent) |
| `EMAIL_FROM` | `Quadra-Seer Licensing <licensing@axiomzetainnovations.com>` |
| `QS_MAX_DEVICES` | `3` |

## Stripe webhook endpoint

URL: `https://<service>.onrender.com/webhooks/stripe`
Events: `checkout.session.completed`, `invoice.payment_succeeded`
(copy the signing secret into `STRIPE_WEBHOOK_SECRET` and redeploy/restart).

## Testing

```bash
# signature test: craft an event, sign with the webhook secret
node -e "
const c=require('node:crypto');
const secret=process.env.STRIPE_WEBHOOK_SECRET;
const payload=JSON.stringify({id:'evt_test_1',type:'checkout.session.completed',data:{object:{id:'cs_test',mode:'subscription',metadata:{product:'quadra-seer'},customer_details:{email:'you@example.com'},subscription:'sub_test'}}});
const t=Math.floor(Date.now()/1000);
const sig=c.createHmac('sha256',secret).update(t+'.'+payload).digest('hex');
fetch('https://<service>.onrender.com/webhooks/stripe',{method:'POST',headers:{'stripe-signature':'t='+t+',v1='+sig,'content-type':'application/json'},body:payload}).then(r=>r.text()).then(console.log);
"
# then check: curl -H "x-admin-key: $QS_ADMIN_KEY" https://<service>.onrender.com/admin/orders
```

## Key hygiene

- The production private key exists in exactly two places: the Render env var
  and Anthony's offline backup. It is never committed.
- Leaked key: `mint.mjs --init --force` rotates everything (invalidates ALL keys).
- Device reset: delete the row from `activations` for that license (via sqlite).
