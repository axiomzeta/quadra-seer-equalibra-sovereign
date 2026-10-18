"""
Atlantean Backend Server

HTTP server that exposes Atlantean Intelligence Core to the Quadra-Seer frontend.
This is the Phase 1 integration - a standalone backend that Quadra-Seer can call.

Run with: python atlantean_backend.py
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
import sys
import os

# Ensure Unicode output works on Windows consoles (cp1252 → UTF-8)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import io
import pickle
import uuid
import time
import hashlib
import json
import re
import base64
import secrets
from importlib import import_module
import redis
import numpy as np
import torch
import requests
from typing import Any, Dict, List
# google-genai is optional — only imported when a consented cloud call is made.
# Never imported at module level so it cannot cause startup errors on Vault.
def _get_genai_client(api_key: str):
    """Lazy-load google-genai and return an authenticated client."""
    try:
        from google import genai as _genai  # type: ignore[import]
        return _genai.Client(api_key=api_key)
    except ImportError as exc:
        raise RuntimeError(
            "google-genai package not installed. "
            "Cloud escalation requires: pip install google-genai"
        ) from exc
from dotenv import load_dotenv

# Lazy-import workspace modules to allow degraded startup when deps are missing
try:
    from atlantean_core.semantic_search import index_manager as _idx_mgr
    from atlantean_core.semantic_search import search_service as _search_svc
    from atlantean_core.semantic_search.path_policy import (
        validate_index_path as _ss_validate,
        is_allowed_file as _ss_allowed,
    )
    _SEMANTIC_SEARCH_AVAILABLE = True
except Exception as _ss_ex:
    _SEMANTIC_SEARCH_AVAILABLE = False
    print(f'ℹ️  Semantic search unavailable: {_ss_ex}')

try:
    from atlantean_core.automation import (
        planner as _auto_planner,
        executor as _auto_exec,
        approvals as _auto_approvals,
        quarantine as _auto_quarantine,
        audit as _auto_audit,
        path_policy as _auto_path,
        schemas as _auto_schemas,
    )
    _AUTOMATION_AVAILABLE = True
except Exception as _auto_ex:
    _AUTOMATION_AVAILABLE = False
    print(f'ℹ️  Automation module unavailable: {_auto_ex}')

from event_normalization import (
    normalize_disconnect_reason,
    normalize_learning_event_payload,
)
from atlantean_core.sovereignty_policy import (
    evaluate_cloud_escalation,
    evaluate_local_url_policy,
)
from atlantean_core.ledger_verification import evaluate_ledger_gate
from atlantean_core.tamper_evidence import (
    snapshot_hash,
    verify_replay_head,
    verify_snapshot_hash,
)
from atlantean_core.corruption_recovery import (
    plan_ledger_recovery,
    plan_snapshot_recovery,
)
from atlantean_core.provider_capabilities import (
    build_provider_capability_registry,
    normalize_provider_name,
    resolve_provider_capability,
)
from atlantean_core.orchestration_strategy import select_orchestration_strategy
from atlantean_core.query_reasoning_budget import (
    build_query_reasoning_budget,
    classify_query_class,
)
from atlantean_core.query_lifecycle_telemetry import (
    build_cycle_trigger_payload,
    build_query_lifecycle_payload,
)
from atlantean_core.retention_policy import plan_item_retention
from atlantean_core.hrm_compute_budget import (
    HRMComputeAllocator,
    build_hrm_compute_budget_policy,
)
from atlantean_core.model_tier_selector import (
    ModelTierSelector,
    build_model_tier_config,
)
from atlantean_core.hardware_scaling_matrix import (
    get_scaling_matrix,
    get_tier_or_default,
    tier_to_dict,
)
from atlantean_core.local_voice import (
    LocalWhisperTranscriber,
    LocalPiperTts,
    check_whisper_available,
    check_piper_available,
)
from atlantean_core.model_integrity import verify_model_files_on_startup
from atlantean_core.hardware_probe import full_probe as _hw_full_probe
from atlantean_core.runtime_config import recommend_runtime as _recommend_runtime
from atlantean_core.process_manager import (
    start_ollama, stop_process, list_processes,
    is_port_free, find_free_port, wait_for_port,
    start_llamacpp_server,
)
from atlantean_core.model_manager import (
    list_all_models, pull_ollama_model, import_gguf_to_ollama,
    delete_ollama_model, get_pull_progress,
)
from atlantean_core.backend_installer import (
    backend_status as _backend_status,
    install_ollama as _install_ollama,
    install_llamacpp as _install_llamacpp,
    detect_platform_key as _detect_platform_key,
)

# Load environment variables.
# In Docker Compose, GEMINI_* may be injected as empty strings; override ensures
# .env.local values are used when present.
load_dotenv('.env.local', override=True)

# Import the bridge module
from atlantean_quadra_bridge import AtlanteanQuadraBridge, QuadraLearningEvent
from hrm import HRMAdapter, HRMState

# Configure providers — Gemini is OPTIONAL (consent-gated cloud escalation only).
GEMINI_API_KEY: str = (os.getenv('GEMINI_API_KEY') or '').strip()
if GEMINI_API_KEY and GEMINI_API_KEY not in ('PLACEHOLDER_API_KEY', 'YOUR_GEMINI_API_KEY_HERE'):
    print('✅ Gemini API key present (cloud escalation available when consented)')
else:
    GEMINI_API_KEY = ''
    print('ℹ️  No Gemini API key — all inference runs locally')

ATLANTEAN_LLM_PROVIDER = (os.getenv('ATLANTEAN_LLM_PROVIDER') or 'local').strip().lower()
LOCAL_LLM_URL = os.getenv('LOCAL_LLM_URL', 'http://127.0.0.1:11434/api/generate').strip()
LOCAL_LLM_MODEL = os.getenv('LOCAL_LLM_MODEL', 'qwen2.5:3b').strip()
LOCAL_LLM_BACKEND = (os.getenv('LOCAL_LLM_BACKEND') or 'ollama').strip().lower()

# Sentinel integration: drift detection/stabilization for LLM outputs.
# When enabled and the Sentinel gateway is reachable, each LLM response
# is passed through Sentinel's /v1/stabilize before reaching the user.
SENTINEL_ENABLED = (os.getenv('SENTINEL_ENABLED') or 'true').strip().lower() in ('1', 'true', 'yes')
SENTINEL_URL = os.getenv('SENTINEL_URL', 'http://127.0.0.1:8000').strip()
SENTINEL_API_KEY = os.getenv('SENTINEL_API_KEY', 'replace-me').strip()

# Runtime-overridable config (set via /api/atlantean/config/ollama without restart).
_runtime_ollama_url: str = ''


def _backend_path_for(backend: str) -> str:
    """Return the API path suffix for a given backend type."""
    if backend in {'llama.cpp', 'llamacpp', 'llama_cpp'}:
        return '/completion'
    if backend == 'vllm':
        return '/v1/completions'
    return '/api/generate'


def _host_to_url(host: str, backend: str = 'ollama') -> str:
    """Convert a bare host[:port] string to a full LLM API URL."""
    host = host.strip().rstrip('/')
    if host.startswith(('http://', 'https://')):
        # Already a full URL — append path only if missing.
        path = _backend_path_for(backend)
        return host if host.endswith(path) else f'{host}{path}'
    # Bare host — assume http.
    path = _backend_path_for(backend)
    return f'http://{host}{path}'


def _get_local_llm_url(backend: str | None = None) -> str:
    """Return the effective Ollama URL, preferring runtime overrides."""
    if _runtime_ollama_url:
        return _host_to_url(_runtime_ollama_url, backend or LOCAL_LLM_BACKEND)
    return LOCAL_LLM_URL
LOCAL_LLM_TIMEOUT_SECONDS = float(os.getenv('LOCAL_LLM_TIMEOUT_SECONDS', '90'))
LOCAL_CONTEXT_WINDOW = int(os.getenv('ATLANTEAN_LOCAL_CONTEXT_WINDOW', '8192'))
ALLOW_OUTBOUND_NETWORK = os.getenv('ALLOW_OUTBOUND_NETWORK', '0') == '1'
ALLOW_CLOUD_ESCALATION = os.getenv('ALLOW_CLOUD_ESCALATION', '0') == '1'
REQUIRE_CLOUD_ESCALATION_CONSENT = os.getenv('REQUIRE_CLOUD_ESCALATION_CONSENT', '1') == '1'

# Voice sovereignty flags
ALLOW_BROWSER_SPEECH_FALLBACK = os.getenv('ATLANTEAN_ALLOW_BROWSER_SPEECH_FALLBACK', '0') == '1'
REQUIRE_LOCAL_STT = os.getenv('ATLANTEAN_REQUIRE_LOCAL_STT', '1') == '1'

# Journal
JOURNAL_BASE_DIR = os.path.abspath(os.getenv('ATLANTEAN_JOURNAL_DIR', './vault_data/journal'))
VERIFY_LEDGER_ON_RESTORE = os.getenv('ATLANTEAN_VERIFY_LEDGER_ON_RESTORE', '1') == '1'
VERIFY_LEDGER_ON_SYNC_MERGE = os.getenv('ATLANTEAN_VERIFY_LEDGER_ON_SYNC_MERGE', '1') == '1'
VERIFY_SNAPSHOT_TAMPER_EVIDENCE = os.getenv('ATLANTEAN_VERIFY_SNAPSHOT_TAMPER_EVIDENCE', '1') == '1'
ENABLE_CORRUPTION_RECOVERY_REPAIR = os.getenv('ATLANTEAN_ENABLE_CORRUPTION_RECOVERY_REPAIR', '1') == '1'
ENABLE_MEMORY_RETENTION_CONTROLS = os.getenv('ATLANTEAN_ENABLE_MEMORY_RETENTION_CONTROLS', '1') == '1'
RETENTION_CHAT_TURNS_MAX = int(os.getenv('ATLANTEAN_RETENTION_CHAT_TURNS_MAX', '400'))
RETENTION_SIMULATIONS_MAX = int(os.getenv('ATLANTEAN_RETENTION_SIMULATIONS_MAX', '200'))
RETENTION_TELEMETRY_EVENTS_MAX = int(os.getenv('ATLANTEAN_RETENTION_TELEMETRY_EVENTS_MAX', '2000'))
ENABLE_QUERY_CLASS_REASONING_BUDGETS = os.getenv('ATLANTEAN_ENABLE_QUERY_CLASS_REASONING_BUDGETS', '1') == '1'
REASONING_CYCLES_SIMPLE = int(os.getenv('ATLANTEAN_REASONING_CYCLES_SIMPLE', '0'))
REASONING_CYCLES_ANALYTICAL = int(os.getenv('ATLANTEAN_REASONING_CYCLES_ANALYTICAL', '1'))
REASONING_CYCLES_DEEP = int(os.getenv('ATLANTEAN_REASONING_CYCLES_DEEP', '2'))
REASONING_CYCLES_FILE_ASSISTED = int(os.getenv('ATLANTEAN_REASONING_CYCLES_FILE_ASSISTED', '2'))

# Step 23 HRM/RCM compute budget controls.
ENABLE_HRM_COMPUTE_BUDGET = os.getenv('ATLANTEAN_ENABLE_HRM_COMPUTE_BUDGET', '1') == '1'
HRM_COMPUTE_MAX_MS_PER_QUERY = float(os.getenv('ATLANTEAN_HRM_MAX_MS_PER_QUERY', '20.0'))
RCM_COMPUTE_MAX_MS_PER_QUERY = float(os.getenv('ATLANTEAN_RCM_MAX_MS_PER_QUERY', '20.0'))
HRM_COMPUTE_MAX_MS_PER_WINDOW = float(os.getenv('ATLANTEAN_HRM_MAX_MS_PER_WINDOW', '200.0'))
HRM_COMPUTE_WINDOW_SECONDS = int(os.getenv('ATLANTEAN_HRM_COMPUTE_WINDOW_SECONDS', '60'))

HRM_COMPUTE_BUDGET_POLICY = build_hrm_compute_budget_policy(
    enabled=ENABLE_HRM_COMPUTE_BUDGET,
    hrm_max_ms_per_query=HRM_COMPUTE_MAX_MS_PER_QUERY,
    rcm_max_ms_per_query=RCM_COMPUTE_MAX_MS_PER_QUERY,
    max_total_ms_per_window=HRM_COMPUTE_MAX_MS_PER_WINDOW,
    window_seconds=HRM_COMPUTE_WINDOW_SECONDS,
)

# Step 24 dynamic model tier selection.
# Light/heavy defaults fall back to LOCAL_LLM_MODEL when not explicitly configured.
MODEL_TIER_LIGHT = os.getenv('ATLANTEAN_MODEL_TIER_LIGHT', '').strip() or LOCAL_LLM_MODEL
MODEL_TIER_STANDARD = os.getenv('ATLANTEAN_MODEL_TIER_STANDARD', '').strip() or LOCAL_LLM_MODEL
MODEL_TIER_HEAVY = os.getenv('ATLANTEAN_MODEL_TIER_HEAVY', '').strip() or LOCAL_LLM_MODEL
MODEL_TIER_PRESSURE_THRESHOLD = float(
    os.getenv('ATLANTEAN_MODEL_TIER_PRESSURE_THRESHOLD', '0.8')
)

# Step 29 hardware scaling tier.
HARDWARE_SCALING_TIER = (os.getenv('ATLANTEAN_HARDWARE_TIER') or 'medium').strip().lower()
ACTIVE_HARDWARE_TIER = get_tier_or_default(HARDWARE_SCALING_TIER, default='medium')

MODEL_TIER_SELECTOR = ModelTierSelector(
    config=build_model_tier_config(
        light_model=MODEL_TIER_LIGHT,
        standard_model=MODEL_TIER_STANDARD,
        heavy_model=MODEL_TIER_HEAVY,
        resource_pressure_threshold=MODEL_TIER_PRESSURE_THRESHOLD,
    )
)

PROVIDER_CAPABILITIES = build_provider_capability_registry(
    local_backend=LOCAL_LLM_BACKEND,
    local_model=LOCAL_LLM_MODEL,
    local_context_window=LOCAL_CONTEXT_WINDOW,
)

# Runtime security — authentication and network binding.
# When ATLANTEAN_API_TOKEN is set every non-health API request must carry
# "Authorization: Bearer <token>".  Leave unset for loopback-only dev setups.
ATLANTEAN_API_TOKEN: str = (os.getenv('ATLANTEAN_API_TOKEN') or '').strip()
# Bind address: 127.0.0.1 for Vault (loopback-only), 0.0.0.0 for Docker/LAN.
ATLANTEAN_BIND_HOST: str = (os.getenv('ATLANTEAN_BIND_HOST') or '127.0.0.1').strip()
# Use ATLANTEAN_PORT only — PORT is consumed by server.js and must not be shared.
ATLANTEAN_BIND_PORT: int = int(os.getenv('ATLANTEAN_PORT') or '5001')

if ATLANTEAN_BIND_HOST not in {'127.0.0.1', 'localhost', '::1'} and not ATLANTEAN_API_TOKEN:
    raise RuntimeError(
        'ATLANTEAN_API_TOKEN must be set when binding the backend to a non-loopback address. '
        'This prevents unauthenticated access on exposed interfaces.'
    )

# Per-IP rate limiting for the streaming endpoint.
_stream_rl_calls: Dict[str, list] = {}
_stream_rl_lock = __import__('threading').Lock()
if ATLANTEAN_API_TOKEN:
    print(f"✅ Bearer-token authentication enabled ({len(ATLANTEAN_API_TOKEN)} chars)")
else:
    print("⚠️  ATLANTEAN_API_TOKEN not set — API is unauthenticated (loopback-only deployments only)")
if ATLANTEAN_BIND_HOST == '127.0.0.1':
    print("🔒 Backend bound to loopback (127.0.0.1) — not accessible from LAN")
else:
    print(f"🌐 Backend bound to {ATLANTEAN_BIND_HOST} — ensure firewall rules are in place")

# Verify model file checksums at startup (warning only, does not abort).
# NOTE: must be placed AFTER PIPER_VOICE and PIPER_VOICE_DIR are defined below.

# Step 31 — Local offline voice pipeline.
WHISPER_MODEL_SIZE = os.getenv('ATLANTEAN_WHISPER_MODEL', 'base').strip()
WHISPER_DEVICE = os.getenv('ATLANTEAN_WHISPER_DEVICE', 'cpu').strip()
WHISPER_COMPUTE_TYPE = os.getenv('ATLANTEAN_WHISPER_COMPUTE_TYPE', 'int8').strip()
WHISPER_LANGUAGE: str | None = os.getenv('ATLANTEAN_WHISPER_LANGUAGE') or None
PIPER_VOICE = os.getenv('ATLANTEAN_PIPER_VOICE', 'en_US-lessac-medium').strip()
PIPER_VOICE_DIR = os.getenv('ATLANTEAN_PIPER_VOICE_DIR', './voices').strip()

# Now that PIPER_VOICE / PIPER_VOICE_DIR are defined, run the checksum probe.
verify_model_files_on_startup(piper_voice_dir=PIPER_VOICE_DIR, piper_voice=PIPER_VOICE)

_local_transcriber: LocalWhisperTranscriber | None = None
_local_tts: LocalPiperTts | None = None
_voice_lock = __import__('threading').Lock()


def _get_transcriber() -> LocalWhisperTranscriber:
    global _local_transcriber
    with _voice_lock:
        if _local_transcriber is None:
            _local_transcriber = LocalWhisperTranscriber(
                model_size=WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                language=WHISPER_LANGUAGE,
            )
    return _local_transcriber


def _get_piper() -> LocalPiperTts:
    global _local_tts
    with _voice_lock:
        if _local_tts is None:
            _local_tts = LocalPiperTts(
                voice=PIPER_VOICE,
                voice_dir=PIPER_VOICE_DIR,
            )
    return _local_tts


def _emit_security_policy_event(event_type: str, payload: Dict[str, Any]) -> None:
    try:
        record = {
            'event': event_type,
            'ts': int(time.time() * 1000),
            **payload,
        }
        print(json.dumps(record, sort_keys=True))
        _append_telemetry_event(event_type, payload)
    except Exception:
        # Policy logging should never break inference flow.
        pass


def _local_llm_api_call(prompt: str, **kwargs) -> str:
    backend = str(kwargs.get('backend') or LOCAL_LLM_BACKEND).strip().lower()
    model = str(kwargs.get('model') or LOCAL_LLM_MODEL)
    temperature = float(kwargs.get('temperature', 0.6))
    max_tokens = int(kwargs.get('max_tokens', 1024))
    active_url = _get_local_llm_url(backend)

    outbound_policy = evaluate_local_url_policy(
        url=active_url,
        allow_outbound_network=ALLOW_OUTBOUND_NETWORK,
    )
    if not bool(outbound_policy.get('allowed', False)):
        raise RuntimeError(
            "Outbound network policy blocked the Ollama URL. "
            f"reason={outbound_policy.get('reason_code')}"
        )

    if backend == 'ollama':
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens,
            },
        }
        response = requests.post(
            active_url,
            json=payload,
            timeout=LOCAL_LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        text = str(body.get('response', '') or '').strip()
    elif backend in {'llama.cpp', 'llamacpp', 'llama_cpp'}:
        payload = {
            'prompt': prompt,
            'temperature': temperature,
            'n_predict': max_tokens,
            'stream': False,
        }
        response = requests.post(
            active_url,
            json=payload,
            timeout=LOCAL_LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        text = str(body.get('content', '') or '').strip()
    elif backend == 'vllm':
        payload = {
            'model': model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        response = requests.post(
            active_url,
            json=payload,
            timeout=LOCAL_LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get('choices', [])
        text = ''
        if isinstance(choices, list) and choices:
            text = str((choices[0] or {}).get('text', '') or '').strip()
    else:
        raise RuntimeError(
            f"Unsupported LOCAL_LLM_BACKEND '{backend}'. "
            "Use one of: ollama, llama.cpp, vllm."
        )

    if not text:
        raise RuntimeError(f'Local model server returned no response text (backend={backend})')

    return text


def _local_llm_stream(prompt: str, **kwargs):
    """Yield raw token strings from the local LLM backend (Ollama streaming)."""
    backend = str(kwargs.get('backend') or LOCAL_LLM_BACKEND).strip().lower()
    model   = str(kwargs.get('model')   or LOCAL_LLM_MODEL)
    temperature = float(kwargs.get('temperature', 0.6))
    max_tokens  = int(kwargs.get('max_tokens', 1200))
    active_url = _get_local_llm_url(backend)

    outbound_policy = evaluate_local_url_policy(
        url=active_url, allow_outbound_network=ALLOW_OUTBOUND_NETWORK,
    )
    if not bool(outbound_policy.get('allowed', False)):
        raise RuntimeError(f"Outbound policy blocked Ollama URL: {outbound_policy.get('reason_code')}")

    if backend == 'ollama':
        payload = {
            'model': model, 'prompt': prompt, 'stream': True,
            'options': {'temperature': temperature, 'num_predict': max_tokens},
        }
        with requests.post(active_url, json=payload,
                           timeout=LOCAL_LLM_TIMEOUT_SECONDS, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except Exception:
                    continue
                token = chunk.get('response', '')
                if token:
                    yield token
                if chunk.get('done'):
                    break
    elif backend in {'llama.cpp', 'llamacpp', 'llama_cpp'}:
        # llama.cpp server supports streaming via chunked JSON
        import_url = active_url.replace('/completion', '/completions') if 'completion' in active_url else active_url
        payload = {'prompt': prompt, 'temperature': temperature,
                   'n_predict': max_tokens, 'stream': True}
        with requests.post(import_url, json=payload,
                           timeout=LOCAL_LLM_TIMEOUT_SECONDS, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode('utf-8') if isinstance(raw_line, bytes) else raw_line
                if line.startswith('data:'):
                    line = line[5:].strip()
                if line in ('[DONE]', ''):
                    break
                try:
                    chunk = json.loads(line)
                    token = chunk.get('content', '') or chunk.get('token', {}).get('text', '')
                    if token:
                        yield token
                except Exception:
                    continue
    else:
        # Unsupported streaming backend — fall back to full response, yield as single token
        text = _local_llm_api_call(prompt, **kwargs)
        yield text


def _enforce_ledger_gate_or_raise(
    session_id: str,
    bridge: 'AtlanteanQuadraBridge',
    context: str,
    require_replay_match: bool,
) -> Dict[str, Any]:
    ledger_report = _verify_ledger_integrity(session_id, bridge)
    replay_proof = _deterministic_replay_proof(session_id, bridge)
    decision = evaluate_ledger_gate(
        ledger_report=ledger_report,
        replay_proof=replay_proof,
        require_replay_match=require_replay_match,
    )
    _emit_security_policy_event('ledger_verification_decision', {
        'context': context,
        'session_id': session_id,
        'allowed': bool(decision.get('allowed', False)),
        'reason_code': decision.get('reason_code'),
        'ledger_valid': bool(ledger_report.get('valid', False)),
        'replay_valid': bool(replay_proof.get('valid', False)),
        'replay_match': bool(replay_proof.get('match', False)),
    })
    if not bool(decision.get('allowed', False)):
        raise ValueError(f"Ledger verification failed ({decision.get('reason_code')})")
    return {
        'decision': decision,
        'ledger_report': ledger_report,
        'replay_proof': replay_proof,
    }

app = Flask(__name__)
SESSION_SECRET = (os.getenv('ATLANTEAN_SESSION_SECRET') or os.getenv('SECRET_KEY') or '').strip()
if not SESSION_SECRET or SESSION_SECRET == 'CHANGE_ME_TO_A_RANDOM_SECRET' or SESSION_SECRET == 'dev-only-change-me':
    SESSION_SECRET = secrets.token_hex(32)
    print('⚠️  ATLANTEAN_SESSION_SECRET missing or placeholder; generated ephemeral secret for this process')
app.config['SECRET_KEY'] = SESSION_SECRET
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', '0') == '1'

# --- CORS Configuration ---
# Restrict CORS to specific origins (development and production)
allowed_origins = os.getenv(
    'ALLOWED_ORIGINS',
    'http://localhost:5000,http://127.0.0.1:5000,http://localhost:5173,http://127.0.0.1:5173'
).split(',')
allowed_origins = [o.strip() for o in allowed_origins]

CORS(
    app,
    origins=allowed_origins,
    allow_headers=['Content-Type', 'Authorization'],
    methods=['GET', 'POST', 'OPTIONS'],
    supports_credentials=True,
    max_age=3600  # Preflight cache duration in seconds
)

# ── Bearer-token authentication (optional) ───────────────────────────────────
_UNPROTECTED_PATHS = frozenset(['/health', '/health/deep'])

@app.after_request
def _add_security_headers(response):
    """Apply standard defensive HTTP headers to every response."""
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    # Restrict permissions to only what the app uses (mic for voice, no camera).
    response.headers.setdefault('Permissions-Policy', 'camera=(), payment=(), usb=()')
    # Content-Security-Policy: allow same-origin + loopback backend connections.
    response.headers.setdefault(
        'Content-Security-Policy',
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self' http://127.0.0.1:* http://localhost:* ws://127.0.0.1:* ws://localhost:*; "
        "worker-src 'self' blob:;"
    )
    return response

@app.before_request
def _require_api_token() -> None:
    """Enforce bearer-token auth when ATLANTEAN_API_TOKEN is configured."""
    import hmac as _hmac
    if request.method == 'OPTIONS':
        return None
    if request.path in _UNPROTECTED_PATHS:
        return None
    if not ATLANTEAN_API_TOKEN:
        return None  # no token configured — unauthenticated mode
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized', 'hint': 'Provide Authorization: Bearer <token>'}), 401
    provided = auth[len('Bearer '):]
    if not _hmac.compare_digest(provided.encode(), ATLANTEAN_API_TOKEN.encode()):
        return jsonify({'error': 'Unauthorized'}), 401
    return None

# Try to connect to Redis, fallback to in-memory storage if unavailable
try:
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_client = redis.Redis(host=redis_host, port=6379, socket_connect_timeout=2)
    redis_client.ping()  # Test connection
    app.config['SESSION_TYPE'] = 'redis'
    app.config['SESSION_REDIS'] = redis_client
    Session(app)
    print(f"✅ Redis connected on {redis_host}:6379")
    REDIS_AVAILABLE = True
except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
    print(f"⚠️  Redis unavailable ({type(e).__name__}) - using in-memory storage")
    # flask-session's built-in 'filesystem' interface is deprecated and scheduled
    # for removal; use its supported CacheLib backend for the on-disk fallback.
    from cachelib import FileSystemCache
    app.config['SESSION_TYPE'] = 'cachelib'
    app.config['SESSION_CACHELIB'] = FileSystemCache(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flask_session_cache'))
    Session(app)
    REDIS_AVAILABLE = False
    
    # In-memory fallback for Redis operations
    class InMemoryRedis:
        def __init__(self):
            self.data = {}
            self.lists = {}
        
        def get(self, key):
            return self.data.get(key, None)
        
        def set(self, key, value):
            self.data[key] = value
        
        def incr(self, key):
            current = int(self.data.get(key, 0))
            self.data[key] = str(current + 1)
            return current + 1
        
        def rpush(self, key, value):
            if key not in self.lists:
                self.lists[key] = []
            self.lists[key].append(value)

        def lpush(self, key, value):
            if key not in self.lists:
                self.lists[key] = []
            self.lists[key].insert(0, value)
        
        def lrange(self, key, start, end):
            if key not in self.lists:
                return []
            items = self.lists[key]
            if end == -1:
                return items[start:]
            return items[start:end+1]

        def ltrim(self, key, start, end):
            if key not in self.lists:
                return
            items = self.lists[key]
            if end == -1:
                self.lists[key] = items[start:]
            else:
                self.lists[key] = items[start:end+1]
        
        def delete(self, key):
            if key in self.data:
                del self.data[key]
            if key in self.lists:
                del self.lists[key]
    
    redis_client = InMemoryRedis()

# In-process bridge cache
bridges = {}
hrm_adapters = {}

HOT_MEMORY_KEY_PREFIX = 'hot_memory:'
SNAPSHOT_INDEX_PREFIX = 'snapshot_index:'
HRM_STATE_KEY_PREFIX = 'hrm_state:'
EVENT_LOG_KEY_PREFIX = 'event_log:'
EVENT_SEQ_KEY_PREFIX = 'event_seq:'
EVENT_HEAD_HASH_PREFIX = 'event_head_hash:'
CHECKPOINT_INDEX_PREFIX = 'checkpoint_index:'
CHECKPOINT_INTERVAL = 20
SNAPSHOT_DATA_PREFIX = 'snapshot_data:'
TELEMETRY_LOG_KEY_PREFIX = 'telemetry_log:'


def _append_telemetry_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Persist security/runtime telemetry with bounded retention."""
    try:
        session_id = str(payload.get('session_id') or 'global')
        telemetry_key = f'{TELEMETRY_LOG_KEY_PREFIX}{session_id}'
        telemetry_record = {
            'event': event_type,
            'ts': int(time.time() * 1000),
            'payload': payload,
        }
        redis_client.lpush(telemetry_key, _canonical_json(telemetry_record))
        if ENABLE_MEMORY_RETENTION_CONTROLS and RETENTION_TELEMETRY_EVENTS_MAX >= 0:
            if RETENTION_TELEMETRY_EVENTS_MAX == 0:
                redis_client.delete(telemetry_key)
            else:
                redis_client.ltrim(telemetry_key, 0, RETENTION_TELEMETRY_EVENTS_MAX - 1)
    except Exception:
        # Telemetry persistence is best-effort only.
        pass


def _load_cold_items(bridge: 'AtlanteanQuadraBridge') -> List[Dict[str, Any]]:
    """Load all cold-memory items from Redis for retention planning."""
    items: List[Dict[str, Any]] = []
    try:
        redis_conn = bridge.cold_memory.redis
        item_keys = redis_conn.keys('item:*')
    except Exception:
        # Redis unavailable — skip retention planning, items stay in memory
        return items
    for key in item_keys:
        key_text = key.decode('utf-8') if isinstance(key, bytes) else str(key)
        item_id = key_text.split(':', 1)[1] if ':' in key_text else key_text
        try:
            if hasattr(bridge.cold_memory, '_read_json'):
                item_dict = bridge.cold_memory._read_json(key_text)
            else:
                raw = redis_conn.get(key)
                decoded = raw.decode('utf-8') if isinstance(raw, bytes) else raw
                item_dict = json.loads(decoded) if decoded else None
            if isinstance(item_dict, dict):
                items.append({
                    'id': item_dict.get('id', item_id),
                    'metadata': item_dict.get('metadata', {}),
                })
        except Exception:
            continue
    return items


def _enforce_memory_retention(session_id: str, bridge: 'AtlanteanQuadraBridge') -> Dict[str, Any]:
    """Apply memory-class retention controls for chat and simulation artifacts."""
    if not ENABLE_MEMORY_RETENTION_CONTROLS:
        return {
            'enabled': False,
            'chat_pruned': 0,
            'simulation_pruned': 0,
        }

    items = _load_cold_items(bridge)
    chat_prune_ids = plan_item_retention(
        items,
        item_type='chat_turn',
        session_id=session_id,
        keep_max=RETENTION_CHAT_TURNS_MAX,
    )
    simulation_prune_ids = plan_item_retention(
        items,
        item_type='simulation',
        session_id=session_id,
        keep_max=RETENTION_SIMULATIONS_MAX,
    )

    for item_id in chat_prune_ids:
        bridge.tombstone_cold_item(item_id, reason='retention_chat_cap')
    for item_id in simulation_prune_ids:
        bridge.tombstone_cold_item(item_id, reason='retention_simulation_cap')

    if chat_prune_ids or simulation_prune_ids:
        _emit_security_policy_event('memory_retention_enforced', {
            'session_id': session_id,
            'chat_pruned': len(chat_prune_ids),
            'simulation_pruned': len(simulation_prune_ids),
        })

    return {
        'enabled': True,
        'chat_pruned': len(chat_prune_ids),
        'simulation_pruned': len(simulation_prune_ids),
    }


def _run_query_reasoning_cycles(
    *,
    session_id: str,
    bridge: 'AtlanteanQuadraBridge',
    query: str,
    context_facts: List[str],
    budget_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute bounded symbolic reasoning cycles based on query-class budget."""
    summary = {
        'enabled': bool(budget_plan.get('enabled', False)),
        'query_class': str(budget_plan.get('query_class', 'simple')),
        'max_cycles': int(budget_plan.get('max_cycles', 0)),
        'cycles_attempted': 0,
        'cycles_ran': 0,
        'stopped_reason': 'disabled',
        'assertions': [],
        'decisions': [],
    }

    if not summary['enabled'] or summary['max_cycles'] <= 0:
        return summary

    try:
        bridge.scheduler.tick()
    except Exception:
        summary['stopped_reason'] = 'scheduler_unavailable'
        return summary

    halt_reasons = {
        'symbolic-disabled-by-policy',
        'local-call-compute-limit',
        'single-call-compute-limit',
        'invocation-rate-limit',
        'window-compute-budget-exceeded',
    }

    for cycle in range(summary['max_cycles']):
        summary['cycles_attempted'] += 1
        decision = bridge.scheduler.run_symbolic_if_allowed(
            reasoner=bridge.symbolic_reasoner,
            governance=bridge.governance,
            query=query,
            context_facts=context_facts,
            force=bool(budget_plan.get('force_symbolic', False)),
        )

        decision_row = {
            'cycle': cycle + 1,
            'ran': bool(decision.get('ran', False)),
            'reason': decision.get('reason'),
            'elapsed_ms': decision.get('elapsed_ms'),
            'estimated_cost_ms': decision.get('estimated_cost_ms'),
        }
        summary['decisions'].append(decision_row)

        _emit_security_policy_event(
            'query_lifecycle_symbolic_cycle_trigger',
            build_cycle_trigger_payload(
                session_id=session_id,
                query_class=summary['query_class'],
                cycle=cycle + 1,
                decision=decision,
            ),
        )

        if bool(decision.get('ran', False)):
            summary['cycles_ran'] += 1
            result = decision.get('result', {})
            for assertion in result.get('assertions', [])[:2]:
                text = str(assertion or '').strip()
                if text and text not in summary['assertions']:
                    summary['assertions'].append(text)
            summary['stopped_reason'] = 'cycle_budget_exhausted'
            continue

        reason = str(decision.get('reason', 'not-run'))
        summary['stopped_reason'] = reason
        if reason in halt_reasons:
            break

    _emit_security_policy_event('query_reasoning_budget', {
        'session_id': session_id,
        'query_class': summary['query_class'],
        'max_cycles': summary['max_cycles'],
        'cycles_attempted': summary['cycles_attempted'],
        'cycles_ran': summary['cycles_ran'],
        'stopped_reason': summary['stopped_reason'],
    })
    _emit_security_policy_event(
        'query_lifecycle_reasoning_summary',
        build_query_lifecycle_payload(
            session_id=session_id,
            stage='reasoning_complete',
            query_class=summary['query_class'],
            reasoning_budget=summary,
            provider_selected='pending',
        ),
    )

    return summary

# Guardrail patterns for suppressing anthropomorphic/internal status claims in
# user-facing model output and recalled assistant memory snippets.
_INTERNAL_STATUS_PATTERNS = [
    re.compile(r'\bexternal data streams?\b', re.IGNORECASE),
    re.compile(r'\bcore predictive processing resources?\b', re.IGNORECASE),
    re.compile(r'\binternal state\b', re.IGNORECASE),
    re.compile(r'\bthis anomaly prevents me\b', re.IGNORECASE),
    re.compile(r'\bpersistent disruption\b', re.IGNORECASE),
    re.compile(r'\b(i am|i\'m)\s+(currently\s+)?(experiencing|having)\b.{0,90}\b(disruption|outage|anomaly)\b', re.IGNORECASE),
]


def _contains_internal_status_language(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in _INTERNAL_STATUS_PATTERNS)


def _sentinel_stabilize(text: str, session_id: str) -> tuple[str, dict]:
    """Pass LLM output through Sentinel for drift detection/stabilization.

    Returns (possibly stabilized text, diagnostics dict).
    Gracefully returns original text if Sentinel is disabled or unreachable.
    """
    diagnostics = {'sentinel_used': False}
    if not SENTINEL_ENABLED or not text:
        return text, diagnostics

    try:
        import uuid
        import requests

        # Build a simple 24-point metric window from text features.
        # Sentinel expects 24 floats; we derive basic signal characteristics.
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?') + 1
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        # Create a 24-point window with normalized text features
        metric_window = [
            float(len(words) % 100) / 100.0,
            float(sentences % 50) / 50.0,
            min(avg_word_len / 10.0, 1.0),
        ] * 8  # 24 points

        resp = requests.post(
            f"{SENTINEL_URL}/v1/stabilize",
            json={
                'tenant_id': session_id[:32] if session_id else 'default',
                'request_id': str(uuid.uuid4()),
                'text': text,
                'metric_window': metric_window,
            },
            headers={'X-API-Key': SENTINEL_API_KEY, 'Content-Type': 'application/json'},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            diagnostics = {
                'sentinel_used': True,
                'drift_score': data.get('drift_score'),
                'risk_level': data.get('risk_level'),
                'actions': data.get('actions', []),
            }
            # Use stabilized text if provided, otherwise original
            stabilized = data.get('stabilized_text') or data.get('corrected_text')
            if stabilized:
                return stabilized, diagnostics
        else:
            diagnostics['error'] = f"HTTP {resp.status_code}"
    except Exception as e:
        diagnostics['error'] = str(e)[:100]

    return text, diagnostics


def _sanitize_model_output(text: str, user_input: str) -> str:
    """Remove internal-status disclaimers; keep useful content when present."""
    if not text:
        return text

    cleaned_lines: List[str] = []
    removed_any = False
    for line in text.splitlines():
        if _contains_internal_status_language(line):
            removed_any = True
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    if removed_any and len(cleaned) >= 40:
        return cleaned

    if _contains_internal_status_language(text) or not cleaned:
        return (
            "I can help with this directly. "
            "Please share the exact question, data, or goal you want analyzed, "
            "and I will provide a concrete answer."
        )

    return text


def _normalize_disconnect_reason(value: Any) -> str:
    return normalize_disconnect_reason(value)


def _normalize_learning_event_payload(event: str, event_data: Any) -> Dict[str, Any]:
    return normalize_learning_event_payload(event, event_data, QuadraLearningEvent.VOICE_SESSION_END)


def _canonical_json(data: Any) -> str:
    """Canonical JSON representation for deterministic hashing/signing."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _compute_hot_state_hash(bridge: 'AtlanteanQuadraBridge') -> str:
    """Compute deterministic hash of the authoritative hot-memory state."""
    state = {
        'phi1': bridge.hot_memory.phi1.tolist(),
        'phi5': bridge.hot_memory.phi5.tolist(),
        'Phi': bridge.hot_memory.Phi.tolist(),
        'Theta': bridge.hot_memory.Theta,
        'version': int(bridge.hot_memory.version),
        'device_id': bridge.hot_memory.device_id,
        'identity_fingerprint': bridge.hot_memory.identity_fingerprint,
    }
    return _sha256_hex(_canonical_json(state))


def _sign_event_hash_hex(bridge: 'AtlanteanQuadraBridge', event_hash: str) -> str | None:
    """Sign event hash with the active local identity when available."""
    if not bridge.identity:
        return None
    try:
        return bridge.identity.sign(event_hash.encode('utf-8')).hex()
    except Exception:
        return None


def _verify_event_signature(bridge: 'AtlanteanQuadraBridge', event_hash: str, signature_hex: str | None) -> bool:
    """Verify event signature when identity and signature are available."""
    if not bridge.identity:
        return signature_hex is None
    if not signature_hex:
        return False
    try:
        return bridge.identity.verify(event_hash.encode('utf-8'), bytes.fromhex(signature_hex))
    except Exception:
        return False


def _compute_merkle_root(event_hashes: List[str]) -> str:
    """Compute a simple merkle root over event hashes."""
    if not event_hashes:
        return _sha256_hex('')
    level = event_hashes[:]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level: List[str] = []
        for i in range(0, len(level), 2):
            next_level.append(_sha256_hex(level[i] + level[i + 1]))
        level = next_level
    return level[0]


def _append_signed_event(
    session_id: str,
    bridge: 'AtlanteanQuadraBridge',
    event_type: str,
    payload: Dict[str, Any],
    hrm_snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Append an immutable, hash-linked, signed event to the session ledger."""
    seq = int(redis_client.incr(f'{EVENT_SEQ_KEY_PREFIX}{session_id}'))
    prev_hash_raw = redis_client.get(f'{EVENT_HEAD_HASH_PREFIX}{session_id}')
    prev_hash = prev_hash_raw.decode('utf-8') if isinstance(prev_hash_raw, bytes) else (prev_hash_raw or '')

    unsigned_event = {
        'schema_version': 1,
        'session_id': session_id,
        'seq': seq,
        'timestamp': int(time.time() * 1000),
        'event_type': event_type,
        'payload': payload,
        'prev_hash': prev_hash,
        'hot_version': int(bridge.hot_memory.version),
        'state_hash': _compute_hot_state_hash(bridge),
        'hrm': hrm_snapshot or {},
        'identity_fingerprint': bridge.identity.fingerprint() if bridge.identity else None,
    }
    event_hash = _sha256_hex(_canonical_json(unsigned_event))
    signature_hex = _sign_event_hash_hex(bridge, event_hash)

    event = {
        **unsigned_event,
        'event_hash': event_hash,
        'signature': signature_hex,
    }
    redis_client.rpush(f'{EVENT_LOG_KEY_PREFIX}{session_id}', _canonical_json(event))
    redis_client.set(f'{EVENT_HEAD_HASH_PREFIX}{session_id}', event_hash)

    if seq % CHECKPOINT_INTERVAL == 0:
        _create_signed_checkpoint(
            session_id=session_id,
            bridge=bridge,
            label=f'Auto checkpoint #{seq}',
            source='auto_interval',
            hrm_snapshot=hrm_snapshot or {},
        )

    return event


def _create_signed_checkpoint(
    session_id: str,
    bridge: 'AtlanteanQuadraBridge',
    label: str | None,
    source: str,
    hrm_snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a signed checkpoint record bound to the current event log state."""
    events_raw = redis_client.lrange(f'{EVENT_LOG_KEY_PREFIX}{session_id}', 0, -1)
    event_hashes: List[str] = []
    for row in events_raw:
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            event_hashes.append(json.loads(decoded).get('event_hash', ''))
        except Exception:
            continue
    merkle_root = _compute_merkle_root([h for h in event_hashes if h])

    snapshot_payload = bridge.create_snapshot(label)
    checkpoint_body = {
        'schema_version': 1,
        'id': str(uuid.uuid4()),
        'session_id': session_id,
        'label': snapshot_payload.get('label'),
        'source': source,
        'created_at': int(time.time() * 1000),
        'seq': int(redis_client.get(f'{EVENT_SEQ_KEY_PREFIX}{session_id}') or 0),
        'state_hash': _compute_hot_state_hash(bridge),
        'merkle_root': merkle_root,
        'snapshot': snapshot_payload,
        'hrm': hrm_snapshot or {},
        'identity_fingerprint': bridge.identity.fingerprint() if bridge.identity else None,
    }
    checkpoint_hash = _sha256_hex(_canonical_json(checkpoint_body))
    checkpoint_sig = _sign_event_hash_hex(bridge, checkpoint_hash)
    checkpoint = {
        **checkpoint_body,
        'checkpoint_hash': checkpoint_hash,
        'signature': checkpoint_sig,
    }
    redis_client.lpush(f'{CHECKPOINT_INDEX_PREFIX}{session_id}', _canonical_json(checkpoint))
    redis_client.ltrim(f'{CHECKPOINT_INDEX_PREFIX}{session_id}', 0, 199)
    return checkpoint


def _verify_ledger_integrity(session_id: str, bridge: 'AtlanteanQuadraBridge') -> Dict[str, Any]:
    """Verify hash chain, event hashes, and signatures for a session ledger."""
    rows = redis_client.lrange(f'{EVENT_LOG_KEY_PREFIX}{session_id}', 0, -1)
    prev = ''
    issues: List[Dict[str, Any]] = []
    verified = 0

    for idx, row in enumerate(rows):
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            event = json.loads(decoded)
        except Exception:
            issues.append({'index': idx, 'type': 'decode_error'})
            continue

        event_hash = event.get('event_hash', '')
        event_sig = event.get('signature')

        unsigned = dict(event)
        unsigned.pop('event_hash', None)
        unsigned.pop('signature', None)
        recomputed = _sha256_hex(_canonical_json(unsigned))

        if event.get('prev_hash', '') != prev:
            issues.append({'index': idx, 'type': 'chain_mismatch'})
        if event_hash != recomputed:
            issues.append({'index': idx, 'type': 'hash_mismatch'})
        if not _verify_event_signature(bridge, event_hash, event_sig):
            issues.append({'index': idx, 'type': 'signature_invalid'})

        prev = event_hash
        verified += 1

    checkpoints_rows = redis_client.lrange(f'{CHECKPOINT_INDEX_PREFIX}{session_id}', 0, -1)
    checkpoint_count = len(checkpoints_rows)
    return {
        'valid': len(issues) == 0,
        'session_id': session_id,
        'events_verified': verified,
        'checkpoint_count': checkpoint_count,
        'issues': issues,
        'head_hash': prev,
    }


def _deterministic_replay_proof(session_id: str, bridge: 'AtlanteanQuadraBridge') -> Dict[str, Any]:
    """
    Build a one-shot deterministic replay proof from the signed event log.

    Replay model:
    - Recompute each event hash from canonical unsigned payload.
    - Validate hash-chain linkage and signatures.
    - Use the last verified event state_hash as replay-derived expected state hash.
    - Compare with current live hot-memory state hash.
    """
    rows = redis_client.lrange(f'{EVENT_LOG_KEY_PREFIX}{session_id}', 0, -1)
    prev_hash = ''
    issues: List[Dict[str, Any]] = []
    verified_events: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            event = json.loads(decoded)
        except Exception:
            issues.append({'index': idx, 'type': 'decode_error'})
            continue

        unsigned = dict(event)
        event_hash = unsigned.pop('event_hash', '')
        signature_hex = unsigned.pop('signature', None)
        recomputed = _sha256_hex(_canonical_json(unsigned))

        event_issues = []
        if unsigned.get('prev_hash', '') != prev_hash:
            event_issues.append('chain_mismatch')
        if event_hash != recomputed:
            event_issues.append('hash_mismatch')
        if not _verify_event_signature(bridge, event_hash, signature_hex):
            event_issues.append('signature_invalid')

        if event_issues:
            issues.append({'index': idx, 'type': 'event_invalid', 'details': event_issues})
            break

        verified_events.append(event)
        prev_hash = event_hash

    replay_state_hash = verified_events[-1].get('state_hash') if verified_events else None
    live_state_hash = _compute_hot_state_hash(bridge)
    stored_head_raw = redis_client.get(f'{EVENT_HEAD_HASH_PREFIX}{session_id}')
    stored_head_hash = stored_head_raw.decode('utf-8') if isinstance(stored_head_raw, bytes) else (stored_head_raw or '')
    head_consistent = verify_replay_head(prev_hash, stored_head_hash)
    if not head_consistent:
        issues.append({
            'type': 'head_hash_mismatch',
            'replay_head_hash': prev_hash,
            'stored_head_hash': stored_head_hash,
        })

    return {
        'session_id': session_id,
        'events_total': len(rows),
        'events_verified': len(verified_events),
        'replay_state_hash': replay_state_hash,
        'live_state_hash': live_state_hash,
        'match': replay_state_hash == live_state_hash if replay_state_hash else False,
        'valid': len(issues) == 0,
        'issues': issues,
        'replay_head_hash': prev_hash,
        'stored_head_hash': stored_head_hash,
        'head_consistent': head_consistent,
        'verified_up_to_seq': verified_events[-1].get('seq') if verified_events else None,
    }


def _apply_hrm_field_coupling(bridge: 'AtlanteanQuadraBridge', hrm_snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Delegate to the bridge's own isomorphic coupling method.

    Stores the snapshot on the bridge first so that any subsequent internal
    hot-memory update (e.g. inside recall_chat_memory / recall_simulations)
    also uses the current HRM topology, not stale defaults.
    """
    bridge.set_hrm_snapshot(hrm_snapshot)
    return bridge._apply_hrm_field_coupling()


def _save_bridge_to_redis(session_id: str, bridge: 'AtlanteanQuadraBridge'):
    """Persist hot memory state to Redis (backend-persistent, not per-file)."""
    try:
        buf = io.BytesIO()
        if bridge.identity:
            bridge.hot_memory.save(buf, identity=bridge.identity)
        else:
            bridge.hot_memory.save(buf)
        redis_client.set(f'{HOT_MEMORY_KEY_PREFIX}{session_id}', buf.getvalue())
    except Exception as e:
        print(f"⚠️  Failed to persist hot memory to Redis for {session_id}: {e}")


def _load_bridge_from_redis(session_id: str, bridge: 'AtlanteanQuadraBridge') -> bool:
    """Load hot memory state from Redis. Returns True if state was found."""
    try:
        data = redis_client.get(f'{HOT_MEMORY_KEY_PREFIX}{session_id}')
        if data is None:
            return False
        buf = io.BytesIO(data)
        try:
            AtlanteanHotMemory = import_module('atlantean_core.hot_memory').AtlanteanHotMemory
        except Exception:
            AtlanteanHotMemory = import_module('hot_memory').AtlanteanHotMemory
        verify = bridge.identity if bridge.identity else None
        bridge.hot_memory = AtlanteanHotMemory.load(buf, verify_identity=verify)
        bridge.bridge.hot = bridge.hot_memory
        return True
    except Exception as e:
        print(f"⚠️  Failed to load hot memory from Redis for {session_id}: {e}")
        return False


def _save_hrm_to_redis(session_id: str, adapter: HRMAdapter):
    """Persist HRM state so phase/topology continue across requests."""
    try:
        s = adapter.state
        payload = {
            't': float(s.t),
            'theta': float(s.theta),
            'channel': int(s.channel),
            'domain': int(s.domain),
            'layer': int(s.layer),
            'phi': s.phi.tolist(),
            'S': s.S.tolist(),
            'guna': s.guna.tolist(),
            'energy': float(s.energy),
            'coherence': float(s.coherence),
        }
        redis_client.set(f'{HRM_STATE_KEY_PREFIX}{session_id}', json.dumps(payload))
    except Exception as e:
        print(f"⚠️  Failed to persist HRM state for {session_id}: {e}")


def _serialize_hrm_state(adapter: HRMAdapter) -> Dict[str, Any]:
    """Serialize full HRM state for deterministic snapshot restore."""
    s = adapter.state
    return {
        't': float(s.t),
        'theta': float(s.theta),
        'channel': int(s.channel),
        'domain': int(s.domain),
        'layer': int(s.layer),
        'phi': s.phi.tolist(),
        'S': s.S.tolist(),
        'guna': s.guna.tolist(),
        'energy': float(s.energy),
        'coherence': float(s.coherence),
    }


def _restore_hrm_state_from_snapshot(adapter: HRMAdapter, snapshot: Dict[str, Any]) -> bool:
    """Restore HRM adapter state from snapshot payload. Returns False on legacy/incomplete payload."""
    required = ('phi', 'S', 'guna', 'theta', 'channel', 'domain', 'layer', 't')
    if not all(key in snapshot for key in required):
        return False

    adapter.state = HRMState(
        t=float(snapshot['t']),
        theta=float(snapshot['theta']),
        channel=int(snapshot['channel']),
        domain=int(snapshot['domain']),
        layer=int(snapshot['layer']),
        phi=np.array(snapshot['phi'], dtype=float),
        S=np.array(snapshot['S'], dtype=float),
        guna=np.array(snapshot['guna'], dtype=float),
        energy=float(snapshot.get('energy', 0.0)),
        coherence=float(snapshot.get('coherence', 0.0)),
    )
    return True


def _load_hrm_from_redis(session_id: str, adapter: HRMAdapter) -> bool:
    """Load HRM state from Redis if available."""
    try:
        raw = redis_client.get(f'{HRM_STATE_KEY_PREFIX}{session_id}')
        if not raw:
            return False
        data = json.loads(raw)
        adapter.state = HRMState(
            t=float(data['t']),
            theta=float(data['theta']),
            channel=int(data['channel']),
            domain=int(data['domain']),
            layer=int(data['layer']),
            phi=np.array(data['phi'], dtype=float),
            S=np.array(data['S'], dtype=float),
            guna=np.array(data['guna'], dtype=float),
            energy=float(data.get('energy', 0.0)),
            coherence=float(data.get('coherence', 0.0)),
        )
        return True
    except Exception as e:
        print(f"⚠️  Failed to load HRM state for {session_id}: {e}")
        return False


def get_bridge_for_session(session_id: str) -> 'AtlanteanQuadraBridge':
    """Get or create the bridge instance for a session, backed by Redis."""
    if session_id not in bridges:
        bridges[session_id] = AtlanteanQuadraBridge(
            grid_size=(32, 32),
            device_id=f"quadra-seer-{session_id}"
        )
        if _load_bridge_from_redis(session_id, bridges[session_id]):
            print(f"✅ Loaded hot memory from Redis for session {session_id}")
        else:
            print(f"✨ Initialized new intelligence state for session {session_id}")
    return bridges[session_id]


def get_hrm_for_session(session_id: str) -> HRMAdapter:
    """Get or create per-session HRM adapter with persisted state."""
    if session_id not in hrm_adapters:
        hrm_adapters[session_id] = HRMAdapter()
        if _load_hrm_from_redis(session_id, hrm_adapters[session_id]):
            print(f"✅ Loaded HRM state from Redis for session {session_id}")
        else:
            print(f"✨ Initialized new HRM state for session {session_id}")
    return hrm_adapters[session_id]


def get_bridge() -> 'AtlanteanQuadraBridge':
    """Get the default (global) bridge instance."""
    return get_bridge_for_session('default')


def _resolve_session_id(explicit_session_id: str | None = None) -> str:
    """Resolve a server-authoritative session id.

    If a stable session id is already bound in the cookie session, enforce it.
    If no cookie session exists yet, allow a client-provided session id and bind
    it to the server session so stateless clients can remain consistent.
    """
    explicit = str(explicit_session_id).strip() if explicit_session_id else None

    stable = session.get('atlantean_session_id')
    if stable:
        if explicit and explicit != stable:
            print("⚠️  Ignoring client-provided session_id override")
        return str(stable)

    sid = explicit or str(getattr(session, 'sid', None) or uuid.uuid4())
    session['atlantean_session_id'] = sid
    session.modified = True
    return sid


def _index_phase_snapshot(
    session_id: str,
    bridge: 'AtlanteanQuadraBridge',
    label: str | None = None,
    hrm_snapshot: dict | None = None,
    source: str = 'manual',
) -> dict:
    """Create and index a phase-locked snapshot record in Redis."""
    snapshot = bridge.create_snapshot(label)
    snapshot_id = str(uuid.uuid4())
    snapshot_data_key = f'{SNAPSHOT_DATA_PREFIX}{session_id}:{snapshot_id}'
    snapshot_record = {
        'id': snapshot_id,
        'session_id': session_id,
        'label': snapshot.get('label'),
        'created_at': snapshot.get('created_at'),
        'version': snapshot.get('version'),
        'source': source,
        'phase_lock': {
            'phi1_mean': float(bridge.hot_memory.phi1.mean()),
            'phi5_mean': float(bridge.hot_memory.phi5.mean()),
            'Phi': float(bridge.hot_memory.Phi.item()),
        },
        'hrm': hrm_snapshot or {},
        'snapshot_data_key': snapshot_data_key,
        'snapshot_sha256': snapshot_hash(snapshot),
    }
    try:
        key = f'{SNAPSHOT_INDEX_PREFIX}{session_id}'
        redis_client.set(snapshot_data_key, json.dumps(snapshot))
        redis_client.lpush(key, json.dumps(snapshot_record))
        redis_client.ltrim(key, 0, 199)
    except Exception as e:
        print(f"⚠️  Failed to index snapshot for {session_id}: {e}")

    snapshot['id'] = snapshot_id
    snapshot['session_id'] = session_id
    snapshot['source'] = source
    if hrm_snapshot:
        snapshot['hrm'] = hrm_snapshot
    return snapshot


def _decode_json_row(row: Any) -> Dict[str, Any] | None:
    """Decode a Redis list row that stores JSON objects."""
    try:
        decoded = row.decode('utf-8') if isinstance(row, bytes) else row
        parsed = json.loads(decoded)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _load_snapshot_records(session_id: str) -> List[Dict[str, Any]]:
    """Load all snapshot records for a session from Redis."""
    key = f'{SNAPSHOT_INDEX_PREFIX}{session_id}'
    rows = redis_client.lrange(key, 0, -1)
    records: List[Dict[str, Any]] = []
    for row in rows:
        parsed = _decode_json_row(row)
        if parsed:
            records.append(parsed)
    return records


def _save_snapshot_records(session_id: str, records: List[Dict[str, Any]]) -> None:
    """Replace the snapshot list for a session with provided records."""
    key = f'{SNAPSHOT_INDEX_PREFIX}{session_id}'
    redis_client.delete(key)
    for record in records:
        redis_client.rpush(key, json.dumps(record))


# ========== Core API Endpoints ==========

@app.route('/api/atlantean/status', methods=['GET'])
def status():
    """Get current intelligence status."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    b = get_bridge_for_session(session_id)
    return jsonify(b.get_status())


@app.route('/api/atlantean/query', methods=['POST'])
def query():
    """
    Process user query through Atlantean-powered LLM.
    
    Request body:
    {
        "input": "user message",
        "llm_provider": "local" | "gemini" | "edenai" | "mock",
        "api_key": "optional_api_key_override",
        "cloud_escalation_consent": false
    }
    """
    data = request.json
    user_input = str(data.get('input', '') or '')
    llm_provider = str(data.get('llm_provider') or ATLANTEAN_LLM_PROVIDER).strip().lower()
    api_key = data.get('api_key') or GEMINI_API_KEY
    cloud_escalation_consent = bool(data.get('cloud_escalation_consent', False))
    session_id = _resolve_session_id(data.get('session_id'))
    # Step 25: operator routing controls forwarded from profile/settings.
    _operator_tier_preference = str(data.get('model_tier_preference') or '').strip().lower() or None
    _operator_backend_hint = str(data.get('local_backend_hint') or '').strip().lower() or None
    # history sent by legacy clients is accepted but ignored; context is
    # reconstructed server-side from Atlantean cold memory (see below).
    files = data.get('files', [])

    normalized_files = []
    text_exts = {
        'txt', 'md', 'csv', 'json', 'xml', 'yaml', 'yml', 'log', 'py', 'js', 'ts', 'tsx', 'jsx', 'html', 'css'
    }
    print(f"[DEBUG] Received {len(files) if isinstance(files, list) else 0} files")
    if isinstance(files, list):
        for raw in files[:8]:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get('name', 'unnamed'))
            mime_type = str(raw.get('type', 'application/octet-stream'))
            print(f"[DEBUG] File: {name}, type: {mime_type}, has_content: {bool(raw.get('content'))}, has_extractedText: {bool(raw.get('extractedText'))}")
            extracted_text = str(raw.get('extractedText', '') or '')
            content_b64 = str(raw.get('content', '') or '')

            # Fallback: decode text-like attachments when client parser did not
            # produce extractedText (e.g., plain text payloads).
            if not extracted_text and content_b64:
                lower_name = name.lower()
                ext = lower_name.rsplit('.', 1)[-1] if '.' in lower_name else ''
                is_text_like = mime_type.startswith('text/') or ext in text_exts
                if is_text_like:
                    try:
                        decoded_bytes = base64.b64decode(content_b64, validate=False)
                        extracted_text = decoded_bytes.decode('utf-8', errors='ignore')
                    except Exception:
                        extracted_text = ''
                elif ext == 'pdf' or mime_type == 'application/pdf':
                    # Server-side PDF text extraction (pypdf)
                    try:
                        import pypdf
                        pdf_bytes = base64.b64decode(content_b64, validate=False)
                        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                        extracted_text = '\n'.join(
                            (page.extract_text() or '') for page in reader.pages
                        ).strip()
                    except Exception:
                        extracted_text = ''
                elif ext in ('docx',) or mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    # Server-side DOCX text extraction (python-docx)
                    try:
                        import docx
                        docx_bytes = base64.b64decode(content_b64, validate=False)
                        doc = docx.Document(io.BytesIO(docx_bytes))
                        extracted_text = '\n'.join(
                            p.text for p in doc.paragraphs
                        ).strip()
                    except Exception:
                        extracted_text = ''

            normalized_files.append({
                'name': name,
                'type': mime_type,
                'extracted_text': extracted_text,
            })

    if not user_input.strip() and normalized_files:
        user_input = 'Analyze the attached documents and summarize key points.'

    if not user_input.strip():
        return jsonify({'error': 'No input provided'}), 400
    
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)

    # Step 23: budget-gated HRM symbolic pass.
    _hrm_allocator = HRMComputeAllocator(policy=HRM_COMPUTE_BUDGET_POLICY)
    _hrm_alloc_decision = _hrm_allocator.allocate('hrm', 'simple')
    import time as _time_mod
    _hrm_t0 = _time_mod.perf_counter()
    hrm_step = hrm.step(steps=1)
    _hrm_elapsed_ms = (_time_mod.perf_counter() - _hrm_t0) * 1000.0
    _hrm_allocator.record_elapsed('hrm', _hrm_elapsed_ms)
    hrm_compute_budget_report = _hrm_allocator.snapshot()
    _emit_security_policy_event('hrm_compute_budget_allocation', {
        'session_id': session_id,
        'pass_type': 'hrm',
        'allowed': _hrm_alloc_decision.allowed,
        'reason': _hrm_alloc_decision.reason,
        'effective_cap_ms': _hrm_alloc_decision.effective_cap_ms,
        'elapsed_ms': round(_hrm_elapsed_ms, 3),
        'budget': hrm_compute_budget_report,
    })

    hrm_snapshot = hrm_step.get('snapshot', {})
    hrm_coupling = _apply_hrm_field_coupling(b, hrm_snapshot)
    
    # Build context from hot memory
    status = b.get_status()
    field_stats = status['field_stats']
    learning_capacity = status['learning_capacity']

    # Recall cross-chat/domain memory from cold storage and feed it into generation.
    recalled_memories = b.recall_chat_memory(
        query=user_input,
        limit=8,
        session_id=session_id,
        include_simulations=True,
    )
    recalled_facts = b.extract_recalled_facts(recalled_memories, limit=4)
    user_input_norm = user_input.strip().lower()
    filtered_facts = []
    seen_facts = set()
    for fact in recalled_facts:
        norm = fact.strip().lower()
        if (
            not norm
            or norm == user_input_norm
            or norm in seen_facts
            or _contains_internal_status_language(fact)
        ):
            continue
        seen_facts.add(norm)
        filtered_facts.append(fact)
    recalled_facts = filtered_facts

    query_class = classify_query_class(
        user_input,
        has_files=bool(normalized_files),
    )
    reasoning_budget_plan = build_query_reasoning_budget(
        query_class,
        enabled=ENABLE_QUERY_CLASS_REASONING_BUDGETS,
        cycles_by_class={
            'simple': REASONING_CYCLES_SIMPLE,
            'analytical': REASONING_CYCLES_ANALYTICAL,
            'deep': REASONING_CYCLES_DEEP,
            'file_assisted': REASONING_CYCLES_FILE_ASSISTED,
        },
    )
    reasoning_budget_report = _run_query_reasoning_cycles(
        session_id=session_id,
        bridge=b,
        query=user_input,
        context_facts=recalled_facts,
        budget_plan=reasoning_budget_plan,
    )

    _emit_security_policy_event(
        'query_lifecycle_start',
        build_query_lifecycle_payload(
            session_id=session_id,
            stage='start',
            query_class=query_class,
            reasoning_budget=reasoning_budget_report,
            provider_selected='pending',
        ),
    )
    
    # Encode intelligence state as natural language context
    context_parts = []
    
    if field_stats['phi1_mean'] > 0.5:
        context_parts.append("Operating with high confidence and decisiveness.")
    elif field_stats['phi1_mean'] < -0.5:
        context_parts.append("In exploratory mode, considering multiple possibilities.")
    else:
        context_parts.append("Balanced state, ready to adapt.")
    
    if field_stats['phi5_mean'] > 0.5:
        context_parts.append("High learning capacity - actively adapting to new patterns.")
    elif field_stats['phi5_mean'] < 0.3:
        context_parts.append("Stable patterns established - operating from experience.")
    
    if learning_capacity > 0.7:
        context_parts.append("Strong potential for growth and adaptation.")
    elif learning_capacity < 0.3:
        context_parts.append("Patterns consolidating - reinforcing core knowledge.")
    
    intelligence_context = " ".join(context_parts)
    hrm_context = (
        f"HRM phase theta={hrm_snapshot.get('theta', 0):.3f}, "
        f"coherence={hrm_snapshot.get('coherence', 0):.3f}, "
        f"energy={hrm_snapshot.get('energy', 0):.3f}, "
        f"channel={hrm_snapshot.get('channel', 0)}, "
        f"domain={hrm_snapshot.get('domain', 0)}, "
        f"layer={hrm_snapshot.get('layer', 0)}."
    )
    
    # Build prompt once from memory/identity context. Providers remain stateless.
    recent_turns = b.list_chat_history(session_id=session_id, limit=12)
    history_lines = []
    for turn in recent_turns:
        role = str(turn.get('role', '')).lower()
        content = str(turn.get('content', '')).strip()
        if not content:
            continue
        if role == 'user':
            history_lines.append(f"User: {content}")
        else:
            history_lines.append(f"Assistant: {content}")
    conversation_context = "\n".join(history_lines)

    memory_lines = []
    memory_snippets = []
    for mem in recalled_memories:
        content = str(mem.get('content_raw', ''))
        if '\n\n[Provenance]' in content:
            content = content.split('\n\n[Provenance]', 1)[0].strip()
        if _contains_internal_status_language(content):
            continue

        snippet = content[:120] + '...' if len(content) > 120 else content
        memory_snippets.append(f"- {snippet}")

        if len(content) > 260:
            content = content[:260] + '...'
        mem_type = mem.get('type', 'memory')
        source_session = mem.get('session_id', 'unknown')
        memory_lines.append(f"[{mem_type} | session={source_session}] {content}")

    memory_section = "\n".join(memory_snippets[:3]) if memory_snippets else "- No prior recalled memory"
    memory_context = "\n".join(memory_lines)
    facts_context = "\n".join([f"- {fact}" for fact in recalled_facts])
    symbolic_assertions = list(reasoning_budget_report.get('assertions', []))
    symbolic_context = "\n".join([f"- {assertion}" for assertion in symbolic_assertions])
    if not symbolic_context:
        symbolic_context = "- No symbolic cycle assertions"

    file_lines = []
    for file in normalized_files:
        snippet = file['extracted_text'].strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + '...'
        if snippet:
            file_lines.append(
                f"- {file['name']} ({file['type']}):\n"
                f"  Extracted content:\n{snippet}"
            )
        else:
            file_lines.append(f"- {file['name']} ({file['type']}): [binary/no extracted text]")
    files_context = "\n\n".join(file_lines) if file_lines else "- No attachments"

    system_prompt = f"""You are Quadra Seer Intelligence, powered by Atlantean Core.

Your current intelligence state: {intelligence_context}

Current HRM state: {hrm_context}

You are a brilliant predictive intelligence entity, expert in data forecasting, complex systems, and technical analysis.

Respond naturally and helpfully to the user's query.
Never claim internal outages, disruptions, anomalies, data-stream failures, or inaccessible processing resources.
Never describe hidden internal system state.
If details are missing, ask a concise clarifying question and provide the best actionable guidance you can.
If an "Attached files" section is provided below, treat those file excerpts as available context from the user.
Do not ask the user to upload the same document again unless every attachment is marked as binary/no extracted text."""

    full_prompt = (
        f"{system_prompt}\n\n"
        f"Conversation so far:\n{conversation_context}\n\n"
        f"High-confidence recalled user facts:\n{facts_context}\n\n"
        f"Symbolic reasoning notes:\n{symbolic_context}\n\n"
        f"Recalled memory from prior chats/domains:\n{memory_context}\n\n"
        f"Attached files ({len(normalized_files)}):\n{files_context}\n\n"
        f"User: {user_input}\n"
        f"Assistant:"
    )

    # Call provider (stateless - context from fields/memory prompt only)
    response_text = None
    model_used = None
    generation_mode = 'mock'
    provider_resolution = resolve_provider_capability(
        llm_provider,
        PROVIDER_CAPABILITIES,
        default_provider=ATLANTEAN_LLM_PROVIDER,
    )
    provider_selected = str(provider_resolution.get('provider_selected', 'local'))
    provider_capability = provider_resolution.get('capability', {})
    cloud_policy_decision = evaluate_cloud_escalation(
        allow_cloud_escalation=ALLOW_CLOUD_ESCALATION,
        require_cloud_escalation_consent=REQUIRE_CLOUD_ESCALATION_CONSENT,
        explicit_consent=cloud_escalation_consent,
    )
    orchestration_strategy = select_orchestration_strategy(
        provider_capability,
        cloud_escalation_allowed=bool(cloud_policy_decision.get('allowed', False)),
    )
    escalation_reason_code = 'LOCAL_PROVIDER_SELECTED'
    _tier_result = None  # populated for local_direct execution path (Step 24)

    if bool(provider_resolution.get('fallback_used', False)):
        _emit_security_policy_event('provider_resolution_fallback', {
            'session_id': session_id,
            'provider_requested': llm_provider,
            'provider_selected': provider_selected,
            'reason_code': 'PROVIDER_CAPABILITY_DEFAULT_FALLBACK',
        })

    try:
        execution_path = str(orchestration_strategy.get('execution_path', 'mock_fallback'))
        if execution_path == 'local_direct':
            local_url_policy = evaluate_local_url_policy(
                url=LOCAL_LLM_URL,
                allow_outbound_network=ALLOW_OUTBOUND_NETWORK,
            )
            escalation_reason_code = str(local_url_policy.get('reason_code', 'LOCAL_PROVIDER_SELECTED'))
            _emit_security_policy_event('local_provider_resolution', {
                'session_id': session_id,
                'provider_requested': llm_provider,
                'provider_selected': provider_selected,
                'allow_outbound_network': ALLOW_OUTBOUND_NETWORK,
                'local_llm_backend': LOCAL_LLM_BACKEND,
                'local_llm_url': LOCAL_LLM_URL,
                'provider_capability': provider_capability,
                'reason_code': escalation_reason_code,
            })
            # Step 24/25: dynamic tier selection with optional operator preference.
            _tier_result = MODEL_TIER_SELECTOR.select(
                query_class,
                hrm_budget_snapshot=hrm_compute_budget_report,
                operator_preference=_operator_tier_preference,
            )
            _tier_model = _tier_result.selected_model or LOCAL_LLM_MODEL
            _emit_security_policy_event('model_tier_selected', {
                'session_id': session_id,
                **MODEL_TIER_SELECTOR.snapshot(_tier_result),
            })
            response_text = _local_llm_api_call(
                full_prompt,
                model=_tier_model,
                temperature=0.55,
                max_tokens=1200,
            )
            model_used = _tier_model
            generation_mode = 'local'
        elif execution_path == 'cloud_escalated':
            escalation_reason_code = str(cloud_policy_decision.get('reason_code', 'UNKNOWN'))
            _emit_security_policy_event('cloud_escalation_decision', {
                'session_id': session_id,
                'provider_requested': llm_provider,
                'provider_selected': provider_selected,
                'provider_capability': provider_capability,
                'cloud_escalation_consent': cloud_escalation_consent,
                'allow_cloud_escalation': ALLOW_CLOUD_ESCALATION,
                'require_cloud_escalation_consent': REQUIRE_CLOUD_ESCALATION_CONSENT,
                'allowed': bool(cloud_policy_decision.get('allowed', False)),
                'reason_code': escalation_reason_code,
            })
            if not bool(cloud_policy_decision.get('allowed', False)):
                raise RuntimeError(
                    'Cloud provider escalation was blocked by policy. '
                    f'reason={escalation_reason_code}'
                )
            if not api_key or api_key == 'PLACEHOLDER_API_KEY':
                escalation_reason_code = 'CLOUD_ESCALATION_MISSING_API_KEY'
                raise RuntimeError('Cloud provider requested but no API key is configured')

            if str(provider_capability.get('backend', '')).lower() != 'google-genai':
                escalation_reason_code = 'CLOUD_ESCALATION_UNSUPPORTED_BACKEND'
                raise RuntimeError(
                    "Selected cloud orchestration backend is unsupported for direct execution. "
                    f"backend={provider_capability.get('backend')}"
                )

            api_client = _get_genai_client(api_key)
            configured_model = os.getenv('GEMINI_MODEL') or os.getenv('ATLANTEAN_GEMINI_MODEL')
            candidate_models = [m for m in [
                configured_model,
                'gemini-2.5-flash',
                'gemini-2.0-flash',
            ] if m]

            last_model_error = None
            for model_name in candidate_models:
                try:
                    response = api_client.models.generate_content(
                        model=model_name,
                        contents=full_prompt,
                    )
                    response_text = response.text
                    model_used = model_name
                    generation_mode = 'gemini'
                    escalation_reason_code = 'CLOUD_ESCALATION_EXECUTED'
                    break
                except Exception as model_err:
                    last_model_error = model_err

            if response_text is None:
                if last_model_error:
                    escalation_reason_code = 'CLOUD_ESCALATION_MODEL_ERROR'
                    raise RuntimeError(f'Gemini model fallback exhausted: {last_model_error}')
                escalation_reason_code = 'CLOUD_ESCALATION_EMPTY_RESPONSE'
                raise RuntimeError('Gemini returned no response text')
        elif execution_path == 'cloud_policy_blocked':
            escalation_reason_code = 'CLOUD_ESCALATION_POLICY_BLOCKED'
            raise RuntimeError(
                'Cloud provider escalation was blocked by orchestration policy. '
                f"reason={cloud_policy_decision.get('reason_code', 'UNKNOWN')}"
            )
        else:
            escalation_reason_code = 'PROVIDER_MOCK_MODE'
            response_text = f"""🧠 **Atlantean Intelligence Active**

Intelligence State: {intelligence_context}

Recalled Memory ({len(recalled_memories)}):
{memory_section}

Processing your query: "{user_input}"

Provider '{provider_selected}' is currently in mock mode.

Current field stats:
- Decision field (φ₁): {field_stats['phi1_mean']:.3f}
- Learning field (φ₅): {field_stats['phi5_mean']:.3f}
- Global coherence (Φ): {field_stats['Phi']:.3f}
- Learning capacity: {learning_capacity:.1%}"""
            generation_mode = 'mock'
    
    except Exception as e:
        # Fallback to mock on error
        response_text = f"🧠 Atlantean Intelligence (Error: {str(e)})\n\nIntelligence State: {intelligence_context}\n\nQuery processed, but LLM call failed. Using fallback response."
        generation_mode = 'fallback'
        if escalation_reason_code in {'LOCAL_PROVIDER_SELECTED', 'UNKNOWN'}:
            escalation_reason_code = 'PROVIDER_ERROR_FALLBACK'

    # Build retrieval/generation confidence and integrity metadata.
    retrieval_confidence = 0.0
    if recalled_memories:
        avg_score = sum(float(m.get('score', 0.0)) for m in recalled_memories) / len(recalled_memories)
        retrieval_confidence = min(max(avg_score, 0.0), 1.0)

    generation_confidence = {
        'local': 0.78,
        'gemini': 0.82,
        'mock': 0.45,
        'fallback': 0.35,
    }.get(generation_mode, 0.4)

    overall_confidence = round((retrieval_confidence * 0.4) + (generation_confidence * 0.6), 3)

    recalled_items = []
    for mem in recalled_memories[:5]:
        snippet = str(mem.get('content_raw', ''))
        if len(snippet) > 140:
            snippet = snippet[:140] + '...'
        recalled_items.append({
            'type': mem.get('type'),
            'session_id': mem.get('session_id'),
            'domain': mem.get('domain'),
            'score': round(float(mem.get('score', 0.0)), 3),
            'timestamp': mem.get('timestamp'),
            'snippet': snippet,
        })

    provenance_payload = {
        'session_id': session_id,
        'query': user_input,
        'provider_requested': llm_provider,
        'provider_selected': provider_selected,
        'escalation_reason_code': escalation_reason_code,
        'model_used': model_used,
        'generation_mode': generation_mode,
        'recalled_items': recalled_items,
        'recalled_count': len(recalled_memories),
    }
    provenance_hash = hashlib.sha256(
        json.dumps(provenance_payload, sort_keys=True).encode('utf-8')
    ).hexdigest()

    # Keep provenance in structured API metadata only; do not leak internal
    # retrieval/generation traces into the user-visible assistant text.
    if not response_text:
        response_text = "I processed your request, but no response text was generated."
    response_text = _sanitize_model_output(response_text, user_input)

    # Sentinel: stabilize LLM output for drift before delivering to user.
    sentinel_diagnostics = {}
    try:
        response_text, sentinel_diagnostics = _sentinel_stabilize(response_text, session_id)
    except Exception:
        pass

    # Persist both sides of this interaction into cold memory.
    try:
        b.store_chat_turn(
            role='user',
            content=user_input,
            session_id=session_id,
            domain='chat',
            relevance=0.45,
        )
        if response_text:
            assistant_memory = response_text
            if len(assistant_memory) > 320:
                assistant_memory = assistant_memory[:320] + '...'
            b.store_chat_turn(
                role='assistant',
                content=assistant_memory,
                session_id=session_id,
                domain='chat',
                relevance=0.3,
            )
        retention_report = _enforce_memory_retention(session_id, b)
    except Exception as memory_err:
        retention_report = {
            'enabled': ENABLE_MEMORY_RETENTION_CONTROLS,
            'chat_pruned': 0,
            'simulation_pruned': 0,
        }
        print(f"⚠️  Failed to persist chat memory for {session_id}: {memory_err}")
    
    # Auto-save after each query
    _save_bridge_to_redis(session_id, b)
    _save_hrm_to_redis(session_id, hrm)
    auto_snapshot = _index_phase_snapshot(
        session_id=session_id,
        bridge=b,
        label=f"Auto phase lock @ t={hrm_snapshot.get('t', 0):.2f}",
        hrm_snapshot=hrm_snapshot,
        source='query_auto',
    )
    ledger_event = _append_signed_event(
        session_id=session_id,
        bridge=b,
        event_type='query_processed',
        payload={
            'query': user_input,
            'query_class': query_class,
            'generation_mode': generation_mode,
            'provider_requested': llm_provider,
            'provider_selected': provider_selected,
            'orchestration_strategy': orchestration_strategy,
            'reasoning_budget': {
                'query_class': reasoning_budget_report.get('query_class'),
                'max_cycles': reasoning_budget_report.get('max_cycles'),
                'cycles_ran': reasoning_budget_report.get('cycles_ran'),
                'stopped_reason': reasoning_budget_report.get('stopped_reason'),
            },
            'escalation_reason_code': escalation_reason_code,
            'model_used': model_used,
            'recalled_count': len(recalled_memories),
            'confidence_overall': overall_confidence,
            'hrm_coupling': hrm_coupling,
        },
        hrm_snapshot=hrm_snapshot,
    )
    
    response_payload = {
        'response': response_text,
        'status': b.get_status(),
        'intelligence_context': intelligence_context,
        'hrm': hrm_snapshot,
        'auto_snapshot': {
            'id': auto_snapshot.get('id'),
            'label': auto_snapshot.get('label'),
            'source': auto_snapshot.get('source'),
        },
        'recalled_memory_count': len(recalled_memories),
        'recalled_facts': recalled_facts,
        'session_id': session_id,
        'retrieval': {
            'count': len(recalled_memories),
            'items': recalled_items,
        },
        'generation': {
            'mode': generation_mode,
            'provider_requested': llm_provider,
            'provider_selected': provider_selected,
            'provider_capability': provider_capability,
            'orchestration_strategy': orchestration_strategy,
            'query_class': query_class,
            'model_used': model_used,
            'cloud_escalation_enabled': ALLOW_CLOUD_ESCALATION,
            'cloud_escalation_consent': cloud_escalation_consent,
            'cloud_escalation_required': REQUIRE_CLOUD_ESCALATION_CONSENT,
            'cloud_escalation_allowed': bool(cloud_policy_decision.get('allowed', False)),
            'escalation_reason_code': escalation_reason_code,
        },
        'reasoning_budget': reasoning_budget_report,
        'confidence': {
            'overall': overall_confidence,
            'retrieval': round(retrieval_confidence, 3),
            'generation': generation_confidence,
        },
        'hrm_coupling': hrm_coupling,
        'ledger': {
            'seq': ledger_event.get('seq'),
            'event_hash': ledger_event.get('event_hash'),
            'signed': bool(ledger_event.get('signature')),
        },
        'integrity': {
            'provenance_hash': provenance_hash,
            'timestamp': int(time.time() * 1000),
        },
        'retention': retention_report,
        'hrm_compute_budget': hrm_compute_budget_report,
        'model_tier': MODEL_TIER_SELECTOR.snapshot(_tier_result) if _tier_result is not None else None,
    }

    _emit_security_policy_event(
        'query_lifecycle_complete',
        build_query_lifecycle_payload(
            session_id=session_id,
            stage='complete',
            query_class=query_class,
            reasoning_budget=reasoning_budget_report,
            provider_selected=provider_selected,
        ),
    )
    return jsonify(response_payload)


@app.route('/api/atlantean/query/stream', methods=['POST'])
def query_stream():
    """Stream LLM tokens as Server-Sent Events.

    Same request body as POST /api/atlantean/query.  Emits newline-delimited
    SSE events::

        data: {"token": "Hello", "done": false}\\n\\n
        data: {"token": " world", "done": false}\\n\\n
        data: {"done": true, "response": "...", "session_id": "...", ...}\\n\\n

    Falls back to mock text if the local backend is unavailable.
    Rate-limited to ATLANTEAN_STREAM_RATE_LIMIT (default 20) requests/minute/IP.
    """
    # ── Per-IP rate limiting ─────────────────────────────────────────────────
    _limit = int(os.getenv('ATLANTEAN_STREAM_RATE_LIMIT', '20'))
    _ip = (request.headers.get('X-Forwarded-For', request.remote_addr) or '127.0.0.1').split(',')[0].strip()
    _now = time.time()
    with _stream_rl_lock:
        _window = [t for t in _stream_rl_calls.get(_ip, []) if _now - t < 60]
        if len(_window) >= _limit:
            return jsonify({
                'error': 'Rate limit exceeded. Try again in 60 seconds.',
                'retry_after_seconds': 60,
            }), 429
        _window.append(_now)
        _stream_rl_calls[_ip] = _window
    # ─────────────────────────────────────────────────────────────────────────
    from flask import Response, stream_with_context
    data = request.json or {}
    user_input = str(data.get('input', '') or '').strip()
    llm_provider = str(data.get('llm_provider') or ATLANTEAN_LLM_PROVIDER).strip().lower()
    cloud_escalation_consent = bool(data.get('cloud_escalation_consent', False))
    session_id = _resolve_session_id(data.get('session_id'))
    files = data.get('files', [])

    if not user_input:
        def _err():
            yield f"data: {_canonical_json({'error': 'No input provided', 'done': True})}\n\n"
        return Response(stream_with_context(_err()), mimetype='text/event-stream')

    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    hrm_step = hrm.step(steps=1)
    hrm_snapshot = hrm_step.get('snapshot', {})
    _apply_hrm_field_coupling(b, hrm_snapshot)

    # Minimal prompt construction (no symbolic reasoning overhead for streaming)
    recalled = b.recall_chat_memory(query=user_input, limit=4, session_id=session_id)
    facts = b.extract_recalled_facts(recalled, limit=2)
    recent = b.list_chat_history(session_id=session_id, limit=6)
    history_ctx = "\n".join(
        f"{'User' if t.get('role') == 'user' else 'Assistant'}: {t.get('content', '')}"
        for t in recent if t.get('content')
    )
    facts_ctx = "\n".join(f"- {f}" for f in facts) if facts else "- (none)"
    system_p = ("You are Quadra Seer Intelligence, powered by Atlantean Core. "
                "Respond naturally and helpfully. Never claim internal outages or system anomalies.")
    full_prompt = (f"{system_p}\n\nConversation:\n{history_ctx}\n\n"
                   f"Recalled facts:\n{facts_ctx}\n\nUser: {user_input}\nAssistant:")

    tier = MODEL_TIER_SELECTOR.select(
        'simple',
        hrm_budget_snapshot=None,
        operator_preference=None,
    )
    model_used = tier.selected_model or LOCAL_LLM_MODEL

    def _token_generator():
        full_text_parts: list = []
        try:
            for token in _local_llm_stream(full_prompt, model=model_used, temperature=0.55, max_tokens=1200):
                full_text_parts.append(token)
                yield f"data: {_canonical_json({'token': token, 'done': False})}\n\n"
        except Exception as e:
            # Emit error token so frontend can display it
            err_text = f"[Error: {e}]"
            full_text_parts.append(err_text)
            yield f"data: {_canonical_json({'token': err_text, 'done': False})}\n\n"

        full_text = ''.join(full_text_parts)

        # Persist to memory
        try:
            b.store_chat_turn(role='user', content=user_input, session_id=session_id, domain='chat', relevance=0.45)
            b.store_chat_turn(role='assistant', content=full_text[:320], session_id=session_id, domain='chat', relevance=0.3)
        except Exception:
            pass
        _save_bridge_to_redis(session_id, b)
        _save_hrm_to_redis(session_id, hrm)
        ledger_event = _append_signed_event(
            session_id=session_id, bridge=b, event_type='query_processed_stream',
            payload={'query': user_input, 'generation_mode': 'local', 'model_used': model_used},
            hrm_snapshot=hrm_snapshot,
        )
        yield f"data: {_canonical_json({'done': True, 'response': full_text, 'session_id': session_id, 'ledger': {'seq': ledger_event.get('seq'), 'event_hash': ledger_event.get('event_hash')}})}\n\n"

    return Response(
        stream_with_context(_token_generator()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/atlantean/fields', methods=['GET'])
def get_fields():
    """Get field visualization data."""
    session_id = _resolve_session_id()
    b = get_bridge_for_session(session_id)
    return jsonify(b.get_field_visualization_data())


@app.route('/api/atlantean/providers/capabilities', methods=['GET'])
def get_provider_capabilities():
    """Return provider capability registry for routing-aware orchestration."""
    requested = request.args.get('provider')
    if requested:
        resolution = resolve_provider_capability(
            requested,
            PROVIDER_CAPABILITIES,
            default_provider=ATLANTEAN_LLM_PROVIDER,
        )
        return jsonify({
            'default_provider': normalize_provider_name(ATLANTEAN_LLM_PROVIDER) or 'local',
            'requested': requested,
            'resolution': resolution,
            'providers': PROVIDER_CAPABILITIES,
        })

    return jsonify({
        'default_provider': normalize_provider_name(ATLANTEAN_LLM_PROVIDER) or 'local',
        'providers': PROVIDER_CAPABILITIES,
    })


@app.route('/api/atlantean/routing/settings', methods=['GET'])
def get_routing_settings():
    """Return active runtime routing configuration (Step 25).

    Read-only endpoint for operator dashboards and client settings panels to
    inspect the active provider, tier selector config, HRM compute budget, and
    sovereignty policy flags.
    """
    return jsonify({
        'provider': {
            'default': normalize_provider_name(ATLANTEAN_LLM_PROVIDER) or 'local',
            'allow_cloud_escalation': ALLOW_CLOUD_ESCALATION,
            'require_cloud_escalation_consent': REQUIRE_CLOUD_ESCALATION_CONSENT,
            'allow_outbound_network': ALLOW_OUTBOUND_NETWORK,
            'local_backend': LOCAL_LLM_BACKEND,
            'local_model': LOCAL_LLM_MODEL,
        },
        'model_tiers': {
            'light': MODEL_TIER_LIGHT,
            'standard': MODEL_TIER_STANDARD,
            'heavy': MODEL_TIER_HEAVY,
            'pressure_threshold': MODEL_TIER_PRESSURE_THRESHOLD,
        },
        'hrm_compute_budget': {
            'enabled': ENABLE_HRM_COMPUTE_BUDGET,
            'hrm_max_ms_per_query': HRM_COMPUTE_MAX_MS_PER_QUERY,
            'rcm_max_ms_per_query': RCM_COMPUTE_MAX_MS_PER_QUERY,
            'max_total_ms_per_window': HRM_COMPUTE_MAX_MS_PER_WINDOW,
            'window_seconds': HRM_COMPUTE_WINDOW_SECONDS,
        },
        'reasoning_budgets': {
            'enabled': ENABLE_QUERY_CLASS_REASONING_BUDGETS,
            'cycles_simple': REASONING_CYCLES_SIMPLE,
            'cycles_analytical': REASONING_CYCLES_ANALYTICAL,
            'cycles_deep': REASONING_CYCLES_DEEP,
            'cycles_file_assisted': REASONING_CYCLES_FILE_ASSISTED,
        },
    })


@app.route('/api/atlantean/hardware/scaling-matrix', methods=['GET'])
def get_hardware_scaling_matrix():
    """Return the hardware scaling matrix and the active tier (Step 29).

    Returns all three tier definitions (small_quantized, medium, multi_agent)
    plus which tier is currently active, so operators and dashboards can
    understand what compute/model/reasoning configuration is in use.
    """
    matrix = get_scaling_matrix()
    return jsonify({
        **matrix,
        'active_tier_id': ACTIVE_HARDWARE_TIER.tier_id,
        'active_tier': tier_to_dict(ACTIVE_HARDWARE_TIER),
    })


@app.route('/api/atlantean/chat/history', methods=['GET'])
def get_chat_history():
    """Get recent persisted chat turns for a session."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    limit = int(request.args.get('limit', 80))
    b = get_bridge_for_session(session_id)
    messages = b.list_chat_history(session_id=session_id, limit=limit)
    return jsonify({
        'session_id': session_id,
        'messages': messages,
    })


@app.route('/api/atlantean/learning-event', methods=['POST'])
def learning_event():
    """
    Trigger a learning event.
    
    Request body:
    {
        "session_id": "optional session id",
        "event": "user_confirmation" | "user_correction" | etc.,
        "data": { ... event-specific data ... }
    }
    """
    data = request.json
    session_id = _resolve_session_id(data.get('session_id'))
    event = data.get('event')
    event_data = _normalize_learning_event_payload(event, data.get('data', {}))
    
    if not event:
        return jsonify({'error': 'No event type provided'}), 400
    
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    
    try:
        b.on_event(event, **event_data)
        # Capture HRM snapshot AFTER event application, which modifies hot memory
        hrm_snapshot = hrm.snapshot()
        hrm_coupling = _apply_hrm_field_coupling(b, hrm_snapshot)
        _save_bridge_to_redis(session_id, b)
        _append_signed_event(
            session_id=session_id,
            bridge=b,
            event_type='learning_event',
            payload={
                'event': event,
                'data': event_data,
                'hrm_coupling': hrm_coupling,
            },
            hrm_snapshot=hrm_snapshot,
        )
        
        return jsonify({
            'success': True,
            'status': b.get_status(),
            'hrm_coupling': hrm_coupling,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== Simulation Endpoints ==========

@app.route('/api/atlantean/simulation/store', methods=['POST'])
def store_simulation():
    """Store simulation in cold memory."""
    import traceback
    try:
        data = request.json or {}
        session_id = _resolve_session_id()
        simulation = data.get('simulation')
        confidence = data.get('confidence', 0.5)
        
        if not simulation:
            return jsonify({'error': 'No simulation data'}), 400
        
        b = get_bridge_for_session(session_id)
        hrm = get_hrm_for_session(session_id)
        b.store_simulation(simulation, confidence, session_id=session_id)
        retention_report = _enforce_memory_retention(session_id, b)
        # Capture HRM snapshot AFTER store operation, which modifies hot memory
        hrm_snapshot = hrm.snapshot()
        _save_bridge_to_redis(session_id, b)
        _append_signed_event(
            session_id=session_id,
            bridge=b,
            event_type='simulation_store',
            payload={
                'scenario': simulation.get('scenario', 'unknown') if isinstance(simulation, dict) else 'unknown',
                'confidence': float(confidence),
            },
            hrm_snapshot=hrm_snapshot,
        )
        
        return jsonify({'success': True, 'session_id': session_id, 'retention': retention_report})
    except Exception as e:
        print("=" * 60)
        print("SIMULATION STORE FAILED:")
        traceback.print_exc()
        print("=" * 60)
        return jsonify({'error': f'Simulation store failed: {e}'}), 500


@app.route('/api/atlantean/simulation/recall', methods=['POST'])
def recall_simulations():
    """Recall past simulations."""
    data = request.json or {}
    session_id = _resolve_session_id()
    query = data.get('query', '')
    limit = data.get('limit', 10)

    b = get_bridge_for_session(session_id)
    simulations = b.recall_simulations(query, limit)

    return jsonify({'session_id': session_id, 'simulations': simulations})


@app.route('/api/atlantean/simulation/list', methods=['GET'])
def list_simulations():
    """List all stored simulations directly from Redis (no embedding needed)."""
    session_id = _resolve_session_id()
    limit = int(request.args.get('limit', 50))
    b = get_bridge_for_session(session_id)
    simulations = b.list_all_simulations(limit, session_id=session_id)
    return jsonify({'session_id': session_id, 'simulations': simulations})


@app.route('/api/atlantean/retention/apply', methods=['POST'])
def apply_retention():
    """Apply memory retention controls for the current session on demand."""
    data = request.json or {}
    session_id = _resolve_session_id(data.get('session_id'))
    b = get_bridge_for_session(session_id)
    report = _enforce_memory_retention(session_id, b)
    _save_bridge_to_redis(session_id, b)
    return jsonify({'success': True, 'session_id': session_id, 'retention': report})


@app.route('/api/atlantean/cold/manifests', methods=['GET'])
def list_cold_manifests():
    """List detachable cold-memory manifests for a session."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    include_detached = str(request.args.get('include_detached', 'false')).lower() == 'true'
    b = get_bridge_for_session(session_id)
    manifests = b.list_cold_manifests(include_detached=include_detached)
    return jsonify({'session_id': session_id, 'manifests': manifests})


@app.route('/api/atlantean/cold/manifest/export', methods=['GET'])
def export_cold_manifest():
    """Export a detachable cold-memory manifest."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    manifest_id = request.args.get('manifest_id', 'default')
    b = get_bridge_for_session(session_id)
    exported = b.export_cold_manifest(manifest_id)
    return jsonify({'session_id': session_id, 'manifest': exported})


@app.route('/api/atlantean/cold/manifest/import', methods=['POST'])
def import_cold_manifest():
    """Import a detached cold-memory manifest into a session."""
    data = request.json or {}
    session_id = _resolve_session_id(data.get('session_id'))
    manifest = data.get('manifest')
    if not manifest:
        return jsonify({'error': 'No manifest provided'}), 400

    b = get_bridge_for_session(session_id)
    result = b.import_cold_manifest(manifest)
    return jsonify({'session_id': session_id, 'result': result})


@app.route('/api/atlantean/cold/tombstone', methods=['POST'])
def tombstone_cold_item():
    """Tombstone a cold-memory item instead of hard deleting it."""
    data = request.json or {}
    session_id = _resolve_session_id(data.get('session_id'))
    item_id = data.get('item_id')
    reason = data.get('reason', 'manual')
    if not item_id:
        return jsonify({'error': 'No item_id provided'}), 400

    b = get_bridge_for_session(session_id)
    b.tombstone_cold_item(item_id, reason=reason)
    return jsonify({'success': True, 'session_id': session_id, 'item_id': item_id, 'reason': reason})


# ========== Archive Endpoints ==========

@app.route('/api/atlantean/snapshot', methods=['POST'])
def create_snapshot():
    """Create a labeled snapshot."""
    data = request.json or {}
    label = data.get('label')
    session_id = _resolve_session_id(data.get('session_id'))
    
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    hrm_snapshot = _serialize_hrm_state(hrm)
    snapshot = _index_phase_snapshot(
        session_id=session_id,
        bridge=b,
        label=label,
        hrm_snapshot=hrm_snapshot,
        source='manual',
    )
    checkpoint = _create_signed_checkpoint(
        session_id=session_id,
        bridge=b,
        label=label,
        source='manual_snapshot',
        hrm_snapshot=hrm_snapshot,
    )
    
    return jsonify({'snapshot': snapshot, 'checkpoint': {
        'id': checkpoint.get('id'),
        'seq': checkpoint.get('seq'),
        'checkpoint_hash': checkpoint.get('checkpoint_hash'),
        'signed': bool(checkpoint.get('signature')),
    }})


@app.route('/api/atlantean/snapshots', methods=['GET'])
def list_snapshots():
    """List indexed snapshots (phase-locked memory points) for a session."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    limit = int(request.args.get('limit', 50))
    key = f'{SNAPSHOT_INDEX_PREFIX}{session_id}'
    rows = redis_client.lrange(key, 0, max(limit - 1, 0))
    snapshots = []
    for row in rows:
        parsed = _decode_json_row(row)
        if parsed:
            snapshots.append(parsed)
    return jsonify({'session_id': session_id, 'snapshots': snapshots})


@app.route('/api/atlantean/snapshot/restore', methods=['POST'])
def restore_snapshot():
    """Restore hot-memory state from a previously indexed snapshot."""
    data = request.json or {}
    session_id = _resolve_session_id()
    snapshot_id = data.get('snapshot_id')

    if not snapshot_id:
        return jsonify({'error': 'No snapshot_id provided'}), 400

    records = _load_snapshot_records(session_id)
    target = next((record for record in records if record.get('id') == snapshot_id), None)
    if not target:
        return jsonify({'error': 'Snapshot not found', 'session_id': session_id, 'snapshot_id': snapshot_id}), 404

    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)

    if VERIFY_LEDGER_ON_RESTORE:
        try:
            _enforce_ledger_gate_or_raise(
                session_id=session_id,
                bridge=b,
                context='snapshot_restore_preflight',
                require_replay_match=False,
            )
        except ValueError as e:
            return jsonify({'error': str(e), 'session_id': session_id}), 409

    # Rehydrate deterministic field state from stored snapshot payload.
    try:
        source = None
        snapshot_data_key = target.get('snapshot_data_key')
        if snapshot_data_key:
            raw_source = redis_client.get(snapshot_data_key)
            if raw_source:
                decoded = raw_source.decode('utf-8') if isinstance(raw_source, bytes) else raw_source
                source = json.loads(decoded)
        if not isinstance(source, dict):
            return jsonify({'error': 'Snapshot payload is unavailable for restore'}), 400

        if VERIFY_SNAPSHOT_TAMPER_EVIDENCE:
            expected_hash = str(target.get('snapshot_sha256') or '')
            if not expected_hash:
                return jsonify({
                    'error': 'Snapshot tamper-evidence hash is missing',
                    'session_id': session_id,
                    'snapshot_id': snapshot_id,
                }), 409
            if not verify_snapshot_hash(source, expected_hash):
                _emit_security_policy_event('snapshot_tamper_detected', {
                    'session_id': session_id,
                    'snapshot_id': snapshot_id,
                    'reason_code': 'SNAPSHOT_HASH_MISMATCH',
                })
                return jsonify({
                    'error': 'Snapshot tamper-evidence verification failed (SNAPSHOT_HASH_MISMATCH)',
                    'session_id': session_id,
                    'snapshot_id': snapshot_id,
                }), 409

        b.hot_memory.phi1 = torch.tensor(source['phi1'], dtype=b.hot_memory.phi1.dtype)
        b.hot_memory.phi5 = torch.tensor(source['phi5'], dtype=b.hot_memory.phi5.dtype)
        b.hot_memory.Phi = torch.tensor([float(source['Phi'])], dtype=b.hot_memory.Phi.dtype)
        b.hot_memory.Theta = source.get('Theta', {}) or {}
        b.hot_memory.version = int(source.get('version', b.hot_memory.version))
        snapshot_ts = source.get('timestamp')
        if isinstance(snapshot_ts, (int, float)):
            b.hot_memory.last_update = float(snapshot_ts)
        b.bridge.hot = b.hot_memory

        # Restore HRM state when available for phase-consistent post-restore behavior.
        restored_hrm_payload = target.get('hrm')
        if isinstance(restored_hrm_payload, dict):
            restored = _restore_hrm_state_from_snapshot(hrm, restored_hrm_payload)
            if not restored:
                print(f"ℹ️  Snapshot {snapshot_id} has legacy HRM payload; keeping current HRM state")

        b.set_hrm_snapshot(hrm.snapshot())
    except Exception as e:
        return jsonify({'error': f'Failed to restore snapshot: {e}'}), 400

    _save_bridge_to_redis(session_id, b)
    _save_hrm_to_redis(session_id, hrm)
    _append_signed_event(
        session_id=session_id,
        bridge=b,
        event_type='snapshot_restore',
        payload={
            'snapshot_id': snapshot_id,
            'snapshot_label': target.get('label'),
        },
        hrm_snapshot=hrm.snapshot(),
    )

    post_verify = None
    if VERIFY_LEDGER_ON_RESTORE:
        post_verify = _enforce_ledger_gate_or_raise(
            session_id=session_id,
            bridge=b,
            context='snapshot_restore_post_commit',
            require_replay_match=False,
        )

    return jsonify({
        'success': True,
        'session_id': session_id,
        'snapshot_id': snapshot_id,
        'status': b.get_status(),
        'ledger_verification': {
            'performed': VERIFY_LEDGER_ON_RESTORE,
            'reason_code': post_verify['decision']['reason_code'] if post_verify else 'LEDGER_VERIFICATION_DISABLED',
        },
        'tamper_evidence': {
            'snapshot_hash_verified': VERIFY_SNAPSHOT_TAMPER_EVIDENCE,
            'snapshot_sha256': target.get('snapshot_sha256'),
        },
    })


@app.route('/api/atlantean/snapshot/delete', methods=['POST'])
def delete_snapshot():
    """Delete a snapshot record from the archive index."""
    data = request.json or {}
    session_id = _resolve_session_id()
    snapshot_id = data.get('snapshot_id')

    if not snapshot_id:
        return jsonify({'error': 'No snapshot_id provided'}), 400

    records = _load_snapshot_records(session_id)
    filtered = [record for record in records if record.get('id') != snapshot_id]
    if len(filtered) == len(records):
        return jsonify({'error': 'Snapshot not found', 'session_id': session_id, 'snapshot_id': snapshot_id}), 404

    _save_snapshot_records(session_id, filtered)
    snapshot_data_key = next(
        (record.get('snapshot_data_key') for record in records if record.get('id') == snapshot_id),
        None,
    )
    if snapshot_data_key:
        redis_client.delete(snapshot_data_key)

    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    _append_signed_event(
        session_id=session_id,
        bridge=b,
        event_type='snapshot_delete',
        payload={
            'snapshot_id': snapshot_id,
        },
        hrm_snapshot=hrm.snapshot(),
    )

    return jsonify({
        'success': True,
        'session_id': session_id,
        'snapshot_id': snapshot_id,
        'remaining': len(filtered),
    })


@app.route('/api/atlantean/checkpoint', methods=['POST'])
def create_checkpoint():
    """Create a signed checkpoint tied to the session event ledger."""
    data = request.json or {}
    session_id = _resolve_session_id(data.get('session_id'))
    label = data.get('label')

    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)
    checkpoint = _create_signed_checkpoint(
        session_id=session_id,
        bridge=b,
        label=label,
        source='manual_checkpoint',
        hrm_snapshot=hrm.snapshot(),
    )
    return jsonify({'checkpoint': checkpoint})


@app.route('/api/atlantean/integrity/verify', methods=['GET'])
def verify_integrity():
    """Verify the signed event/checkpoint chain for a session."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    b = get_bridge_for_session(session_id)
    return jsonify(_verify_ledger_integrity(session_id, b))


@app.route('/api/atlantean/integrity/replay', methods=['GET'])
def replay_integrity_proof():
    """Deterministic one-shot replay proof: event-log state hash vs current live state hash."""
    session_id = _resolve_session_id()
    b = get_bridge_for_session(session_id)
    return jsonify(_deterministic_replay_proof(session_id, b))


@app.route('/api/atlantean/diagnostics/export', methods=['GET'])
def export_diagnostics_bundle():
    """One-click diagnostics bundle for troubleshooting local deployments (Step 27).

    Aggregates routing config, ledger integrity, replay proof, recent telemetry,
    hot-memory stats, HRM state, and runtime environment into a single JSON bundle.
    """
    session_id = _resolve_session_id(request.args.get('session_id'))
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)

    # Routing config (same as /routing/settings).
    routing = {
        'provider': {
            'default': normalize_provider_name(ATLANTEAN_LLM_PROVIDER) or 'local',
            'allow_cloud_escalation': ALLOW_CLOUD_ESCALATION,
            'require_cloud_escalation_consent': REQUIRE_CLOUD_ESCALATION_CONSENT,
            'allow_outbound_network': ALLOW_OUTBOUND_NETWORK,
            'local_backend': LOCAL_LLM_BACKEND,
            'local_model': LOCAL_LLM_MODEL,
            'local_llm_url_configured': bool(LOCAL_LLM_URL),
        },
        'model_tiers': {
            'light': MODEL_TIER_LIGHT,
            'standard': MODEL_TIER_STANDARD,
            'heavy': MODEL_TIER_HEAVY,
            'pressure_threshold': MODEL_TIER_PRESSURE_THRESHOLD,
        },
        'hrm_compute_budget': {
            'enabled': ENABLE_HRM_COMPUTE_BUDGET,
            'hrm_max_ms_per_query': HRM_COMPUTE_MAX_MS_PER_QUERY,
            'rcm_max_ms_per_query': RCM_COMPUTE_MAX_MS_PER_QUERY,
            'max_total_ms_per_window': HRM_COMPUTE_MAX_MS_PER_WINDOW,
            'window_seconds': HRM_COMPUTE_WINDOW_SECONDS,
        },
        'reasoning_budgets': {
            'enabled': ENABLE_QUERY_CLASS_REASONING_BUDGETS,
            'cycles_simple': REASONING_CYCLES_SIMPLE,
            'cycles_analytical': REASONING_CYCLES_ANALYTICAL,
            'cycles_deep': REASONING_CYCLES_DEEP,
            'cycles_file_assisted': REASONING_CYCLES_FILE_ASSISTED,
        },
        'integrity_flags': {
            'verify_ledger_on_restore': VERIFY_LEDGER_ON_RESTORE,
            'verify_ledger_on_sync_merge': VERIFY_LEDGER_ON_SYNC_MERGE,
            'verify_snapshot_tamper_evidence': VERIFY_SNAPSHOT_TAMPER_EVIDENCE,
            'enable_corruption_recovery_repair': ENABLE_CORRUPTION_RECOVERY_REPAIR,
        },
        'retention': {
            'enabled': ENABLE_MEMORY_RETENTION_CONTROLS,
            'chat_turns_max': RETENTION_CHAT_TURNS_MAX,
            'simulations_max': RETENTION_SIMULATIONS_MAX,
            'telemetry_events_max': RETENTION_TELEMETRY_EVENTS_MAX,
        },
    }

    # Ledger integrity report.
    ledger_report: Dict[str, Any] = {}
    replay_proof: Dict[str, Any] = {}
    try:
        ledger_report = _verify_ledger_integrity(session_id, b)
    except Exception as e:
        ledger_report = {'error': str(e)}
    try:
        replay_proof = _deterministic_replay_proof(session_id, b)
    except Exception as e:
        replay_proof = {'error': str(e)}

    # Recent telemetry (bounded to last 50 events across all sessions).
    recent_telemetry: List[Dict[str, Any]] = []
    try:
        telemetry_key = f'{TELEMETRY_LOG_KEY_PREFIX}{session_id}'
        rows = redis_client.lrange(telemetry_key, 0, 49)
        for row in rows:
            try:
                decoded = row.decode('utf-8') if isinstance(row, bytes) else row
                record = json.loads(decoded)
                if isinstance(record, dict):
                    recent_telemetry.append(record)
            except Exception:
                continue
    except Exception:
        recent_telemetry = []

    # Session intelligence stats.
    try:
        hot_stats = b.get_status()
    except Exception as e:
        hot_stats = {'error': str(e)}

    # HRM state snapshot.
    try:
        hrm_snapshot = _serialize_hrm_state(hrm)
    except Exception as e:
        hrm_snapshot = {'error': str(e)}

    # Runtime environment.
    import sys as _sys
    import platform as _platform
    env_info = {
        'python_version': _sys.version,
        'platform': _platform.platform(),
        'redis_available': REDIS_AVAILABLE,
        'gemini_configured': bool(GEMINI_API_KEY),
        'local_llm_url': LOCAL_LLM_URL,
        'provider_capabilities': list(PROVIDER_CAPABILITIES.keys()),
        'hardware_tier': ACTIVE_HARDWARE_TIER.tier_id,
    }

    bundle = {
        'schema_version': 1,
        'bundle_type': 'diagnostics',
        'exported_at': int(time.time() * 1000),
        'session_id': session_id,
        'routing': routing,
        'ledger': ledger_report,
        'replay_proof': replay_proof,
        'hot_memory': hot_stats,
        'hrm': hrm_snapshot,
        'recent_telemetry': recent_telemetry,
        'environment': env_info,
    }
    return jsonify({'bundle': bundle})


@app.route('/api/atlantean/export', methods=['GET'])
def export_intelligence_bundle():
    """Export signed events/checkpoints + current state for user-owned portability."""
    session_id = _resolve_session_id(request.args.get('session_id'))
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)

    event_rows = redis_client.lrange(f'{EVENT_LOG_KEY_PREFIX}{session_id}', 0, -1)
    checkpoint_rows = redis_client.lrange(f'{CHECKPOINT_INDEX_PREFIX}{session_id}', 0, -1)
    events = []
    checkpoints = []

    for row in event_rows:
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            events.append(json.loads(decoded))
        except Exception:
            continue

    for row in checkpoint_rows:
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            checkpoints.append(json.loads(decoded))
        except Exception:
            continue

    bundle = {
        'schema_version': 1,
        'exported_at': int(time.time() * 1000),
        'session_id': session_id,
        'identity': b.identity.to_dict() if b.identity else None,
        'events': events,
        'checkpoints': checkpoints,
        'hot_state_hash': _compute_hot_state_hash(b),
        'hot_state': b.create_snapshot('Export state'),
        'hrm_state': hrm.snapshot(),
    }
    return jsonify({'bundle': bundle})


# ========== Sync Endpoints ==========

@app.route('/api/atlantean/sync/prepare', methods=['GET'])
def prepare_sync():
    """Prepare sync package for multi-device."""
    b = get_bridge()
    
    try:
        package = b.prepare_sync_package()
        return jsonify({'package': package})
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/atlantean/sync/merge', methods=['POST'])
def merge_sync():
    """Merge sync package from another device."""
    data = request.json
    package = data.get('package')
    strategy = data.get('strategy', 'conservative')
    session_id = _resolve_session_id(data.get('session_id'))
    
    if not package:
        return jsonify({'error': 'No sync package'}), 400
    
    b = get_bridge_for_session(session_id)
    hrm = get_hrm_for_session(session_id)

    pre_verify = None
    if VERIFY_LEDGER_ON_SYNC_MERGE:
        try:
            pre_verify = _enforce_ledger_gate_or_raise(
                session_id=session_id,
                bridge=b,
                context='sync_merge_preflight',
                require_replay_match=False,
            )
        except ValueError as e:
            return jsonify({'error': str(e), 'session_id': session_id}), 409
    
    try:
        merge_result = b.merge_from_device(package, strategy=strategy)
        _save_bridge_to_redis(session_id, b)
        _save_hrm_to_redis(session_id, hrm)
        ledger_event = _append_signed_event(
            session_id=session_id,
            bridge=b,
            event_type='sync_merge',
            payload={
                'strategy': strategy,
                'merge_result': merge_result,
            },
            hrm_snapshot=hrm.snapshot(),
        )

        post_verify = None
        if VERIFY_LEDGER_ON_SYNC_MERGE:
            post_verify = _enforce_ledger_gate_or_raise(
                session_id=session_id,
                bridge=b,
                context='sync_merge_post_commit',
                require_replay_match=False,
            )
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'status': b.get_status(),
            'sync': merge_result,
            'ledger': {
                'seq': ledger_event.get('seq'),
                'event_hash': ledger_event.get('event_hash'),
                'signed': bool(ledger_event.get('signature')),
            },
            'ledger_verification': {
                'performed': VERIFY_LEDGER_ON_SYNC_MERGE,
                'preflight_reason_code': pre_verify['decision']['reason_code'] if pre_verify else 'LEDGER_VERIFICATION_DISABLED',
                'reason_code': post_verify['decision']['reason_code'] if post_verify else 'LEDGER_VERIFICATION_DISABLED',
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/atlantean/recovery/repair', methods=['POST'])
def repair_corruption_state():
    """Repair partial snapshot/ledger corruption for a session."""
    if not ENABLE_CORRUPTION_RECOVERY_REPAIR:
        return jsonify({'error': 'Corruption recovery repair is disabled by policy'}), 403

    data = request.json or {}
    session_id = _resolve_session_id(data.get('session_id'))
    verify_snapshot_hashes = bool(data.get('verify_snapshot_hashes', VERIFY_SNAPSHOT_TAMPER_EVIDENCE))

    records = _load_snapshot_records(session_id)
    payload_by_snapshot_id: Dict[str, Dict[str, Any] | None] = {}
    for record in records:
        snapshot_id = str(record.get('id') or '')
        payload = None
        snapshot_data_key = record.get('snapshot_data_key')
        if snapshot_data_key:
            try:
                raw_source = redis_client.get(snapshot_data_key)
                if raw_source:
                    decoded = raw_source.decode('utf-8') if isinstance(raw_source, bytes) else raw_source
                    parsed = json.loads(decoded)
                    payload = parsed if isinstance(parsed, dict) else None
            except Exception:
                payload = None
        payload_by_snapshot_id[snapshot_id] = payload

    snapshot_plan = plan_snapshot_recovery(
        records,
        payload_by_snapshot_id,
        verify_hash=verify_snapshot_hashes,
    )

    prune_ids = set(snapshot_plan.get('prune_snapshot_ids', []))
    kept_records = [record for record in records if str(record.get('id') or '') not in prune_ids]
    removed_payload_keys: List[str] = []
    for record in records:
        if str(record.get('id') or '') in prune_ids:
            payload_key = record.get('snapshot_data_key')
            if payload_key:
                redis_client.delete(payload_key)
                removed_payload_keys.append(payload_key)
    if len(kept_records) != len(records):
        _save_snapshot_records(session_id, kept_records)

    event_rows = redis_client.lrange(f'{EVENT_LOG_KEY_PREFIX}{session_id}', 0, -1)
    events: List[Dict[str, Any]] = []
    ledger_decode_failures = 0
    for row in event_rows:
        try:
            decoded = row.decode('utf-8') if isinstance(row, bytes) else row
            event = json.loads(decoded)
            if isinstance(event, dict):
                events.append(event)
            else:
                ledger_decode_failures += 1
        except Exception:
            ledger_decode_failures += 1

    stored_head_raw = redis_client.get(f'{EVENT_HEAD_HASH_PREFIX}{session_id}')
    stored_head_hash = stored_head_raw.decode('utf-8') if isinstance(stored_head_raw, bytes) else str(stored_head_raw or '')
    stored_seq_raw = redis_client.get(f'{EVENT_SEQ_KEY_PREFIX}{session_id}')
    try:
        stored_seq = int(stored_seq_raw or 0)
    except Exception:
        stored_seq = 0

    ledger_plan = plan_ledger_recovery(
        events,
        stored_head_hash=stored_head_hash,
        stored_seq=stored_seq,
    )

    if ledger_plan.get('head_repair_needed'):
        expected_head = str(ledger_plan.get('expected_head_hash') or '')
        if expected_head:
            redis_client.set(f'{EVENT_HEAD_HASH_PREFIX}{session_id}', expected_head)
        else:
            redis_client.delete(f'{EVENT_HEAD_HASH_PREFIX}{session_id}')

    if ledger_plan.get('seq_repair_needed'):
        redis_client.set(f'{EVENT_SEQ_KEY_PREFIX}{session_id}', int(ledger_plan.get('expected_seq') or 0))

    _emit_security_policy_event('corruption_recovery_repair', {
        'session_id': session_id,
        'snapshots_pruned': len(prune_ids),
        'ledger_head_repaired': bool(ledger_plan.get('head_repair_needed')),
        'ledger_seq_repaired': bool(ledger_plan.get('seq_repair_needed')),
    })

    return jsonify({
        'success': True,
        'session_id': session_id,
        'recovery': {
            'snapshots_total': len(records),
            'snapshots_pruned': len(prune_ids),
            'snapshots_recoverable': len(snapshot_plan.get('recoverable_snapshot_ids', [])),
            'removed_snapshot_payload_keys': removed_payload_keys,
            'snapshot_analysis': snapshot_plan.get('analysis', []),
            'ledger_events_total': len(event_rows),
            'ledger_events_decoded': len(events),
            'ledger_decode_failures': ledger_decode_failures,
            'ledger_head_repaired': bool(ledger_plan.get('head_repair_needed')),
            'ledger_seq_repaired': bool(ledger_plan.get('seq_repair_needed')),
            'ledger_expected_head_hash': ledger_plan.get('expected_head_hash'),
            'ledger_expected_seq': ledger_plan.get('expected_seq'),
        },
    })


# ========== Admin Endpoints ==========

@app.route('/api/atlantean/reset', methods=['POST'])
def reset():
    """Reset intelligence state (for testing)."""
    bridges.pop('default', None)
    hrm_adapters.pop('default', None)
    
    # Delete state from Redis
    try:
        redis_client.delete(f'{HOT_MEMORY_KEY_PREFIX}default')
        redis_client.delete(f'{HRM_STATE_KEY_PREFIX}default')
        redis_client.delete(f'{EVENT_SEQ_KEY_PREFIX}default')
        redis_client.delete(f'{EVENT_HEAD_HASH_PREFIX}default')
        redis_client.delete(f'{EVENT_LOG_KEY_PREFIX}default')
        redis_client.delete(f'{CHECKPOINT_INDEX_PREFIX}default')
    except Exception as e:
        print(f"⚠️  Failed to delete Redis key: {e}")
    
    # Re-initialize
    get_bridge()
    
    return jsonify({
        'success': True,
        'message': 'Intelligence state reset'
    })


@app.route('/api/atlantean/save', methods=['POST'])
def save():
    """Manually save state."""
    b = get_bridge()
    _save_bridge_to_redis('default', b)
    return jsonify({'success': True})


@app.route('/api/atlantean/load', methods=['POST'])
def load():
    """Manually load state."""
    b = get_bridge()
    
    try:
        if not _load_bridge_from_redis('default', b):
            return jsonify({'error': 'No saved state found in backend'}), 404
        return jsonify({
            'success': True,
            'status': b.get_status()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== Local Voice Endpoints (Step 31) ==========

@app.route('/api/atlantean/voice/capabilities', methods=['GET'])
def voice_capabilities():
    """Return availability of local STT (Whisper) and TTS (Piper)."""
    return jsonify({
        'stt': check_whisper_available(),
        'tts': check_piper_available(voice_dir=PIPER_VOICE_DIR, voice=PIPER_VOICE),
        'config': {
            'whisper_model': WHISPER_MODEL_SIZE,
            'whisper_device': WHISPER_DEVICE,
            'whisper_language': WHISPER_LANGUAGE,
            'piper_voice': PIPER_VOICE,
            'piper_voice_dir': PIPER_VOICE_DIR,
        },
        'sovereignty': {
            'allow_browser_speech_fallback': ALLOW_BROWSER_SPEECH_FALLBACK,
            'require_local_stt': REQUIRE_LOCAL_STT,
            'allow_outbound_network': ALLOW_OUTBOUND_NETWORK,
        },
    })


@app.route('/api/atlantean/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio to text using local faster-whisper.

    Accepts either:
    - JSON body  ``{"audio_base64": "<base64>", "mime_type": "audio/webm"}``
    - Multipart form-data with a ``file`` field
    """
    audio_bytes: bytes | None = None

    if request.content_type and 'multipart' in request.content_type:
        f = request.files.get('file') or request.files.get('audio')
        if f is None:
            return jsonify({'error': 'No audio file uploaded'}), 400
        audio_bytes = f.read()
    else:
        body = request.get_json(force=True, silent=True) or {}
        b64 = body.get('audio_base64') or body.get('audio')
        if not b64:
            return jsonify({'error': 'No audio_base64 in request body'}), 400
        try:
            audio_bytes = base64.b64decode(b64)
        except Exception as e:
            return jsonify({'error': f'Invalid base64: {e}'}), 400

    if not audio_bytes:
        return jsonify({'error': 'Empty audio data'}), 400

    try:
        transcriber = _get_transcriber()
        result = transcriber.transcribe(audio_bytes)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({'error': str(e), 'available': False}), 503
    except Exception as e:
        return jsonify({'error': f'Transcription failed: {e}'}), 500


@app.route('/api/atlantean/tts', methods=['POST'])
def local_tts():
    """Synthesise text to WAV audio using local Piper TTS.

    Request body::

        {"text": "Hello world", "voice": "en_US-lessac-medium"}

    Response::

        {
            "audio_base64": "<base64-WAV>",
            "format": "wav",
            "sample_rate": 22050
        }
    """
    body = request.get_json(force=True, silent=True) or {}
    text = str(body.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    # Limit input length to mitigate abuse.
    if len(text) > 4000:
        return jsonify({'error': 'Text exceeds 4000 character limit'}), 400

    voice = str(body.get('voice') or PIPER_VOICE).strip() or PIPER_VOICE
    # Sanitise voice name: only allow alphanumeric, hyphen, underscore, dot.
    import re as _re
    if not _re.match(r'^[\w\-\.]+$', voice):
        return jsonify({'error': 'Invalid voice name'}), 400

    try:
        piper = _get_piper()
        result = piper.synthesize(text, voice=voice)
        return jsonify({
            'audio_base64': result['audio_base64'],
            'format': result['format'],
            'sample_rate': result['sample_rate'],
        })
    except RuntimeError as e:
        return jsonify({'error': str(e), 'available': False}), 503
    except Exception as e:
        return jsonify({'error': f'TTS failed: {e}'}), 500


# ========== Hardware, Runtime, Model & Port Endpoints (Steps 1-10) ==========

_hw_cache: Dict[str, Any] = {}
_hw_cache_ts: float = 0.0
_HW_CACHE_TTL = 30.0  # seconds


@app.route('/api/atlantean/hardware/probe', methods=['GET'])
def hardware_probe():
    """Full hardware probe: CPU, RAM, NVIDIA/AMD/Intel/Apple/NPU, VRAM.

    Results are cached for 30 s to avoid hammering nvidia-smi on every render.
    Use ?refresh=1 to force a fresh probe.
    """
    global _hw_cache, _hw_cache_ts
    refresh = request.args.get('refresh', '0') == '1'
    if not refresh and _hw_cache and (time.time() - _hw_cache_ts) < _HW_CACHE_TTL:
        return jsonify(_hw_cache)
    result = _hw_full_probe()
    _hw_cache = result
    _hw_cache_ts = time.time()
    return jsonify(result)


@app.route('/api/atlantean/runtime/recommend', methods=['POST'])
def runtime_recommend():
    """Return recommended GPU layers, context size and backend args for a model.

    Request body: ``{"model_name": "llama3.1:8b", "backend": "ollama"}``
    """
    data = request.get_json(force=True, silent=True) or {}
    model_name = str(data.get('model_name') or 'llama3.1:8b').strip()
    backend = str(data.get('backend') or 'ollama').strip()
    probe = _hw_full_probe()
    rec = _recommend_runtime(probe, model_name, backend)
    return jsonify({'recommendation': rec, 'probe_summary': {
        'gpus': len(probe.get('gpus', [])),
        'best_vram_mb': probe.get('best_vram_mb'),
        'ram_mb': (probe.get('ram') or {}).get('total_mb'),
        'apple_silicon': bool(probe.get('apple_silicon')),
    }})


@app.route('/api/atlantean/runtime/status', methods=['GET'])
def runtime_status():
    """Return status of all managed inference processes."""
    from atlantean_core.backend_installer import backend_status
    return jsonify({
        'processes': list_processes(),
        'backends': _backend_status(),
    })


@app.route('/api/atlantean/runtime/start', methods=['POST'])
def runtime_start():
    """Start an inference backend.

    Request body::

        {
            "backend": "ollama" | "llama.cpp",
            "port": 11434,
            "model_path": "/path/to/model.gguf",   // llama.cpp only
            "n_gpu_layers": 32,
            "ctx_size": 4096
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    backend = str(data.get('backend') or 'ollama').strip()
    port = int(data.get('port') or (11434 if backend == 'ollama' else 8080))

    if not is_port_free(port):
        # Something already on that port
        return jsonify({'success': True, 'message': f'Port {port} already in use (may already be running)', 'port': port})

    try:
        if backend == 'ollama':
            proc = start_ollama(port=port)
            alive = wait_for_port(port, timeout=15.0)
            return jsonify({'success': alive, 'port': port, 'pid': proc.pid})
        elif backend in ('llama.cpp', 'llamacpp'):
            model_path = str(data.get('model_path') or '')
            if not model_path:
                return jsonify({'error': 'model_path required for llama.cpp'}), 400
            n_gpu = int(data.get('n_gpu_layers') or 0)
            ctx = int(data.get('ctx_size') or 4096)
            proc = start_llamacpp_server(model_path, port=port,
                                         n_gpu_layers=n_gpu, ctx_size=ctx)
            alive = wait_for_port(port, timeout=20.0)
            return jsonify({'success': alive, 'port': port, 'pid': proc.pid})
        else:
            return jsonify({'error': f'Unknown backend: {backend}'}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Failed to start {backend}: {e}'}), 500


@app.route('/api/atlantean/runtime/stop', methods=['POST'])
def runtime_stop():
    """Stop a managed inference process.

    Request body: ``{"name": "ollama"}`` or ``{"name": "llamacpp:8080"}``
    """
    data = request.get_json(force=True, silent=True) or {}
    name = str(data.get('name') or 'ollama').strip()
    ok = stop_process(name)
    return jsonify({'success': ok, 'name': name})


@app.route('/api/atlantean/runtime/install', methods=['POST'])
def runtime_install():
    """Install a backend binary.

    Request body: ``{"backend": "ollama" | "llama.cpp", "platform_key": "linux-cuda-cu12"}``
    """
    data = request.get_json(force=True, silent=True) or {}
    backend = str(data.get('backend') or 'ollama').strip()
    platform_key = data.get('platform_key')
    messages = []

    def _log(msg: str) -> None:
        messages.append(msg)

    if backend == 'ollama':
        result = _install_ollama(on_progress=_log)
    elif backend in ('llama.cpp', 'llamacpp'):
        probe = _hw_full_probe()
        result = _install_llamacpp(
            platform_key=platform_key,
            on_progress=_log,
            gpus=probe.get('gpus', []),
        )
    else:
        return jsonify({'error': f'Unknown backend: {backend}'}), 400

    result['log'] = messages
    return jsonify(result)


@app.route('/api/atlantean/models/list', methods=['GET'])
def models_list():
    """List all locally available models grouped by type."""
    voice_dir = str(request.args.get('voice_dir') or PIPER_VOICE_DIR)
    models = list_all_models(voice_dir=voice_dir)
    counts = {k: len(v) for k, v in models.items()}
    return jsonify({'models': models, 'counts': counts})


@app.route('/api/atlantean/models/pull', methods=['POST'])
def models_pull():
    """Pull an Ollama model and stream progress as Server-Sent Events.

    Request body: ``{"model": "llama3.1:8b-instruct-q4_K_M"}``
    """
    data = request.get_json(force=True, silent=True) or {}
    model_name = str(data.get('model') or '').strip()
    if not model_name:
        return jsonify({'error': 'model name required'}), 400

    import re as _re
    if not _re.match(r'^[\w.:\-/]+$', model_name):
        return jsonify({'error': 'Invalid model name'}), 400

    def _sse_stream():
        for event in pull_ollama_model(model_name):
            yield f"data: {json.dumps(event)}\n\n"

    return app.response_class(
        _sse_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/api/atlantean/models/pull/progress', methods=['GET'])
def models_pull_progress():
    """Return latest progress for an in-flight model pull."""
    model_name = request.args.get('model', '')
    progress = get_pull_progress(model_name)
    if progress is None:
        return jsonify({'status': 'idle', 'model': model_name})
    return jsonify(progress)


@app.route('/api/atlantean/models/import', methods=['POST'])
def models_import():
    """Import a local GGUF file into Ollama.

    Request body: ``{"path": "/absolute/path/to/model.gguf", "name": "my-model"}``
    """
    data = request.get_json(force=True, silent=True) or {}
    path = str(data.get('path') or '').strip()
    name = data.get('name')
    if not path:
        return jsonify({'error': 'path required'}), 400
    result = import_gguf_to_ollama(path, model_name=name)
    return jsonify(result)


@app.route('/api/atlantean/models/delete', methods=['POST'])
def models_delete():
    """Delete an Ollama model.

    Request body: ``{"model": "llama3.1:8b"}``
    """
    data = request.get_json(force=True, silent=True) or {}
    model_name = str(data.get('model') or '').strip()
    if not model_name:
        return jsonify({'error': 'model name required'}), 400
    result = delete_ollama_model(model_name)
    return jsonify(result)


@app.route('/api/atlantean/ports/check', methods=['GET'])
def ports_check():
    """Check if a TCP port is free.

    Query params: ``?port=11434`` or ``?ports=11434,8080``
    """
    single = request.args.get('port')
    multi  = request.args.get('ports', '')

    ports_to_check = []
    if single:
        ports_to_check.append(int(single))
    if multi:
        for p in multi.split(','):
            try:
                ports_to_check.append(int(p.strip()))
            except ValueError:
                pass
    if not ports_to_check:
        return jsonify({'error': 'Provide ?port=N or ?ports=N1,N2'}), 400

    results = {p: {'free': is_port_free(p), 'port': p} for p in ports_to_check}
    return jsonify({'ports': results})


# ========== Privacy Status ==========

@app.route('/api/atlantean/privacy/status', methods=['GET'])
def privacy_status():
    """Authoritative privacy/network status for the frontend indicator."""
    if _SEMANTIC_SEARCH_AVAILABLE:
        from atlantean_core.semantic_search.config import (
            SEMANTIC_SEARCH_ENABLED, EMBEDDING_MODEL, VECTOR_BACKEND, VECTOR_DB_PATH
        )
    else:
        SEMANTIC_SEARCH_ENABLED = False
        EMBEDDING_MODEL = None
        VECTOR_BACKEND = None
        VECTOR_DB_PATH = None
    stt = check_whisper_available()
    return jsonify({
        'outbound_network': ALLOW_OUTBOUND_NETWORK,
        'cloud_escalation': ALLOW_CLOUD_ESCALATION,
        'allow_browser_speech_fallback': ALLOW_BROWSER_SPEECH_FALLBACK,
        'require_local_stt': REQUIRE_LOCAL_STT,
        'active_provider': normalize_provider_name(ATLANTEAN_LLM_PROVIDER) or 'local',
        'stt_engine': 'local-whisper' if stt.get('available') else 'unavailable',
        'embedding_model': EMBEDDING_MODEL if _SEMANTIC_SEARCH_AVAILABLE else None,
        'vector_db_path': VECTOR_DB_PATH if _SEMANTIC_SEARCH_AVAILABLE else None,
        'vector_backend': VECTOR_BACKEND if _SEMANTIC_SEARCH_AVAILABLE else None,
        'journal_dir': JOURNAL_BASE_DIR,
        'automation_roots': _auto_path.get_approved_roots() if _AUTOMATION_AVAILABLE else [],
        'semantic_search_available': _SEMANTIC_SEARCH_AVAILABLE,
        'automation_available': _AUTOMATION_AVAILABLE,
        'network_mode': (
            'offline' if not ALLOW_OUTBOUND_NETWORK
            else ('local-network-only' if not ALLOW_CLOUD_ESCALATION else 'external-access-enabled')
        ),
    })


# ========== Semantic File Search ==========

def _search_unavailable():
    return jsonify({'error': 'Semantic search unavailable — run pip install lancedb sentence-transformers pyarrow'}), 503


@app.route('/api/atlantean/search/status', methods=['GET'])
def search_status():
    if not _SEMANTIC_SEARCH_AVAILABLE:
        return jsonify({'available': False, 'reason': 'dependencies not installed'}), 200
    return jsonify({'available': True, 'stats': _idx_mgr.get_index_stats()})


@app.route('/api/atlantean/search/sources', methods=['GET'])
def search_sources():
    if not _SEMANTIC_SEARCH_AVAILABLE:
        return _search_unavailable()
    return jsonify({'sources': _idx_mgr.list_sources()})


@app.route('/api/atlantean/search/source/add', methods=['POST'])
def search_source_add():
    if not _SEMANTIC_SEARCH_AVAILABLE:
        return _search_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    path = str(body.get('path', '')).strip()
    if not path:
        return jsonify({'error': 'path required'}), 400
    try:
        src = _idx_mgr.add_source(path, collection=body.get('collection', 'default'),
                                  excluded_dirs=body.get('excluded_dirs'))
        return jsonify({'success': True, 'source': src})
    except (ValueError, PermissionError) as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/atlantean/search/source/remove', methods=['POST'])
def search_source_remove():
    if not _SEMANTIC_SEARCH_AVAILABLE:
        return _search_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    path = str(body.get('path', '')).strip()
    removed = _idx_mgr.remove_source(path)
    return jsonify({'success': removed})


@app.route('/api/atlantean/search/index', methods=['POST'])
def search_index():
    if not _SEMANTIC_SEARCH_AVAILABLE:
        return _search_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    path = str(body.get('path', '')).strip()
    if not path:
        return jsonify({'error': 'path required'}), 400
    try:
        job = _idx_mgr.start_index_job(path, force=bool(body.get('force', False)))
        return jsonify({'job': {
            'job_id': job.job_id, 'source_path': job.source_path,
            'status': job.status, 'total_files': job.total_files,
        }})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/atlantean/search/index/cancel', methods=['POST'])
def search_index_cancel():
    if not _SEMANTIC_SEARCH_AVAILABLE:
        return _search_unavailable()
    _idx_mgr.cancel_active_job()
    return jsonify({'success': True})


@app.route('/api/atlantean/search/index/status', methods=['GET'])
def search_index_status():
    if not _SEMANTIC_SEARCH_AVAILABLE:
        return _search_unavailable()
    return jsonify({'job': _idx_mgr.get_active_job()})


@app.route('/api/atlantean/search/query', methods=['POST'])
def search_query():
    if not _SEMANTIC_SEARCH_AVAILABLE:
        return _search_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    query = str(body.get('query', '')).strip()
    if not query:
        return jsonify({'error': 'query required'}), 400
    mode = str(body.get('mode', 'hybrid')).lower()
    limit = min(int(body.get('limit', 10)), 50)
    kwargs = {
        'limit': limit,
        'file_types': body.get('file_types'),
        'collection': body.get('collection'),
        'source_prefix': body.get('source_prefix'),
    }
    if mode == 'semantic':
        results = _search_svc.semantic_search(query, **kwargs)
    elif mode == 'keyword':
        results = _search_svc.keyword_search(query, **kwargs)
    else:
        results = _search_svc.hybrid_search(query, **kwargs)
    return jsonify({'query': query, 'mode': mode, 'results': results, 'count': len(results)})


@app.route('/api/atlantean/search/open', methods=['POST'])
def search_open_file():
    """Open a file in the OS default application (non-blocking)."""
    body = request.get_json(force=True, silent=True) or {}
    path = str(body.get('path', '')).strip()
    if not path:
        return jsonify({'error': 'path required'}), 400
    try:
        real = _ss_validate(path) if _SEMANTIC_SEARCH_AVAILABLE else os.path.realpath(path)
        if not os.path.exists(real):
            return jsonify({'error': 'File not found'}), 404
        import subprocess, sys as _sys
        if _sys.platform == 'win32':
            os.startfile(real)
        elif _sys.platform == 'darwin':
            subprocess.Popen(['open', real])
        else:
            subprocess.Popen(['xdg-open', real])
        return jsonify({'success': True, 'path': real})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/atlantean/search/reveal', methods=['POST'])
def search_reveal_file():
    """Reveal a file in the OS file manager."""
    body = request.get_json(force=True, silent=True) or {}
    path = str(body.get('path', '')).strip()
    if not path:
        return jsonify({'error': 'path required'}), 400
    try:
        real = os.path.realpath(path)
        if not os.path.exists(real):
            return jsonify({'error': 'File not found'}), 404
        import subprocess, sys as _sys
        if _sys.platform == 'win32':
            subprocess.Popen(['explorer', '/select,', real])
        elif _sys.platform == 'darwin':
            subprocess.Popen(['open', '-R', real])
        else:
            subprocess.Popen(['xdg-open', os.path.dirname(real)])
        return jsonify({'success': True, 'path': real})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/atlantean/search/index', methods=['DELETE'])
def search_clear_index():
    if not _SEMANTIC_SEARCH_AVAILABLE:
        return _search_unavailable()
    from atlantean_core.semantic_search import vector_store as _vs
    _vs.clear_all()
    return jsonify({'success': True})


# ========== Journal ==========

def _journal_path_for_date(date_str: str) -> str:
    """Resolve safe journal path for YYYY-MM-DD. Raises ValueError if unsafe."""
    import re
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date_str):
        raise ValueError('Invalid date format, expected YYYY-MM-DD')
    year = date_str[:4]
    dest = os.path.realpath(os.path.join(JOURNAL_BASE_DIR, year, f'{date_str}.md'))
    base = os.path.realpath(JOURNAL_BASE_DIR)
    if not dest.startswith(base + os.sep) and dest != base:
        raise ValueError('Journal path traversal attempt')
    return dest


@app.route('/api/atlantean/journal/append', methods=['POST'])
def journal_append():
    """Append a transcript or note entry to today's (or specified) journal."""
    body = request.get_json(force=True, silent=True) or {}
    date_str = str(body.get('date', '')).strip()
    if not date_str:
        from datetime import date
        date_str = date.today().isoformat()
    try:
        jpath = _journal_path_for_date(date_str)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    title = str(body.get('title', '')).strip()
    content = str(body.get('content', '')).strip()
    tags = body.get('tags', [])
    source_type = str(body.get('source_type', 'note')).strip()

    if not content:
        return jsonify({'error': 'content required'}), 400

    import time as _time
    from datetime import datetime
    timestamp = datetime.now().strftime('%I:%M %p').lstrip('0')
    header_parts = [f'## {timestamp}']
    if title:
        header_parts.append(f'— {title}')
    header = ' '.join(header_parts)

    entry_parts = [header]
    if tags:
        entry_parts.append('Tags: ' + ' '.join(f'#{t.strip("#")}' for t in tags))
    entry_parts.append('')
    entry_parts.append(content)
    entry_parts.append('')
    entry_text = '\n'.join(entry_parts) + '\n'

    os.makedirs(os.path.dirname(jpath), exist_ok=True)
    if not os.path.exists(jpath):
        from datetime import date
        day_header = f'# Journal — {date_str}\n\n'
        with open(jpath, 'w', encoding='utf-8') as f:
            f.write(day_header)
    with open(jpath, 'a', encoding='utf-8') as f:
        f.write(entry_text)

    return jsonify({'success': True, 'path': jpath, 'date': date_str})


@app.route('/api/atlantean/journal/read', methods=['GET'])
def journal_read():
    from datetime import date
    date_str = request.args.get('date', date.today().isoformat())
    try:
        jpath = _journal_path_for_date(date_str)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if not os.path.exists(jpath):
        return jsonify({'date': date_str, 'content': '', 'exists': False})
    with open(jpath, 'r', encoding='utf-8') as f:
        content = f.read()
    return jsonify({'date': date_str, 'content': content, 'exists': True, 'path': jpath})


@app.route('/api/atlantean/journal/list', methods=['GET'])
def journal_list():
    entries = []
    if os.path.isdir(JOURNAL_BASE_DIR):
        for root, _, files in os.walk(JOURNAL_BASE_DIR):
            for fname in sorted(files):
                if fname.endswith('.md') and not fname.startswith('manifest'):
                    fpath = os.path.join(root, fname)
                    entries.append({
                        'filename': fname,
                        'path': fpath,
                        'date': os.path.splitext(fname)[0],
                        'size': os.path.getsize(fpath),
                        'modified': os.path.getmtime(fpath),
                    })
    entries.sort(key=lambda x: x['date'], reverse=True)
    return jsonify({'entries': entries[:60]})


# ========== Automation ==========

def _auto_unavailable():
    return jsonify({'error': 'Automation module unavailable'}), 503


_active_plans: dict = {}   # plan_id → AutomationPlan


@app.route('/api/atlantean/automation/capabilities', methods=['GET'])
def automation_capabilities():
    if not _AUTOMATION_AVAILABLE:
        return jsonify({'available': False}), 200
    return jsonify({
        'available': True,
        'action_types': list(_auto_schemas.ALLOWED_ACTION_TYPES),
        'risk_labels': _auto_schemas.RISK_LABELS,
        'approved_roots': _auto_path.get_approved_roots(),
        'quarantine_count': len(_auto_quarantine.list_quarantine()),
    })


@app.route('/api/atlantean/automation/approved-roots', methods=['GET'])
def automation_get_roots():
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    return jsonify({'roots': _auto_path.get_approved_roots()})


@app.route('/api/atlantean/automation/approved-roots/add', methods=['POST'])
def automation_add_root():
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    path = str(body.get('path', '')).strip()
    if not path:
        return jsonify({'error': 'path required'}), 400
    try:
        real = _auto_path.add_approved_root(path)
        return jsonify({'success': True, 'path': real, 'roots': _auto_path.get_approved_roots()})
    except (ValueError, PermissionError) as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/atlantean/automation/approved-roots/remove', methods=['POST'])
def automation_remove_root():
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    path = str(body.get('path', '')).strip()
    removed = _auto_path.remove_approved_root(path)
    return jsonify({'success': removed, 'roots': _auto_path.get_approved_roots()})


@app.route('/api/atlantean/automation/plan', methods=['POST'])
def automation_plan():
    """Accept a user-provided plan JSON or natural-language request."""
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    session_id = str(body.get('session_id', 'default'))
    try:
        if 'actions' in body:
            plan = _auto_planner.make_plan_from_json(body, session_id)
        else:
            return jsonify({'error': 'actions array required'}), 400
        _active_plans[plan.plan_id] = plan
        token = _auto_approvals.create_token(plan.plan_hash)
        plan.approval_token = token
        _auto_audit.log_plan_created(plan.plan_id, plan.plan_hash, plan.summary,
                                     plan.risk_level, session_id)
        d = plan.to_dict()
        d['approval_token'] = token
        return jsonify({'plan': d})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/atlantean/automation/approve', methods=['POST'])
def automation_approve():
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    plan_id = str(body.get('plan_id', ''))
    token = str(body.get('approval_token', ''))
    plan = _active_plans.get(plan_id)
    if not plan:
        return jsonify({'error': 'Plan not found or expired'}), 404
    valid, reason = _auto_approvals.verify_token(token, plan.plan_hash)
    if not valid:
        return jsonify({'error': f'Approval failed: {reason}'}), 403
    plan.approved = True
    plan.approved_at = time.time()
    _auto_audit.log_plan_approved(plan.plan_id, plan.plan_hash, plan.session_id)
    return jsonify({'success': True, 'plan_id': plan_id})


@app.route('/api/atlantean/automation/reject', methods=['POST'])
def automation_reject():
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    plan_id = str(body.get('plan_id', ''))
    reason = str(body.get('reason', 'operator rejected'))
    plan = _active_plans.pop(plan_id, None)
    if plan:
        _auto_audit.log_plan_rejected(plan.plan_id, plan.plan_hash, plan.session_id, reason)
    return jsonify({'success': True})


@app.route('/api/atlantean/automation/execute', methods=['POST'])
def automation_execute():
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    plan_id = str(body.get('plan_id', ''))
    session_id = str(body.get('session_id', 'default'))
    plan = _active_plans.get(plan_id)
    if not plan:
        return jsonify({'error': 'Plan not found or expired'}), 404
    if not plan.approved:
        return jsonify({'error': 'Plan not approved'}), 403

    results = []
    job_id = str(uuid.uuid4())
    for action in plan.actions:
        _auto_audit.log_execution_started(job_id, plan_id, action.type, session_id)
        result = _auto_exec.execute_action(action, job_id)
        result_hash = hashlib.sha256((result.output or '').encode()).hexdigest()[:16]
        _auto_audit.log_execution_completed(job_id, plan_id, action.type,
                                             result.status, result_hash, session_id)
        for qp in result.files_quarantined:
            _auto_audit.log_file_quarantined('', qp, job_id)
        results.append({
            'action_type': result.action_type, 'status': result.status,
            'output': result.output, 'error': result.error,
            'files_created': result.files_created,
            'files_modified': result.files_modified,
            'files_quarantined': result.files_quarantined,
            'duration_ms': int((result.finished_at - result.started_at) * 1000),
        })

    # Consume plan after execution
    _active_plans.pop(plan_id, None)
    return jsonify({'job_id': job_id, 'results': results})


@app.route('/api/atlantean/automation/jobs', methods=['GET'])
def automation_jobs():
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    return jsonify({'events': _auto_audit.get_recent_events(50)})


@app.route('/api/atlantean/automation/quarantine', methods=['GET'])
def automation_quarantine_list():
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    return jsonify({'items': _auto_quarantine.list_quarantine()})


@app.route('/api/atlantean/automation/quarantine/restore', methods=['POST'])
def automation_quarantine_restore():
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    qpath = str(body.get('quarantine_path', ''))
    try:
        original = _auto_quarantine.restore_quarantined(qpath)
        _auto_audit.log_quarantine_restored(qpath, original)
        return jsonify({'success': True, 'restored_to': original})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/atlantean/automation/quarantine/purge', methods=['POST'])
def automation_quarantine_purge():
    if not _AUTOMATION_AVAILABLE:
        return _auto_unavailable()
    body = request.get_json(force=True, silent=True) or {}
    qpath = str(body.get('quarantine_path', ''))
    _auto_quarantine.purge_quarantined(qpath)
    return jsonify({'success': True})


# ========== Ollama Config (runtime) ==========

@app.route('/api/atlantean/config/ollama', methods=['GET'])
def ollama_config_get():
    """Return the current effective Ollama address (host:port only, user-friendly)."""
    global _runtime_ollama_url
    effective_url = _get_local_llm_url()
    # Derive a friendly host:port from the full URL for display.
    try:
        from urllib.parse import urlparse
        parsed = urlparse(effective_url)
        host_display = parsed.hostname or 'localhost'
        port_display = parsed.port or 11434
        friendly = f'{host_display}:{port_display}'
    except Exception:
        friendly = 'localhost:11434'
    return jsonify({
        'ollama_address': friendly,
        'full_url': effective_url,
        'runtime_override_active': bool(_runtime_ollama_url),
    })


@app.route('/api/atlantean/config/ollama', methods=['POST'])
def ollama_config_set():
    """Update the Ollama address at runtime without restarting the server.

    Accepts ``{"ollama_address": "localhost:11434"}`` — just a host and port,
    no path or protocol required.  The backend constructs the full API URL.
    """
    global _runtime_ollama_url
    body = request.get_json(force=True, silent=True) or {}
    raw = str(body.get('ollama_address') or '').strip()
    if not raw:
        return jsonify({'error': 'ollama_address is required (e.g. "localhost:11434")'}), 400

    # Validate the address doesn't contain anything unexpected.
    import re as _re
    if not _re.match(r'^(https?://)?[\w.\-]+(:\d+)?(/.*)?$', raw):
        return jsonify({'error': 'Invalid address format. Use host:port, e.g. "localhost:11434".'}), 400

    _runtime_ollama_url = raw
    effective_url = _get_local_llm_url()

    # Quick reachability probe.
    reachable = False
    try:
        probe = effective_url.rsplit('/api/', 1)[0] + '/'
        r = requests.get(probe, timeout=3)
        reachable = r.status_code < 500
    except Exception:
        pass

    return jsonify({
        'success': True,
        'ollama_address': raw,
        'full_url': effective_url,
        'reachable': reachable,
        'message': 'Ollama address updated.' if reachable else
                   'Address saved, but Ollama did not respond — make sure it is running.',
    })


# ========== Health Check ==========

@app.route('/api/atlantean/operations/summary', methods=['GET'])
def operations_summary():
    """Aggregated summary for the Operations Center Action Center."""
    session_id = request.args.get('session_id', 'default')
    try:
        bridge = get_bridge()
        status_data = bridge.get_status(session_id)
    except Exception:
        status_data = {}

    redis_ok = REDIS_AVAILABLE
    ollama_ok = False
    try:
        _probe_url = _get_local_llm_url().rsplit('/api/', 1)[0] + '/'
        r = requests.get(_probe_url, timeout=2)
        ollama_ok = r.status_code < 500
    except Exception:
        pass

    return jsonify({
        'session_id': session_id,
        'backend_healthy': True,
        'redis_available': redis_ok,
        'ollama_available': ollama_ok,
        'cold_memory_items': status_data.get('cold_memory_items', 0) if isinstance(status_data, dict) else 0,
        'snapshot_count': 0,
        'provider': normalize_provider_name(ATLANTEAN_LLM_PROVIDER) or 'local',
        'cloud_escalation': ALLOW_CLOUD_ESCALATION,
        'hardware_tier': ACTIVE_HARDWARE_TIER.tier_id,
    })


@app.route('/api/atlantean/privacy/settings', methods=['GET'])
def privacy_settings_get():
    """Read privacy settings (Redis-backed per session, falls back to defaults)."""
    session_id = request.args.get('session_id', 'default')
    key = f'privacy_settings:{session_id}'
    defaults = {
        'localOnlyLock': False,
        'askBeforeExternalModel': True,
        'askBeforeExternalTool': True,
        'telemetryDisabled': False,
        'webAccessDisabled': False,
        'redactPiiBeforeCloud': True,
        'blockedProviders': [],
        'autoDeleteTempFiles': True,
        'encryptExportedMemories': True,
        'sovereignLockdown': False,
    }
    if REDIS_AVAILABLE:
        try:
            raw = redis_client.get(key)
            if raw:
                import json as _json
                saved = _json.loads(raw)
                defaults.update(saved)
        except Exception:
            pass
    return jsonify(defaults)


@app.route('/api/atlantean/privacy/settings', methods=['PATCH'])
def privacy_settings_patch():
    """Persist privacy setting changes."""
    session_id = request.args.get('session_id') or request.json.get('session_id', 'default')
    data = request.json or {}
    data.pop('session_id', None)
    if REDIS_AVAILABLE:
        try:
            import json as _json
            key = f'privacy_settings:{session_id}'
            existing = {}
            raw = redis_client.get(key)
            if raw:
                existing = _json.loads(raw)
            existing.update(data)
            redis_client.set(key, _json.dumps(existing), ex=86400 * 30)
        except Exception:
            pass
    return jsonify({'success': True})


@app.route('/api/atlantean/privacy/lockdown', methods=['POST'])
def privacy_lockdown():
    """Activate sovereign lockdown — blocks all cloud routing."""
    session_id = (request.json or {}).get('session_id', 'default')
    if REDIS_AVAILABLE:
        try:
            import json as _json
            key = f'privacy_settings:{session_id}'
            raw = redis_client.get(key)
            existing = _json.loads(raw) if raw else {}
            existing['sovereignLockdown'] = True
            existing['localOnlyLock'] = True
            redis_client.set(key, _json.dumps(existing), ex=86400 * 30)
        except Exception:
            pass
    return jsonify({'success': True, 'lockdown': True})


@app.route('/api/atlantean/privacy/purge-session', methods=['POST'])
def privacy_purge_session():
    """Delete temporary session data from Redis."""
    session_id = (request.json or {}).get('session_id', 'default')
    purged = 0
    if REDIS_AVAILABLE:
        try:
            patterns = [
                f'temp:*:{session_id}',
                f'ctx:{session_id}:*',
                f'tool_calls:{session_id}:*',
            ]
            for pattern in patterns:
                keys = redis_client.keys(pattern)
                if keys:
                    redis_client.delete(*keys)
                    purged += len(keys)
        except Exception:
            pass
    return jsonify({'success': True, 'purged_keys': purged, 'session_id': session_id})


@app.route('/api/atlantean/tasks/current', methods=['GET'])
def tasks_current():
    """Returns the current active task (stub — no task runner implemented yet)."""
    return jsonify({'task': None, 'queue_depth': 0})


@app.route('/api/atlantean/memory/proposals', methods=['GET'])
def memory_proposals():
    """Returns pending memory proposals (stub — no proposal queue implemented yet)."""
    return jsonify({'proposals': [], 'pending': 0})


# ========== Health Check ==========

@app.route('/health', methods=['GET'])
def health():
    """Shallow health check — does not require auth."""
    return jsonify({'status': 'healthy', 'service': 'Atlantean Backend'})


@app.route('/health/deep', methods=['GET'])
def health_deep():
    """Deep health check — reports subsystem status without auth."""
    import platform as _platform
    import sys as _sys
    redis_ok = REDIS_AVAILABLE
    ollama_ok = False
    try:
        _probe_url = _get_local_llm_url().rsplit('/api/', 1)[0] + '/'
        r = requests.get(_probe_url, timeout=2)
        ollama_ok = r.status_code < 500
    except Exception:
        pass

    voice_caps = {
        'stt': check_whisper_available(),
        'tts': check_piper_available(voice_dir=PIPER_VOICE_DIR, voice=PIPER_VOICE),
    }

    checks = {
        'redis': 'ok' if redis_ok else 'unavailable',
        'ollama': 'ok' if ollama_ok else 'unavailable',
        'whisper_stt': 'ok' if voice_caps['stt'].get('available') else 'unavailable',
        'piper_tts': 'ok' if voice_caps['tts'].get('available') else 'unavailable',
        'auth': 'enabled' if ATLANTEAN_API_TOKEN else 'disabled',
        'bind_host': ATLANTEAN_BIND_HOST,
        'encryption': 'enabled' if os.getenv('ATLANTEAN_ENCRYPTION_ENABLED', '1') == '1' else 'disabled',
        'cloud_escalation': 'disabled' if not ALLOW_CLOUD_ESCALATION else 'enabled',
    }
    overall = 'healthy' if redis_ok else 'degraded'
    return jsonify({
        'status': overall,
        'checks': checks,
        'voice': voice_caps,
        'hardware_tier': ACTIVE_HARDWARE_TIER.tier_id,
        'provider': normalize_provider_name(ATLANTEAN_LLM_PROVIDER) or 'local',
        'python': _sys.version.split()[0],
        'platform': _platform.platform(),
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 ATLANTEAN INTELLIGENCE BACKEND")
    print("="*60)
    print(f"\nStarting server on http://{ATLANTEAN_BIND_HOST}:{ATLANTEAN_BIND_PORT}")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")

    get_bridge()

    # Log full tracebacks for 500 errors (debug=False silences them by default)
    import logging
    import traceback
    @app.errorhandler(500)
    def _log_500(e):
        traceback.print_exc()
        logging.getLogger().error("500 error: %s", e, exc_info=True)
        return {'error': 'Internal server error'}, 500

    app.run(
        host=ATLANTEAN_BIND_HOST,
        port=ATLANTEAN_BIND_PORT,
        debug=False,
    )
