"""
Atlantean-Quadra Bridge

Integration layer connecting Atlantean Intelligence Core 
with Quadra-Seer Intelligence UI/UX.

This module provides:
- Stateless LLM integration
- Learning signal mapping from UI events
- Simulation storage in cold memory
- Field visualization for Neural Archives
- Session management without conversation history
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'atlantean_core'))

import torch
import numpy as np
import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from hot_memory import AtlanteanHotMemory
from vector_cold_memory import VectorColdMemory
from memory_bridge import AtlanteanMemoryBridge
from identity import AtlanteanIdentity
from learning import (
    apply_learning_signal,
    apply_contradiction_signal,
    apply_outcome_signal,
    apply_relevance_signal,
    compute_learning_capacity
)
from sync import AtlanteanSyncEngine, MergeStrategy
from llm_interface import call_llm_with_context
from governance import GovernanceController, GovernancePolicy
from scheduler import ComputeGatingControls, ComputeScheduler
from symbolic_reasoner import SymbolicReasoner


class QuadraLearningEvent:
    """Events from Quadra-Seer that trigger learning signals."""
    USER_CONFIRMATION = "user_confirmation"
    USER_CORRECTION = "user_correction"
    USER_POSITIVE_FEEDBACK = "user_positive_feedback"
    USER_NEGATIVE_FEEDBACK = "user_negative_feedback"
    PREDICTION_SUCCESS = "prediction_success"
    PREDICTION_FAILURE = "prediction_failure"
    SIMULATION_COMPLETE = "simulation_complete"
    VOICE_SESSION_END = "voice_session_end"
    HIGH_ENGAGEMENT = "high_engagement"
    LOW_ENGAGEMENT = "low_engagement"
    HELPFUL_RESPONSE = "helpful_response"
    UNHELPFUL_RESPONSE = "unhelpful_response"
    CLARIFICATION_NEEDED = "clarification_needed"


class AtlanteanQuadraBridge:
    """
    Bridge between Atlantean Core and Quadra-Seer Intelligence.
    
    Replaces Quadra-Seer's state management with Atlantean field dynamics.
    Provides stateless LLM interface while maintaining intelligence continuity.
    """
    
    def __init__(
        self,
        grid_size=(32, 32),
        embedder=None,
        device_id: Optional[str] = None,
        enable_crypto: bool = True
    ):
        """
        Initialize the bridge.
        
        Args:
            grid_size: Size of hot memory field grids
            embedder: Function to embed text to vectors (for cold memory)
            device_id: Unique device identifier
            enable_crypto: Enable cryptographic identity
        """
        # Initialize identity
        self.identity = None
        if enable_crypto:
            try:
                self.identity = AtlanteanIdentity(
                    device_id=device_id or "quadra-seer",
                    metadata={"platform": "quadra-seer"}
                )
            except Exception as e:
                print(f"Warning: Could not initialize identity: {e}")
        
        # Initialize hot memory (intelligence)
        self.hot_memory = AtlanteanHotMemory.initialize(
            grid_size=grid_size,
            identity=self.identity,
            device_id=device_id
        )
        
        # Initialize cold memory (content)
        self.embedder = embedder or self._default_embedder
        self.cold_memory = VectorColdMemory(embedder=self.embedder)
        
        # Create memory bridge
        self.bridge = AtlanteanMemoryBridge(
            self.hot_memory,
            self.cold_memory,
            embedder=self.embedder
        )
        
        # Sync engine (optional)
        self.sync_engine = None
        if self.identity:
            self.sync_engine = AtlanteanSyncEngine(self.identity)

        self.sync_metadata: Dict[str, Any] = {
            'last_merged_device': None,
            'last_merge_strategy': None,
            'last_merged_at': None,
            'conflict_counters': {
                'total_merges': 0,
                'concurrent_merges': 0,
                'remote_updates_applied': 0,
                'noop_merges': 0,
            },
        }

        # Latest HRM snapshot — kept current by callers via set_hrm_snapshot().
        self._hrm_snapshot: Dict[str, Any] = {}

        # Optional symbolic/governance stack (compute-bounded and explicit).
        symbolic_enabled = os.getenv('QUADRA_SYMBOLIC_ENABLED', '0') == '1'
        self.compute_controls = ComputeGatingControls(
            enable_symbolic_reasoning=symbolic_enabled,
            symbolic_every_n_queries=int(os.getenv('QUADRA_SYMBOLIC_EVERY_N_QUERIES', '25')),
            symbolic_max_compute_ms_per_call=float(os.getenv('QUADRA_SYMBOLIC_MAX_COMPUTE_MS', '6.0')),
            symbolic_budget_ms_per_minute=float(os.getenv('QUADRA_SYMBOLIC_BUDGET_MS_PER_MIN', '120.0')),
        )
        self.governance = GovernanceController(
            GovernancePolicy(
                allow_symbolic_reasoning=symbolic_enabled,
                window_seconds=int(os.getenv('QUADRA_GOVERNANCE_WINDOW_SECONDS', '60')),
                max_symbolic_invocations_per_window=int(os.getenv('QUADRA_GOVERNANCE_MAX_INVOCATIONS', '32')),
                max_symbolic_compute_ms_per_window=float(os.getenv('QUADRA_SYMBOLIC_BUDGET_MS_PER_MIN', '120.0')),
                max_single_symbolic_compute_ms=float(os.getenv('QUADRA_SYMBOLIC_MAX_COMPUTE_MS', '6.0')),
            )
        )
        self.scheduler = ComputeScheduler(self.compute_controls)
        self.symbolic_reasoner = SymbolicReasoner()
        self.last_symbolic_decision: Dict[str, Any] = {"ran": False, "reason": "not-run"}

    # ========== HRM Field Coupling ==========

    def set_hrm_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Update the stored HRM snapshot so coupling is always current."""
        self._hrm_snapshot = snapshot or {}

    def _apply_hrm_field_coupling(self) -> Dict[str, float]:
        """
        Isomorphic HRM→field coupling applied from *inside* the bridge.

        Reads the stored HRM snapshot and writes deterministic deltas into
        hot-memory fields (Φ, φ1, φ5) using the same formula as the backend's
        standalone helper, so every internal hot-memory update site is covered
        regardless of whether HRM was stepped at the request-handler level.

        Returns a dict of the applied deltas for logging/ledger payloads.
        """
        snap = self._hrm_snapshot
        coherence = float(snap.get('coherence', 0.0))
        energy    = float(snap.get('energy',    0.0))
        channel   = int(snap.get('channel', 0))
        domain    = int(snap.get('domain',  0))
        layer     = int(snap.get('layer',   0))

        # Global semantic pressure.
        phi_delta = torch.tensor(
            [(coherence * 0.04) - (energy * 0.02)],
            dtype=self.hot_memory.Phi.dtype,
        )
        self.hot_memory.update_Phi(phi_delta)

        # Topology-aware spatial modulation of excitatory (φ1) and plastic (φ5) fields.
        h, w = self.hot_memory.phi1.shape
        row_idx = torch.arange(h, dtype=self.hot_memory.phi1.dtype).unsqueeze(1)
        col_idx = torch.arange(w, dtype=self.hot_memory.phi1.dtype).unsqueeze(0)
        focus = (((row_idx % 4) == (channel % 4)).to(self.hot_memory.phi1.dtype) *
                 ((col_idx % 4) == (domain  % 4)).to(self.hot_memory.phi1.dtype))

        exc_delta     = focus * (((coherence - 0.5) * 0.01) + ((layer - 1.5) * 0.0025))
        plastic_delta = focus * (((0.5 - min(max(energy, 0.0), 1.0)) * 0.008) +
                                 ((domain - 1.5) * 0.0015))

        self.hot_memory.update_phi1(exc_delta)
        self.hot_memory.update_phi5(plastic_delta)

        return {
            'phi_delta':          float(phi_delta.item()),
            'exc_delta_mean':     float(exc_delta.mean().item()),
            'plastic_delta_mean': float(plastic_delta.mean().item()),
        }

    # ========== Core Query Interface ==========
    
    async def query(
        self,
        user_input: str,
        llm_function,
        **llm_kwargs
    ) -> str:
        """
        Process user query through Atlantean-powered LLM.
        
        This replaces Quadra-Seer's chat functions.
        
        Args:
            user_input: User's message/question
            llm_function: Async function that calls LLM API
                         Signature: async (prompt: str, **kwargs) -> str
            **llm_kwargs: Parameters for LLM (temperature, max_tokens, etc.)
        
        Returns:
            LLM response (ephemeral - not stored)
        """
        # Advance scheduler cadence counter regardless of optional module use.
        self.scheduler.tick()

        # Use Atlantean's stateless interface
        response = call_llm_with_context(
            user_input,
            self.hot_memory
        )

        # Optional explicit symbolic layer, compute-gated and governance-controlled.
        # Disabled by default unless QUADRA_SYMBOLIC_ENABLED=1 or force_symbolic=True.
        force_symbolic = bool(llm_kwargs.get('force_symbolic', False))
        context_facts = llm_kwargs.get('symbolic_context_facts')
        self.last_symbolic_decision = self.scheduler.run_symbolic_if_allowed(
            reasoner=self.symbolic_reasoner,
            governance=self.governance,
            query=user_input,
            context_facts=context_facts,
            force=force_symbolic,
        )
        
        # Note: We do NOT store the prompt or response
        # State lives in hot_memory fields only
        
        return response

    def governance_snapshot(self) -> Dict[str, Any]:
        """Return current governance counters for optional modules."""
        return self.governance.snapshot()
    
    # ========== Simulation Integration ==========
    
    def store_simulation(
        self,
        simulation: Dict[str, Any],
        confidence: float = 0.5,
        session_id: Optional[str] = None,
    ):
        """
        Store simulation result in cold memory.
        
        Replaces in-memory simulation storage in Quadra-Seer.
        
        Args:
            simulation: Simulation data (serializable dict)
            confidence: Simulation confidence score
        """
        self.bridge.ingest(
            content=json.dumps(simulation),
            metadata={
                'relevance': confidence,
                'retrieval_weight': confidence,
                'learning_weight': min(1.0, confidence + 0.1),
                'manifest_id': 'simulations',
                'type': 'simulation',
                'timestamp': datetime.now().isoformat(),
                'scenario': simulation.get('scenario', 'unknown'),
                'session_id': session_id,
            }
        )

    def store_chat_turn(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        domain: str = 'chat',
        relevance: float = 0.5,
    ):
        """
        Persist a chat turn into cold memory so recall can work across chats/sessions.
        """
        text = (content or '').strip()
        if not text:
            return

        payload = {
            'role': role,
            'content': text,
            'session_id': session_id,
            'domain': domain,
            'created_at': datetime.now().isoformat(),
        }

        self.bridge.ingest(
            content=json.dumps(payload),
            metadata={
                'relevance': relevance,
                'retrieval_weight': relevance,
                'learning_weight': max(0.0, min(1.0, relevance * 0.9)),
                'manifest_id': f'session:{session_id or "default"}',
                'type': 'chat_turn',
                'role': role,
                'timestamp': datetime.now().isoformat(),
                'scenario': f'{role}_turn',
                'session_id': session_id or 'default',
                'domain': domain,
            }
        )

    def recall_chat_memory(
        self,
        query: str,
        limit: int = 8,
        session_id: Optional[str] = None,
        include_simulations: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Recall memory across sessions/domains using direct Redis scan + lexical scoring.
        This bypasses weak demo embeddings and gives consistent cross-chat memory retrieval.
        """
        query_terms = {t for t in query.lower().split() if len(t) > 2}
        memories: List[Dict[str, Any]] = []

        try:
            for item in self.cold_memory.items.values():
                meta = item.metadata
                if meta.get('tombstone'):
                    continue
                item_type = meta.get('type')
                if item_type not in ('chat_turn', 'simulation'):
                    continue
                if item_type == 'simulation' and not include_simulations:
                    continue

                text = str(item.content)
                lowered = text.lower()
                overlap = sum(1 for t in query_terms if t in lowered) if query_terms else 0

                # Score by lexical overlap, metadata relevance, and session continuity.
                retrieval_weight = float(meta.get('retrieval_weight', meta.get('relevance', 0.2)))
                learning_weight = float(meta.get('learning_weight', meta.get('relevance', 0.2)))
                score = retrieval_weight + float(overlap) * 0.2
                role = str(meta.get('role', '')).lower()
                if role == 'user':
                    score += 0.25
                elif role == 'assistant':
                    score -= 0.05
                if session_id and meta.get('session_id') == session_id:
                    score += 0.25

                ts_raw = meta.get('timestamp', '')
                try:
                    ts_ms = int(datetime.fromisoformat(ts_raw).timestamp() * 1000)
                except Exception:
                    ts_ms = int(datetime.now().timestamp() * 1000)

                memories.append({
                    'type': item_type,
                    'role': role,
                    'score': score,
                    'timestamp': ts_ms,
                    'session_id': meta.get('session_id'),
                    'domain': meta.get('domain', 'chat'),
                    'relevance': float(meta.get('relevance', 0.0)),
                    'retrieval_weight': retrieval_weight,
                    'learning_weight': learning_weight,
                    'content_raw': text,
                })
        except Exception as e:
            print(f'Failed to recall chat memory: {e}')
            return []

        memories.sort(key=lambda m: (m['score'], m['timestamp']), reverse=True)
        selected = memories[:limit]

        # Reinforce hot memory when useful recall exists, tying cold recall into core state.
        # Apply full isomorphic HRM coupling instead of a flat scalar so topology is respected.
        if selected:
            if self._hrm_snapshot:
                self._apply_hrm_field_coupling()
            else:
                self.hot_memory.update_Phi(torch.tensor([min(len(selected), 6) * 0.02]))

        return selected

    def extract_recalled_facts(self, recalled_memories: List[Dict[str, Any]], limit: int = 4) -> List[str]:
        """Extract short, user-centric fact snippets from recalled memories."""
        facts: List[str] = []
        for mem in recalled_memories:
            if mem.get('type') != 'chat_turn':
                continue
            if str(mem.get('role', '')).lower() != 'user':
                continue
            raw = str(mem.get('content_raw', ''))
            try:
                payload = json.loads(raw)
                text = str(payload.get('content', '')).strip()
            except Exception:
                text = raw.strip()
            if not text:
                continue
            # Keep concise and useful for prompt grounding.
            if len(text) > 180:
                text = text[:180] + '...'
            facts.append(text)
            if len(facts) >= limit:
                break
        return facts

    def list_chat_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        """Return chat turns for a session in chronological order."""
        history: List[Dict[str, Any]] = []

        try:
            for item in self.cold_memory.items.values():
                meta = item.metadata
                if meta.get('tombstone'):
                    continue
                if meta.get('type') != 'chat_turn':
                    continue
                if session_id and meta.get('session_id') != session_id:
                    continue

                content_raw = str(item.content).strip()
                if not content_raw:
                    continue

                role = str(meta.get('role', 'assistant')).lower()
                if role not in ('user', 'assistant'):
                    role = 'assistant'

                try:
                    payload = json.loads(content_raw)
                    text = str(payload.get('content', '')).strip() or content_raw
                except Exception:
                    text = content_raw

                ts_raw = meta.get('timestamp', '')
                try:
                    ts_ms = int(datetime.fromisoformat(ts_raw).timestamp() * 1000)
                except Exception:
                    ts_ms = int(datetime.now().timestamp() * 1000)

                history.append({
                    'id': item.id,
                    'role': role,
                    'content': text,
                    'timestamp': ts_ms,
                    'session_id': meta.get('session_id', 'default'),
                })
        except Exception as e:
            print(f'Failed to list chat history: {e}')
            return []

        history.sort(key=lambda m: m.get('timestamp', 0))
        if limit > 0:
            history = history[-limit:]
        return history
    
    def list_all_simulations(
        self,
        limit: int = 50,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all stored simulations directly from Redis without embedding/search.
        This is reliable regardless of the embedder quality.
        """
        simulations = []
        try:
            redis_conn = self.cold_memory.redis
            item_keys = redis_conn.keys('item:*')
            for key in item_keys:
                raw = redis_conn.get(key)
                if not raw:
                    continue
                item_dict = json.loads(raw)
                meta = item_dict.get('metadata', {})
                if meta.get('tombstone'):
                    continue
                if meta.get('type') != 'simulation':
                    continue
                stored_session_id = meta.get('session_id')
                if session_id and stored_session_id and stored_session_id != session_id:
                    continue
                try:
                    sim = json.loads(item_dict['content'])
                    ts_raw = meta.get('timestamp', '')
                    try:
                        from datetime import datetime as _dt
                        ts_ms = int(_dt.fromisoformat(ts_raw).timestamp() * 1000)
                    except Exception:
                        ts_ms = int(datetime.now().timestamp() * 1000)
                    simulations.append({
                        'scenario': meta.get('scenario', sim.get('scenario', 'unknown')),
                        'confidence': meta.get('retrieval_weight', meta.get('relevance', 0.5)),
                        'learning_effect': meta.get('learning_weight', meta.get('relevance', 0.5)),
                        'timestamp': ts_ms,
                        'session_id': stored_session_id,
                        'content': sim,
                    })
                except Exception:
                    pass
        except Exception as e:
            print(f'Redis unavailable, listing simulations from memory: {e}')
            # Fall back to in-memory items
            for item_id, item in self.cold_memory.items.items():
                meta = item.metadata
                if meta.get('tombstone'):
                    continue
                if meta.get('type') != 'simulation':
                    continue
                stored_session_id = meta.get('session_id')
                if session_id and stored_session_id and stored_session_id != session_id:
                    continue
                try:
                    sim = json.loads(item.content) if isinstance(item.content, str) else item.content
                    ts_raw = meta.get('timestamp', '')
                    try:
                        from datetime import datetime as _dt
                        ts_ms = int(_dt.fromisoformat(ts_raw).timestamp() * 1000)
                    except Exception:
                        ts_ms = int(datetime.now().timestamp() * 1000)
                    simulations.append({
                        'scenario': meta.get('scenario', sim.get('scenario', 'unknown') if isinstance(sim, dict) else 'unknown'),
                        'confidence': meta.get('retrieval_weight', meta.get('relevance', 0.5)),
                        'learning_effect': meta.get('learning_weight', meta.get('relevance', 0.5)),
                        'timestamp': ts_ms,
                        'session_id': stored_session_id,
                        'content': sim,
                    })
                except Exception:
                    pass
        # Sort newest first
        simulations.sort(key=lambda x: x['timestamp'], reverse=True)
        return simulations[:limit]

    def recall_simulations(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve past simulations from cold memory.
        
        Args:
            query: Natural language query
            limit: Max results to return
        
        Returns:
            List of simulation dicts
        """
        results = self.bridge.recall(query)
        # bridge.recall() already applies learning-pressure to Φ; now additionally
        # apply topology-aware HRM coupling so φ1/φ5 are also modulated here.
        if self._hrm_snapshot:
            self._apply_hrm_field_coupling()

        simulations = []
        for match in results[:limit]:
            item = match.item
            if item.metadata.get('type') == 'simulation':
                try:
                    sim = json.loads(item.content)
                    # Normalize to the shape the UI expects:
                    # { scenario, confidence, timestamp (ms number), ...rest }
                    ts_raw = item.metadata.get('timestamp', '')
                    try:
                        from datetime import datetime as _dt
                        ts_ms = int(_dt.fromisoformat(ts_raw).timestamp() * 1000)
                    except Exception:
                        ts_ms = int(datetime.now().timestamp() * 1000)

                    simulations.append({
                        'scenario': item.metadata.get('scenario', sim.get('scenario', 'unknown')),
                        'confidence': float(match.retrieval_score),
                        'learning_effect': float(match.learning_effect_score),
                        'timestamp': ts_ms,
                        'content': sim,
                    })
                except Exception:
                    pass
        
        return simulations

    # ========== Cold Memory Normalization ==========

    def list_cold_manifests(self, include_detached: bool = False) -> List[Dict[str, Any]]:
        """List detachable cold-memory manifests."""
        return self.cold_memory.list_manifests(include_detached=include_detached)

    def export_cold_manifest(self, manifest_id: str) -> Dict[str, Any]:
        """Export a detachable cold-memory manifest."""
        return self.bridge.export_manifest(manifest_id)

    def import_cold_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Import and attach a previously detached cold-memory manifest."""
        return self.bridge.import_manifest(manifest)

    def tombstone_cold_item(self, item_id: str, reason: str = 'manual'):
        """Tombstone an item without hard deletion for lineage/audit continuity."""
        self.bridge.detach_item(item_id, reason=reason)
    
    # ========== Neural Archives Integration ==========
    
    def get_field_visualization_data(self) -> Dict[str, Any]:
        """
        Get field state for Neural Archives visualization.
        
        Returns data suitable for Quadra-Seer's visualization components.
        """
        return {
            'phi1': self.hot_memory.phi1.tolist(),
            'phi5': self.hot_memory.phi5.tolist(),
            'Phi': float(self.hot_memory.Phi.item()),
            'version': self.hot_memory.version,
            'timestamp': self.hot_memory.last_update,
            'learning_capacity': float(compute_learning_capacity(self.hot_memory)),
            'stats': {
                'phi1_mean': float(self.hot_memory.phi1.mean()),
                'phi1_std': float(self.hot_memory.phi1.std()),
                'phi5_mean': float(self.hot_memory.phi5.mean()),
                'phi5_std': float(self.hot_memory.phi5.std()),
            }
        }
    
    def create_snapshot(self, label: str = None) -> Dict[str, Any]:
        """
        Create a labeled snapshot for Neural Archives.
        
        Args:
            label: Optional label for this snapshot
        
        Returns:
            Snapshot data with metadata
        """
        raw_snapshot = self.hot_memory.snapshot()
        snapshot = {
            'phi1': raw_snapshot['phi1'].tolist(),
            'phi5': raw_snapshot['phi5'].tolist(),
            'Phi': float(raw_snapshot['Phi'].item()),
            'Theta': raw_snapshot['Theta'],
            'device_id': raw_snapshot['device_id'],
            'version': raw_snapshot['version'],
            'timestamp': raw_snapshot['timestamp'],
            'identity_fingerprint': raw_snapshot['identity_fingerprint'],
            'schema_version': raw_snapshot['schema_version'],
        }
        snapshot['label'] = label or f"Snapshot v{self.hot_memory.version}"
        snapshot['created_at'] = datetime.now().isoformat()
        
        return snapshot
    
    # ========== Learning Signal Mapping ==========
    
    def on_event(self, event: str, **event_data):
        """
        Handle Quadra-Seer UI events and apply appropriate learning signals.
        
        This is how user interactions train the intelligence.
        
        Args:
            event: QuadraLearningEvent constant
            **event_data: Event-specific data
        """
        if event == QuadraLearningEvent.USER_CONFIRMATION:
            # User explicitly confirmed response was good
            apply_learning_signal(self.hot_memory, signal_strength=0.6)
        
        elif event == QuadraLearningEvent.USER_CORRECTION:
            # User corrected or rejected response
            apply_contradiction_signal(self.hot_memory)
        
        elif event == QuadraLearningEvent.PREDICTION_SUCCESS:
            # Prediction matched reality
            apply_outcome_signal(
                self.hot_memory,
                predicted=True,
                actual=True
            )
        
        elif event == QuadraLearningEvent.PREDICTION_FAILURE:
            # Prediction was wrong
            apply_outcome_signal(
                self.hot_memory,
                predicted=True,
                actual=False
            )
        
        elif event == QuadraLearningEvent.SIMULATION_COMPLETE:
            # Simulation ran successfully
            accuracy = event_data.get('accuracy', 0.5)
            apply_learning_signal(self.hot_memory, signal_strength=accuracy)
        
        elif event == QuadraLearningEvent.VOICE_SESSION_END:
            # Voice interaction completed
            engagement = event_data.get('engagement', 0.5)
            apply_relevance_signal(self.hot_memory, relevance_score=engagement)
        
        elif event == QuadraLearningEvent.HIGH_ENGAGEMENT:
            # User highly engaged (time spent, interactions, etc.)
            apply_learning_signal(self.hot_memory, signal_strength=0.4)
        
        elif event == QuadraLearningEvent.LOW_ENGAGEMENT:
            # User disengaged quickly
            apply_learning_signal(self.hot_memory, signal_strength=-0.2)
        
        elif event == QuadraLearningEvent.USER_POSITIVE_FEEDBACK:
            # User gave positive feedback (thumbs up, etc.)
            apply_learning_signal(self.hot_memory, signal_strength=0.5)
        
        elif event == QuadraLearningEvent.USER_NEGATIVE_FEEDBACK:
            # User gave negative feedback
            apply_learning_signal(self.hot_memory, signal_strength=-0.3)
        
        elif event == QuadraLearningEvent.HELPFUL_RESPONSE:
            # Response was marked as helpful
            apply_learning_signal(self.hot_memory, signal_strength=0.4)
        
        elif event == QuadraLearningEvent.UNHELPFUL_RESPONSE:
            # Response was marked as unhelpful
            apply_learning_signal(self.hot_memory, signal_strength=-0.4)
        
        elif event == QuadraLearningEvent.CLARIFICATION_NEEDED:
            # User needed clarification
            apply_learning_signal(self.hot_memory, signal_strength=-0.1)
    
    # ========== Persistence ==========
    
    def save_state(self, path: str):
        """
        Save complete intelligence state.
        
        Replaces Quadra-Seer's session storage.
        """
        if self.identity:
            self.hot_memory.save(path, identity=self.identity)
        else:
            self.hot_memory.save(path)
    
    def load_state(self, path: str):
        """
        Load intelligence state from previous session.
        """
        if self.identity:
            self.hot_memory = AtlanteanHotMemory.load(
                path,
                verify_identity=self.identity
            )
        else:
            self.hot_memory = AtlanteanHotMemory.load(path)
        
        # Reconnect bridge
        self.bridge.hot = self.hot_memory
    
    # ========== Multi-Device Sync ==========
    
    def prepare_sync_package(self) -> Dict[str, Any]:
        """
        Prepare state for sync to other devices.
        
        Enables Quadra-Seer instances on multiple devices to share intelligence.
        """
        if not self.sync_engine:
            raise RuntimeError("Sync requires cryptographic identity")
        
        return self.sync_engine.prepare_sync_package(self.hot_memory)
    
    def merge_from_device(
        self,
        sync_package: Dict[str, Any],
        strategy: MergeStrategy = MergeStrategy.CONSERVATIVE
    ):
        """
        Merge intelligence from another device.
        
        Args:
            sync_package: Package from prepare_sync_package()
            strategy: How to resolve conflicts
        """
        if not self.sync_engine:
            raise RuntimeError("Sync requires cryptographic identity")

        if isinstance(strategy, str):
            strategy = MergeStrategy(strategy)
        
        merged = self.sync_engine.merge(
            self.hot_memory,
            sync_package,
            strategy=strategy
        )

        merge_details = getattr(self.sync_engine, 'last_merge_details', {}) or {}
        action = merge_details.get('action', 'noop')

        counters = self.sync_metadata['conflict_counters']
        counters['total_merges'] = int(counters.get('total_merges', 0)) + 1
        if action == 'concurrent_merge':
            counters['concurrent_merges'] = int(counters.get('concurrent_merges', 0)) + 1
        elif action == 'remote_newer_applied':
            counters['remote_updates_applied'] = int(counters.get('remote_updates_applied', 0)) + 1
        else:
            counters['noop_merges'] = int(counters.get('noop_merges', 0)) + 1

        self.sync_metadata['last_merged_device'] = merge_details.get('remote_device')
        self.sync_metadata['last_merge_strategy'] = merge_details.get('strategy', strategy.value)
        self.sync_metadata['last_merged_at'] = merge_details.get('timestamp') or datetime.now(timezone.utc).replace(tzinfo=None).isoformat()  # lazy default + naive-UTC (utcnow deprecated)
        
        self.hot_memory = merged
        self.bridge.hot = merged

        return {
            **merge_details,
            'sync_metadata': self.sync_metadata,
        }
    
    # ========== Utilities ==========
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get complete system status for dashboard/debugging.
        """
        return {
            'device_id': self.hot_memory.device_id,
            'version': self.hot_memory.version,
            'last_update': datetime.fromtimestamp(self.hot_memory.last_update).isoformat(),
            'learning_capacity': float(compute_learning_capacity(self.hot_memory)),
            'cold_memory_items': len(self.cold_memory.items),
            'fingerprint': self.identity.fingerprint() if self.identity else None,
            'field_stats': {
                'phi1_mean': float(self.hot_memory.phi1.mean()),
                'phi5_mean': float(self.hot_memory.phi5.mean()),
                'Phi': float(self.hot_memory.Phi.item())
            },
            'sync': self.sync_metadata,
        }
    
    @staticmethod
    def _default_embedder(text: str) -> np.ndarray:
        """Simple embedder for demo/testing."""
        digest = hashlib.sha256(str(text).encode('utf-8')).digest()
        seed = int.from_bytes(digest[:4], byteorder='big', signed=False)
        np.random.seed(seed)
        return np.random.randn(128)


# ========== React/TypeScript Wrapper Helpers ==========

def create_react_hooks_example():
    """
    Example of how to use this in React components.
    
    This would be implemented in TypeScript in the actual Quadra-Seer app.
    """
    example = """
    // useAtlanteanBridge.ts
    import { useState, useEffect } from 'react';
    import { AtlanteanQuadraBridge } from './atlantean-bridge';
    
    export function useAtlanteanBridge() {
        const [bridge] = useState(() => new AtlanteanQuadraBridge());
        const [status, setStatus] = useState(null);
        
        useEffect(() => {
            // Load persisted state on mount
            bridge.loadState('user_intelligence.bin').catch(() => {
                // First time, no state exists yet
            });
            
            // Save state periodically
            const interval = setInterval(() => {
                bridge.saveState('user_intelligence.bin');
            }, 60000); // Every minute
            
            return () => clearInterval(interval);
        }, []);
        
        const query = async (input: string, llmService) => {
            const response = await bridge.query(input, llmService.complete);
            return response;
        };
        
        const onUserFeedback = (type: 'positive' | 'negative') => {
            if (type === 'positive') {
                bridge.onEvent('user_confirmation');
            } else {
                bridge.onEvent('user_correction');
            }
        };
        
        return { bridge, query, onUserFeedback, status };
    }
    
    // In a component:
    function ChatInterface() {
        const { query, onUserFeedback } = useAtlanteanBridge();
        
        const handleSend = async (message: string) => {
            const response = await query(message, geminiService);
            // Display response
            // Note: No conversation history stored!
        };
        
        return (
            <div>
                <ChatMessages />
                <FeedbackButtons 
                    onPositive={() => onUserFeedback('positive')}
                    onNegative={() => onUserFeedback('negative')}
                />
            </div>
        );
    }
    """
    return example
