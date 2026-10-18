# sync.py
import torch
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum


class MergeStrategy(Enum):
    """Strategies for resolving conflicts during sync."""
    LAST_WRITE_WINS = "last_write_wins"
    FIELD_AVERAGE = "field_average"
    MAX_ENERGY = "max_energy"
    MAX_PLASTICITY = "max_plasticity"  # Preserve highest plasticity
    CONSERVATIVE = "conservative"  # Weighted blend favoring local
    CUSTOM = "custom"


@dataclass
class SyncMetadata:
    """
    Metadata for tracking sync state across devices.
    
    This enables conflict-free merges using vector clocks and field physics.
    """
    device_id: str
    version: int
    timestamp: str
    vector_clock: Dict[str, int]  # device_id -> logical clock
    signature: Optional[bytes] = None
    
    def to_dict(self):
        return {
            "device_id": self.device_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "vector_clock": self.vector_clock,
            "signature": self.signature.hex() if self.signature else None
        }
    
    @staticmethod
    def from_dict(data):
        return SyncMetadata(
            device_id=data["device_id"],
            version=data["version"],
            timestamp=data["timestamp"],
            vector_clock=data["vector_clock"],
            signature=bytes.fromhex(data["signature"]) if data.get("signature") else None
        )


class AtlanteanSyncEngine:
    """
    Conflict-free synchronization for Atlantean hot memory across devices.
    
    Key principles:
    1. Fields are merged using physics-inspired strategies, not simple overwrites
    2. Vector clocks detect concurrent updates
    3. Energy conservation guides conflict resolution
    4. Identity signatures prevent corruption
    
    Unlike traditional CRDTs, this treats field state as a physical system
    where merges preserve meaningful properties (energy, topology, entropy).
    """
    
    def __init__(self, identity):
        """
        Args:
            identity: AtlanteanIdentity instance for signing updates
        """
        self.identity = identity
        self.device_id = identity.device_id
        self.vector_clock = {self.device_id: 0}
        self.last_merge_details: Dict[str, Any] = {
            "timestamp": None,
            "remote_device": None,
            "strategy": None,
            "concurrent": False,
            "action": "none",
            "remote_version": None,
            "local_version_before": None,
        }
    
    def prepare_sync_package(self, hot_memory):
        """
        Package hot memory state for transmission to another device.
        
        Args:
            hot_memory: AtlanteanHotMemory instance
            
        Returns:
            Dict with state + metadata + signature
        """
        # Increment local clock
        self.vector_clock[self.device_id] += 1
        
        # Create metadata
        metadata = SyncMetadata(
            device_id=self.device_id,
            version=self.vector_clock[self.device_id],
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),  # naive-UTC (utcnow() deprecated)
            vector_clock=self.vector_clock.copy()
        )
        
        # Serialize state
        state_dict = {
            "phi1": hot_memory.phi1.tolist(),
            "phi5": hot_memory.phi5.tolist(),
            "Phi": hot_memory.Phi.tolist(),
            "Theta": hot_memory.Theta,
            "schema_version": hot_memory.schema_version,
            "identity_fingerprint": hot_memory.identity_fingerprint,
            "device_id": hot_memory.device_id,
            "version": hot_memory.version,
            "timestamp": hot_memory.last_update,
            "cognitive_signature": hot_memory.cognitive_signature,
            "cognitive_signature_prev": hot_memory.cognitive_signature_prev,
            "lineage_depth": hot_memory.lineage_depth,
            "lineage_event": hot_memory.lineage_event,
            "lineage_signature_hex": hot_memory.lineage_signature_hex,
        }
        
        # Sign the state
        state_bytes = json.dumps(state_dict, sort_keys=True).encode('utf-8')
        signature = self.identity.sign(state_bytes)
        metadata.signature = signature
        
        return {
            "state": state_dict,
            "metadata": metadata.to_dict()
        }
    
    def merge(self, local_memory, remote_package, strategy=MergeStrategy.FIELD_AVERAGE):
        """
        Merge remote hot memory state with local state using conflict-free semantics.
        
        Args:
            local_memory: AtlanteanHotMemory instance (current device)
            remote_package: Sync package from another device
            strategy: MergeStrategy to use for conflicts
            
        Returns:
            Merged AtlanteanHotMemory instance
        """
        remote_state = remote_package["state"]
        remote_meta = SyncMetadata.from_dict(remote_package["metadata"])

        local_clock_before = self.vector_clock.copy()
        remote_clock = remote_meta.vector_clock
        remote_device = remote_meta.device_id
        remote_version = remote_clock.get(remote_device, 0)
        local_version_before = local_clock_before.get(remote_device, 0)

        is_concurrent = self._is_concurrent(remote_meta, local_clock=local_clock_before)
        
        # Update vector clock (merge clocks)
        for device, version in remote_meta.vector_clock.items():
            current = self.vector_clock.get(device, 0)
            self.vector_clock[device] = max(current, version)

        merge_timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()  # naive-UTC, format-identical to deprecated utcnow()
        action = "noop"
        merged_memory = local_memory
        
        if is_concurrent:
            print(f"Concurrent update detected from {remote_meta.device_id}, merging fields...")
            merged_memory = self._merge_fields(local_memory, remote_state, strategy)
            action = "concurrent_merge"
        else:
            # Check if remote is strictly newer
            if remote_version > local_version_before:
                print(f"Remote is newer, applying update from {remote_device}")
                merged_memory = self._apply_remote(remote_state)
                action = "remote_newer_applied"
            else:
                print("Local is up-to-date, no merge needed")
                merged_memory = local_memory
                action = "noop"

        self.last_merge_details = {
            "timestamp": merge_timestamp,
            "remote_device": remote_device,
            "strategy": strategy.value if isinstance(strategy, MergeStrategy) else str(strategy),
            "concurrent": bool(is_concurrent),
            "action": action,
            "remote_version": int(remote_version),
            "local_version_before": int(local_version_before),
            "vector_clock": self.vector_clock.copy(),
        }

        return merged_memory
    
    def _is_concurrent(self, remote_meta, local_clock=None):
        """
        Check if remote update is concurrent with local state using vector clocks.
        
        Returns:
            True if concurrent (requires merging), False otherwise
        """
        if local_clock is None:
            local_clock = self.vector_clock

        remote_clock = remote_meta.vector_clock
        
        # Check if either happened-before relationship exists
        local_before_remote = all(
            local_clock.get(d, 0) <= remote_clock.get(d, 0)
            for d in set(local_clock.keys()) | set(remote_clock.keys())
        )
        
        remote_before_local = all(
            remote_clock.get(d, 0) <= local_clock.get(d, 0)
            for d in set(local_clock.keys()) | set(remote_clock.keys())
        )
        
        # Concurrent if neither happened before the other
        return not (local_before_remote or remote_before_local)
    
    def _merge_fields(self, local_memory, remote_state, strategy):
        """
        Merge fields using physics-inspired strategies.
        
        This is where the magic happens: we don't just overwrite,
        we blend the fields in ways that preserve meaningful structure.
        """
        remote_phi1 = torch.tensor(remote_state["phi1"])
        remote_phi5 = torch.tensor(remote_state["phi5"])
        remote_Phi = torch.tensor(remote_state["Phi"])
        
        if strategy == MergeStrategy.FIELD_AVERAGE:
            # Average the fields (preserves total energy)
            merged_phi1 = 0.5 * (local_memory.phi1 + remote_phi1)
            merged_phi5 = 0.5 * (local_memory.phi5 + remote_phi5)
            merged_Phi = 0.5 * (local_memory.Phi + remote_Phi)
            
        elif strategy == MergeStrategy.MAX_ENERGY:
            # Take maximum absolute values (preserves strongest signals)
            merged_phi1 = torch.where(
                torch.abs(local_memory.phi1) > torch.abs(remote_phi1),
                local_memory.phi1,
                remote_phi1
            )
            merged_phi5 = torch.maximum(local_memory.phi5, remote_phi5)
            merged_Phi = torch.maximum(local_memory.Phi, remote_Phi)
            
        elif strategy == MergeStrategy.LAST_WRITE_WINS:
            # Simple overwrite (not recommended for fields)
            merged_phi1 = remote_phi1
            merged_phi5 = remote_phi5
            merged_Phi = remote_Phi
            
        elif strategy == MergeStrategy.MAX_PLASTICITY:
            # Preserve highest plasticity (most learning capacity)
            # Decision topology: weighted by plasticity
            # Plasticity: always take maximum
            merged_phi1 = torch.where(
                local_memory.phi5 > remote_phi5,
                local_memory.phi1,
                remote_phi1
            )
            merged_phi5 = torch.maximum(local_memory.phi5, remote_phi5)
            merged_Phi = 0.5 * (local_memory.Phi + remote_Phi)
            
        elif strategy == MergeStrategy.CONSERVATIVE:
            # Conservative merge: favor local, blend remote cautiously
            # This prevents remote updates from destroying local learning
            alpha = 0.3  # Weight for remote contribution
            merged_phi1 = (1 - alpha) * local_memory.phi1 + alpha * remote_phi1
            merged_phi5 = torch.maximum(local_memory.phi5, remote_phi5)  # Never reduce plasticity
            merged_Phi = (local_memory.Phi + remote_Phi) / 2  # Average meaning
            
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        # Merge Theta parameters (dict merge)
        merged_Theta = {**local_memory.Theta, **remote_state["Theta"]}
        
        # Create merged state
        from hot_memory import AtlanteanHotMemory
        return AtlanteanHotMemory(
            phi1=merged_phi1,
            phi5=merged_phi5,
            Phi=merged_Phi,
            Theta=merged_Theta,
            schema_version=max(local_memory.schema_version, remote_state["schema_version"]),
            identity_fingerprint=local_memory.identity_fingerprint or remote_state.get("identity_fingerprint"),
            device_id=local_memory.device_id or remote_state.get("device_id"),
            version=max(int(local_memory.version), int(remote_state.get("version", 0))),
            last_update=max(float(local_memory.last_update), float(remote_state.get("timestamp", 0.0) or 0.0)),
            cognitive_signature=remote_state.get("cognitive_signature", local_memory.cognitive_signature),
            cognitive_signature_prev=remote_state.get("cognitive_signature_prev", local_memory.cognitive_signature_prev),
            lineage_depth=max(int(local_memory.lineage_depth), int(remote_state.get("lineage_depth", 0))),
            lineage_event=f"merge_{strategy.value}",
            lineage_signature_hex=remote_state.get("lineage_signature_hex", local_memory.lineage_signature_hex),
        )
    
    def _apply_remote(self, remote_state):
        """Simply apply remote state (when it's strictly newer)."""
        from hot_memory import AtlanteanHotMemory
        return AtlanteanHotMemory(
            phi1=torch.tensor(remote_state["phi1"]),
            phi5=torch.tensor(remote_state["phi5"]),
            Phi=torch.tensor(remote_state["Phi"]),
            Theta=remote_state["Theta"],
            schema_version=remote_state["schema_version"],
            identity_fingerprint=remote_state.get("identity_fingerprint"),
            device_id=remote_state.get("device_id"),
            version=int(remote_state.get("version", 0)),
            last_update=float(remote_state.get("timestamp", 0.0) or 0.0),
            cognitive_signature=remote_state.get("cognitive_signature", ""),
            cognitive_signature_prev=remote_state.get("cognitive_signature_prev", ""),
            lineage_depth=int(remote_state.get("lineage_depth", 0)),
            lineage_event=remote_state.get("lineage_event", "sync_apply_remote"),
            lineage_signature_hex=remote_state.get("lineage_signature_hex"),
        )
    
    def verify_package(self, package, identity):
        """
        Verify cryptographic signature on a sync package.
        
        Args:
            package: Sync package to verify
            identity: AtlanteanIdentity to verify against
            
        Returns:
            True if signature is valid
        """
        state_bytes = json.dumps(package["state"], sort_keys=True).encode('utf-8')
        signature = bytes.fromhex(package["metadata"]["signature"])
        return identity.verify(state_bytes, signature)


# ========== Simplified Sync Protocol ==========

def merge_hot_memories(local, remote, alpha=0.5):
    """
    Simplified device-to-device merge (CRDT-like, field-safe).
    
    CRITICAL PRINCIPLE: We never overwrite fields. We merge them conservatively.
    
    This function provides a simpler alternative to the full AtlanteanSyncEngine
    for direct device-to-device sync without vector clocks.
    
    Merge rules:
    1. φ₁ (decision topology): Weighted blend
    2. φ₅ (plasticity): Max-preserving (never reduce learning capacity)
    3. Φ (global meaning): Average
    4. Θ (parameters): Additive merge (never delete)
    
    Args:
        local: AtlanteanHotMemory instance (will be modified in-place)
        remote: Dict snapshot from remote device (from snapshot() method)
        alpha: Weight for remote contribution (0=ignore remote, 1=full remote)
        
    Returns:
        None (modifies local in-place)
        
    Note:
        Only applies remote if remote version is newer.
        For full conflict resolution with vector clocks, use AtlanteanSyncEngine.
    """
    if remote["version"] <= local.version:
        # Remote is stale, skip merge
        return
    
    # Convert remote tensors if needed
    remote_phi1 = remote["phi1"] if torch.is_tensor(remote["phi1"]) else torch.tensor(remote["phi1"])
    remote_phi5 = remote["phi5"] if torch.is_tensor(remote["phi5"]) else torch.tensor(remote["phi5"])
    remote_Phi = remote["Phi"] if torch.is_tensor(remote["Phi"]) else torch.tensor(remote["Phi"])
    
    # Merge excitability (decision topology) - weighted blend
    # This preserves both local and remote decision landscapes
    local.phi1 = (1 - alpha) * local.phi1 + alpha * remote_phi1
    
    # Merge plasticity with max-preserving rule
    # CRITICAL: Never reduce learning capacity during sync
    local.phi5 = torch.maximum(local.phi5, remote_phi5)
    
    # Merge global meaning - simple average
    # Both perspectives contribute equally to semantic coherence
    local.Phi = (local.Phi + remote_Phi) / 2
    
    # Merge learned parameters conservatively
    # Add new parameters, keep existing ones (no deletion)
    for k, v in remote["Theta"].items():
        if k not in local.Theta:
            local.Theta[k] = v
        else:
            # If parameter exists in both, average them
            if isinstance(v, (int, float)):
                local.Theta[k] = (local.Theta[k] + v) / 2
    
    # Update metadata
    local.version = remote["version"]
    local.last_update = remote["timestamp"]
    local.apply_local_update()  # Increment version post-merge


def sync_devices_simple(device_a, device_b, alpha=0.5):
    """
    Bidirectional sync between two devices using simple merge.
    
    Both devices end up with the same merged state.
    
    Args:
        device_a: AtlanteanHotMemory instance
        device_b: AtlanteanHotMemory instance
        alpha: Merge weight (0.5 = equal blend)
        
    Note:
        For production use with multiple devices, use AtlanteanSyncEngine
        which provides proper conflict detection via vector clocks.
    """
    snapshot_a = device_a.snapshot()
    snapshot_b = device_b.snapshot()
    
    # Each device merges the other's state
    merge_hot_memories(device_a, snapshot_b, alpha=alpha)
    merge_hot_memories(device_b, snapshot_a, alpha=alpha)
    
    # Both devices should now be at the same version
    max_version = max(snapshot_a["version"], snapshot_b["version"]) + 1
    device_a.version = max_version
    device_b.version = max_version
