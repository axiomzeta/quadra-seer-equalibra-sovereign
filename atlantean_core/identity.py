# identity.py
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
import secrets

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: cryptography package not installed. Install with: pip install cryptography")


@dataclass
class AtlanteanIdentity:
    """
    Cryptographic identity for an Atlantean intelligence instance.
    
    This ensures:
    - Each intelligence has a unique, verifiable identity
    - Hot memory state can be signed and verified
    - Multi-device sync can verify authentic updates
    - No impostor cores can corrupt the intelligence
    """
    device_id: str
    public_key_pem: str
    created_at: str
    metadata: dict
    
    def __init__(self, device_id: Optional[str] = None, metadata: Optional[dict] = None):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography package required for identity features")
            
        self.device_id = device_id or self._generate_device_id()
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()  # naive-UTC, format-identical to deprecated utcnow()
        
        # Generate keypair
        self._private_key = ed25519.Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
        
        # Serialize public key for storage
        self.public_key_pem = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
    
    @staticmethod
    def _generate_device_id():
        """Generate a unique device identifier."""
        return f"atlantean-{secrets.token_hex(16)}"
    
    def sign(self, data: bytes) -> bytes:
        """
        Sign arbitrary data with this identity's private key.
        
        Args:
            data: Raw bytes to sign
            
        Returns:
            Signature bytes
        """
        return self._private_key.sign(data)
    
    def verify(self, data: bytes, signature: bytes) -> bool:
        """
        Verify a signature against this identity's public key.
        
        Args:
            data: Original data
            signature: Signature to verify
            
        Returns:
            True if signature is valid
        """
        try:
            self._public_key.verify(signature, data)
            return True
        except Exception:
            return False

    def sign_cognition_lineage(self, lineage_hash: str) -> str:
        """Sign a cognitive-lineage hash and return a hex signature."""
        signature = self.sign(str(lineage_hash).encode("utf-8"))
        return signature.hex()

    def verify_cognition_lineage(self, lineage_hash: str, signature_hex: str) -> bool:
        """Verify a cognitive-lineage signature from a hex-encoded signature."""
        try:
            signature = bytes.fromhex(signature_hex)
        except Exception:
            return False
        return self.verify(str(lineage_hash).encode("utf-8"), signature)
    
    def fingerprint(self) -> str:
        """
        Generate a human-readable fingerprint of this identity.
        
        Returns:
            SHA256 hash of the public key (hex)
        """
        return hashlib.sha256(self.public_key_pem.encode()).hexdigest()[:16]
    
    def save_private_key(self, path: str, password: Optional[bytes] = None):
        """
        Save private key to disk (CRITICAL: Keep this secure!).
        
        Args:
            path: File path to save to
            password: Optional encryption password
        """
        if password:
            encryption = serialization.BestAvailableEncryption(password)
        else:
            encryption = serialization.NoEncryption()
            
        pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption
        )
        
        with open(path, 'wb') as f:
            f.write(pem)
    
    @staticmethod
    def load_private_key(path: str, password: Optional[bytes] = None):
        """
        Load private key from disk.
        
        Args:
            path: File path to load from
            password: Optional decryption password
        """
        with open(path, 'rb') as f:
            pem = f.read()
            
        private_key = serialization.load_pem_private_key(
            pem, 
            password=password
        )
        
        # Reconstruct identity (simplified - in production, load full metadata)
        identity = object.__new__(AtlanteanIdentity)
        identity._private_key = private_key
        identity._public_key = private_key.public_key()
        identity.public_key_pem = identity._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        return identity
    
    def to_dict(self) -> dict:
        """Serialize identity metadata (not private key!)."""
        return {
            "device_id": self.device_id,
            "public_key_pem": self.public_key_pem,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "fingerprint": self.fingerprint()
        }
    
    def __repr__(self):
        return f"AtlanteanIdentity(device={self.device_id}, fingerprint={self.fingerprint()})"
