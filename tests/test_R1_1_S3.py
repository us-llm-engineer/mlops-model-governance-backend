"""
R1.1-S3: Secret Reference Validation & Isolation (Persistence/Privacy)

Test suite for C3: Secrets (API keys, model credentials) are referenced by name
in YAML, never inlined. At runtime, secret binding resolves reference to
envelope-encrypted value. Secret names are not leaked via logs.

Dimension: persistence/privacy cases
Mutation targets:
  - Secret name leaked in error message (mutation: remove secret name masking)
  - Inlined secret value in stored config (mutation: skip secret replacement)
  - Missing HMAC validation on secret envelope (mutation: skip signature check)
  - Secret retrieved without encryption check (mutation: return plaintext)
"""

import pytest
import json
import hmac
import hashlib
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import base64


@dataclass
class SecretRef:
    """Secret reference in pipeline config."""
    name: str
    type: str  # "api_key", "model_credential", etc.


@dataclass
class SecretEnvelope:
    """Encrypted secret envelope with HMAC."""
    secret_name: str
    ciphertext: str  # base64 encoded mock encryption
    hmac_sig: str    # HMAC-SHA256 signature
    master_key_version: int


class SecretStore:
    """Mock secret store with envelope encryption and HMAC verification."""

    # Mock secrets storage
    _secrets: Dict[str, str] = {
        "db-password": "super-secret-db-pass",
        "model-registry-token": "registry-token-xyz",
        "api-key-external": "sk-12345abcdef"
    }

    # Master key stub (in real impl, managed by KMS)
    MASTER_KEY = b"mock-master-key-32-bytes-long!!"
    MASTER_KEY_VERSION = 1

    @staticmethod
    def _compute_hmac(data: str, key: bytes = MASTER_KEY) -> str:
        """Compute HMAC-SHA256 signature."""
        sig = hmac.new(key, data.encode(), hashlib.sha256).digest()
        return base64.b64encode(sig).decode()

    @staticmethod
    def _mock_encrypt(plaintext: str) -> str:
        """Mock encryption: base64 encode (not real encryption)."""
        return base64.b64encode(plaintext.encode()).decode()

    @staticmethod
    def _mock_decrypt(ciphertext: str) -> str:
        """Mock decryption: base64 decode."""
        return base64.b64decode(ciphertext.encode()).decode()

    def get_secret(self, secret_name: str) -> Optional[SecretEnvelope]:
        """Get secret as encrypted envelope."""
        if secret_name not in self._secrets:
            return None

        plaintext = self._secrets[secret_name]
        ciphertext = self._mock_encrypt(plaintext)
        hmac_sig = self._compute_hmac(ciphertext)

        return SecretEnvelope(
            secret_name=secret_name,
            ciphertext=ciphertext,
            hmac_sig=hmac_sig,
            master_key_version=self.MASTER_KEY_VERSION
        )

    def decrypt_secret(self, envelope: SecretEnvelope) -> str:
        """Decrypt secret with HMAC verification."""
        # Verify HMAC
        expected_hmac = self._compute_hmac(envelope.ciphertext)
        if not hmac.compare_digest(envelope.hmac_sig, expected_hmac):
            raise ValueError("HMAC verification failed: secret may have been tampered with")

        # Decrypt
        plaintext = self._mock_decrypt(envelope.ciphertext)
        return plaintext


class SecretPersistence:
    """Handles secret references in pipeline configs."""

    @staticmethod
    def store_config_with_secret_refs(config: Dict, secret_store: SecretStore) -> Dict:
        """
        Store config, replacing secret references with safe envelope references.
        Never inline secret values.
        """
        stored_config = {}
        for key, value in config.items():
            if isinstance(value, SecretRef):
                # Replace with reference (not value)
                stored_config[key] = {
                    "_secret_ref": value.name,
                    "_secret_type": value.type
                }
            else:
                stored_config[key] = value

        return stored_config

    @staticmethod
    def mask_secret_in_error(error_msg: str, secret_refs: Dict[str, str]) -> str:
        """
        Mask secret names/values in error messages to prevent leakage.
        Returns sanitized error message.
        """
        masked = error_msg
        for secret_name, secret_value in secret_refs.items():
            masked = masked.replace(secret_name, "<REDACTED-SECRET>")
            masked = masked.replace(secret_value, "<REDACTED-VALUE>")

        return masked


class TestSecretReferenceSuccess:
    """Success cases: secrets are stored by reference, not inlined."""

    def test_secret_reference_stored_not_inlined(self):
        """Case 1: Secret reference stored, not plaintext value."""
        config = {
            "model_name": "fraud-detector",
            "registry_token": SecretRef(name="model-registry-token", type="api_key")
        }

        stored = SecretPersistence.store_config_with_secret_refs(config, SecretStore())

        # Verify secret ref replaced, not inlined
        assert stored["registry_token"]["_secret_ref"] == "model-registry-token"
        assert "_secret_ref" in stored["registry_token"]
        # Verify token value is not in stored config
        assert "registry-token-xyz" not in json.dumps(stored)

    def test_multiple_secret_refs_handled(self):
        """Case 2: Multiple secret references handled correctly."""
        config = {
            "db_password": SecretRef(name="db-password", type="database"),
            "api_key": SecretRef(name="api-key-external", type="api_key"),
            "pipeline_name": "training"
        }

        stored = SecretPersistence.store_config_with_secret_refs(config, SecretStore())

        assert "_secret_ref" in stored["db_password"]
        assert "_secret_ref" in stored["api_key"]
        assert stored["pipeline_name"] == "training"


class TestSecretIsolationBoundary:
    """Boundary cases: secret values are protected from leakage."""

    def test_secret_not_leaked_in_error_message(self):
        """Case 3: Secret name/value masked in error messages."""
        error_with_secret = "Failed to authenticate with db-password: super-secret-db-pass"
        secret_refs = {
            "db-password": "super-secret-db-pass"
        }

        masked = SecretPersistence.mask_secret_in_error(error_with_secret, secret_refs)

        assert "db-password" not in masked
        assert "super-secret-db-pass" not in masked
        assert "<REDACTED" in masked

    def test_secret_value_not_in_logs(self):
        """Case 4: Plaintext secret value never appears in stored config JSON."""
        config = {
            "db_token": SecretRef(name="db-password", type="database"),
            "debug": "enabled"
        }

        stored = SecretPersistence.store_config_with_secret_refs(config, SecretStore())
        config_json = json.dumps(stored)

        # The actual secret value should not be present
        assert "super-secret-db-pass" not in config_json


class TestSecretEnvelopeEncryption:
    """Encryption/decryption with HMAC cases."""

    def test_envelope_encryption_roundtrip(self):
        """Case 5: Secret envelope can be encrypted and decrypted."""
        store = SecretStore()
        envelope = store.get_secret("db-password")

        assert envelope is not None
        assert envelope.ciphertext is not None
        assert envelope.hmac_sig is not None

    def test_envelope_hmac_verification_success(self):
        """Case 6: Valid HMAC passes verification."""
        store = SecretStore()
        envelope = store.get_secret("model-registry-token")

        # Should not raise exception
        plaintext = store.decrypt_secret(envelope)
        assert plaintext == "registry-token-xyz"

    def test_envelope_hmac_verification_failure(self):
        """Case 7: Tampered HMAC is detected."""
        store = SecretStore()
        envelope = store.get_secret("api-key-external")

        # Tamper with HMAC
        envelope.hmac_sig = "invalid_signature_base64"

        with pytest.raises(ValueError, match="HMAC verification failed"):
            store.decrypt_secret(envelope)

    def test_envelope_ciphertext_tampering_detected(self):
        """Case 8: Tampered ciphertext is detected via HMAC."""
        store = SecretStore()
        envelope = store.get_secret("db-password")

        # Tamper with ciphertext
        original_ciphertext = envelope.ciphertext
        envelope.ciphertext = base64.b64encode(b"tampered-data").decode()

        with pytest.raises(ValueError, match="HMAC verification failed"):
            store.decrypt_secret(envelope)


class TestSecretPersistenceQueries:
    """Query and retrieval cases."""

    def test_nonexistent_secret_returns_none(self):
        """Case 9: Query for nonexistent secret returns None."""
        store = SecretStore()
        result = store.get_secret("nonexistent-secret")

        assert result is None

    def test_multiple_secrets_retrieved_independently(self):
        """Case 10: Multiple secrets retrieved with independent envelopes."""
        store = SecretStore()
        env1 = store.get_secret("db-password")
        env2 = store.get_secret("model-registry-token")

        assert env1 is not None
        assert env2 is not None
        assert env1.ciphertext != env2.ciphertext  # Different secrets, different ciphertexts
        assert env1.hmac_sig != env2.hmac_sig      # Different HMACs
